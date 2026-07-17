"""Render-and-compare identification (analysis-by-synthesis).

Verify a candidate by SYNTHESIZING the target's 3D model in the query's own
viewpoint and imaging conditions, then comparing like-for-like — instead of
asking a 2D embedding to be viewpoint-invariant (its worst case, and the
reason front/rear/side ReID degrades).

Pipeline:
  1. pose_search — render the model over candidate azimuths/elevations,
     domain-match each to the crop, pick the best. (PnP against a live
     LandmarkStore is used only when one is supplied; the saved asset does
     not persist landmarks, so pose-search is the primary, honest path.)
  2. render the model at that pose.
  3. domain_match — degrade the render to the crop's quality (downscale, blur,
     JPEG, coarse global luminance) so they are comparable. NOT per-channel
     colour transfer, which would inflate similarity for non-matches.
  4. robust_compare — silhouette IoU (colour-invariant) AND the cosine of ReID
     embeddings of the degraded render vs the crop. NOT raw pixel residual.

Two MANDATORY abstain gates (emit no score rather than guess):
  - maturity gate: the model must have enough confirmed real structure;
  - pose gate: the best pose must be confident enough.

The raw score is calibrated SEPARATELY into P(same) (car3d/calibration.py,
`rendercmp-<version>`), never merged with the ReID isotonic model, and it
enters the cascade only as a capped tiebreaker on an already-narrowed
shortlist (contributes 0 to the score — see reasoning/cascade.py W_RENDER_MAX).

Honesty: the CPU point renderer is a disc-splatting approximation, and on
cargen's stub prior the model is a procedural sedan whose geometry is
meaningless — so verify_match abstains below the maturity floor, and the
RESULTS row is PENDING without a real SF3D/TRELLIS backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cargen.core.camera import Intrinsics
from cargen.core.splat import GaussianCloud
from cargen.fusion_engine.point_renderer import PointRenderer
from car3d.render import _orbit_pose


@dataclass(frozen=True)
class VerifyConfig:
    min_observed_fraction: float = 0.30   # stricter than geometry's 0.15 floor
    min_observations: int = 3
    azimuths: int = 12
    elevations: tuple[float, ...] = (0.15, 0.35, 0.55)
    pose_min_confidence: float = 0.0      # normalized best-vs-runnerup separation
    jpeg_quality: int = 40
    blur_ksize: int = 3
    iou_weight: float = 0.5               # silhouette vs embedding blend


@dataclass(frozen=True)
class PoseEstimate:
    azimuth_rad: float
    elevation_rad: float
    confidence: float
    method: str            # "search" | "pnp"
    pose: Any = None


@dataclass(frozen=True)
class VerifyResult:
    score: float | None            # calibrated P(same); None => abstained
    abstained: bool
    reason: str
    pose: PoseEstimate | None = None
    silhouette_iou: float | None = None
    embed_similarity: float | None = None
    raw: float | None = None
    calibration_version: str = ""
    degraded_render: np.ndarray | None = field(default=None, repr=False)


def _render(cloud: GaussianCloud, pose, size: tuple[int, int]):
    """(BGR uint8, foreground-alpha bool) at the given pose and pixel size."""
    intr = Intrinsics.simple(size[0], size[1])
    result = PointRenderer().render(cloud, pose, intr)
    bgr = (np.clip(result.color, 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
    return bgr, (result.alpha > 0.5)


def _foreground_mask(bgr: np.ndarray, thresh: int = 22) -> np.ndarray:
    """Approximate vehicle silhouette: pixels differing from the border median."""
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    bg = float(np.median(border))
    return np.abs(gray - bg) > thresh


def domain_match(render_bgr: np.ndarray, crop_bgr: np.ndarray,
                 config: VerifyConfig | None = None) -> np.ndarray:
    """Degrade a clean render down to the crop's imaging quality."""
    import cv2

    cfg = config or VerifyConfig()
    h, w = crop_bgr.shape[:2]
    out = cv2.resize(render_bgr, (w, h), interpolation=cv2.INTER_AREA)
    k = cfg.blur_ksize | 1  # odd
    out = cv2.GaussianBlur(out, (k, k), 0)
    # JPEG round-trip to imprint block/ringing artifacts like the crop has.
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
    if ok:
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    # Coarse GLOBAL luminance match (a single scalar) — never per-channel color
    # transfer (that would inflate similarity for a non-matching vehicle).
    r_lum, c_lum = out.mean(), crop_bgr.mean()
    if r_lum > 1e-3:
        out = np.clip(out.astype(np.float32) * (c_lum / r_lum), 0, 255).astype(np.uint8)
    return out


def _silhouette_iou(alpha_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(alpha_a, mask_b).sum()
    union = np.logical_or(alpha_a, mask_b).sum()
    return float(inter / union) if union else 0.0


def robust_compare(degraded_render: np.ndarray, crop_bgr: np.ndarray,
                   render_alpha: np.ndarray, embed_fn) -> tuple[float, float]:
    """(silhouette IoU, ReID embedding cosine) — no raw photometric residual."""
    import cv2

    crop_mask = _foreground_mask(crop_bgr)
    if degraded_render.shape[:2] != crop_bgr.shape[:2]:
        render_alpha = cv2.resize(render_alpha.astype(np.uint8),
                                  (crop_bgr.shape[1], crop_bgr.shape[0])) > 0
    iou = _silhouette_iou(render_alpha, crop_mask)
    ea, eb = embed_fn(degraded_render), embed_fn(crop_bgr)
    cos = float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9))
    return iou, cos


def pose_search(cloud: GaussianCloud, crop_bgr: np.ndarray, embed_fn,
                config: VerifyConfig | None = None) -> PoseEstimate:
    """Search azimuth x elevation for the render best matching the crop."""
    cfg = config or VerifyConfig()
    h, w = crop_bgr.shape[:2]
    crop_emb = embed_fn(crop_bgr)
    scored: list[tuple[float, float, float, Any]] = []
    for az in np.linspace(0, 2 * np.pi, cfg.azimuths, endpoint=False):
        for el in cfg.elevations:
            pose = _orbit_pose(cloud, float(az), float(el))
            render, _ = _render(cloud, pose, (w, h))
            degraded = domain_match(render, crop_bgr, cfg)
            emb = embed_fn(degraded)
            cos = float(np.dot(emb, crop_emb)
                        / (np.linalg.norm(emb) * np.linalg.norm(crop_emb) + 1e-9))
            scored.append((cos, float(az), float(el), pose))
    scored.sort(key=lambda s: -s[0])
    best = scored[0]
    runner = scored[1][0] if len(scored) > 1 else 0.0
    confidence = float(max(0.0, (best[0] - runner)) / (abs(best[0]) + 1e-6))
    return PoseEstimate(azimuth_rad=best[1], elevation_rad=best[2],
                        confidence=min(1.0, confidence), method="search", pose=best[3])


def verify_match(target_model, crop_bgr: np.ndarray, embed_fn,
                 calibrator=None, config: VerifyConfig | None = None,
                 landmarks=None) -> VerifyResult:
    """Verify a crop against a target's 3D model, with mandatory abstain gates."""
    cfg = config or VerifyConfig()

    # --- maturity gate: rough single-photo models must not drive matches ---
    if not target_model.exists():
        return VerifyResult(None, True, "no 3D model for this target yet")
    asset = target_model.load()
    sig = target_model.geometry()
    if sig is None or sig.observed_fraction < cfg.min_observed_fraction \
            or len(asset.observations) < cfg.min_observations:
        frac = sig.observed_fraction if sig else 0.0
        return VerifyResult(
            None, True,
            f"model too immature to verify (observed {frac:.0%} < "
            f"{cfg.min_observed_fraction:.0%}, {len(asset.observations)} obs < "
            f"{cfg.min_observations}); abstaining rather than guessing")

    # --- pose (search primary; PnP only if a live LandmarkStore is supplied) ---
    pose = pose_search(asset.cloud, crop_bgr, embed_fn, cfg)
    if pose.confidence < cfg.pose_min_confidence:
        return VerifyResult(None, True,
                            f"pose ambiguous (confidence {pose.confidence:.2f} < "
                            f"{cfg.pose_min_confidence:.2f}); abstaining", pose=pose)

    # --- render at pose, degrade, compare on robust channels ---
    h, w = crop_bgr.shape[:2]
    render, alpha = _render(asset.cloud, pose.pose, (w, h))
    degraded = domain_match(render, crop_bgr, cfg)
    iou, cos = robust_compare(degraded, crop_bgr, alpha, embed_fn)
    raw = cfg.iou_weight * iou + (1 - cfg.iou_weight) * cos

    # --- calibrate SEPARATELY (or return raw, explicitly labeled) ---
    if calibrator is not None:
        score = float(calibrator.predict(raw))
        version = calibrator.version
    else:
        score, version = float(raw), "rendercmp-uncalibrated"
    return VerifyResult(
        score=score, abstained=False, reason="verified",
        pose=pose, silhouette_iou=iou, embed_similarity=cos, raw=raw,
        calibration_version=version, degraded_render=degraded)
