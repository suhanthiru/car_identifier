"""Per-target persistent 3D model, gated exactly like profile updates.

THE PARALLEL, made explicit: Eyes Everywhere gates profile updates behind
plate- or operator-confirmation with reversible snapshots; cargen gates
model merges behind pending-approval (auto_merge off) with per-splat
provenance. This module unifies them — a sighting's crop fuses into the
target's 3D asset ONLY on the same events that open the profile-update
gate, and every fusion snapshots the previous cloud for rollback. The two
anti-poisoning mechanisms are one mechanism.

Backends follow cargen's own CARGEN_* selection, degrading to its CPU stubs
when a real one cannot be constructed (see _build_pipeline). On the stub the
prior is a PROCEDURAL SEDAN — a placeholder shape whose observed-vs-prior
provenance is still real, but whose geometry means nothing for identity.
Either way the geometry attributes refuse to fire until enough of the cloud
is OBSERVED (see car3d/geometry.py).

Build the Pipeline ONCE and inject it. A real prior backend loads gigabytes
of weights and holds most of an 8 GB GPU; constructing a Target3DModel per
event without a shared pipeline reloads all of it every single fusion.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from car3d.compat import require_subject_detail
from car3d.geometry import GeometrySignature, signature_from_cloud
from car3d.render import turntable_strip
from cargen.core.asset import VehicleAsset
from cargen.export.exporter import export_all

DEFAULT_ROOT = Path("data/targets3d")


# CPU stub path: 20k splats keeps a fuse under a few seconds. A real backend
# wants cargen's own 120k default — below that cargen's docstring calls the
# result "gravel rather than bodywork". Which one applies is decided from the
# backend actually constructed (see _build_pipeline), never from a config guess.
STUB_PRIOR_POINTS = 20_000
REAL_PRIOR_POINTS = 120_000

# Backend fallbacks are reported once per stage, not once per fusion: the
# reason matters, a thousand copies of it in the console do not.
_FALLBACK_REPORTED: set[str] = set()


def _report_fallback_once(stage: str, exc: Exception, consequence: str) -> None:
    if stage in _FALLBACK_REPORTED:
        return
    _FALLBACK_REPORTED.add(stage)
    print(f"car3d: {stage} backend unavailable ({type(exc).__name__}: {exc}); "
          f"falling back to cargen's stub. {consequence}")


def build_pipeline(prior_points: int | None = None):
    """cargen Pipeline honoring CARGEN_* env, falling back to CPU stubs.

    cargen's server defaults assume its ML installs (rembg segmenter, SF3D
    prior). This venv may not have them, so each heavy backend is probed and
    replaced by its stub on failure — the capability check the addendum
    requires, with the fallback stated on stdout rather than hidden.

    The probe catches Exception, not ImportError. A real backend fails in
    ways that are not import failures: gated Hugging Face weights raise
    OSError, an exhausted GPU raises torch.cuda.OutOfMemoryError, a broken
    driver/toolkit pairing raises RuntimeError. Catching only ImportError
    left an *installed but unusable* SF3D raising on every single fusion
    forever, with the real reason buried in a per-event traceback.
    """
    from cargen import backends
    from cargen.pipeline import Pipeline
    from cargen.prior_generation.stub import StubPriorGenerator

    try:
        segmenter = backends.build_segmenter()
    except Exception as exc:  # noqa: BLE001 — any failure must degrade, not raise
        _report_fallback_once(
            "segmenter", exc,
            "A rectangle mask tints the prior's paint with background colour.")
        segmenter = backends.build_segmenter("stub")
    try:
        prior = backends.build_prior_generator()
    except Exception as exc:  # noqa: BLE001
        _report_fallback_once(
            "image-to-3D prior", exc,
            "Geometry is a procedural sedan and means nothing for identity; "
            "geometry attributes refuse to fire until real evidence dominates.")
        prior = backends.build_prior_generator("stub")

    if prior_points is None:
        prior_points = (STUB_PRIOR_POINTS if isinstance(prior, StubPriorGenerator)
                        else REAL_PRIOR_POINTS)

    matcher = _build_matcher()
    return Pipeline(segmenter=segmenter, prior_generator=prior,
                    matcher=matcher,
                    registrar=_build_registrar(matcher),
                    fusion_config=_fusion_config(),
                    prior_points=prior_points)


def _build_matcher():
    """LightGlue when it is installed, ORB otherwise.

    Not a preference — a measured difference. Matching two masked crops of the
    same car, ORB against LightGlue:

        same camera, consecutive frames    8  vs  688
        across two cameras (wide baseline) 1  vs  208

    cargen's own ORB docstring says as much: wide-baseline matching "is where
    ORB gets brittle on cars — glossy, textureless panels and bilateral
    symmetry — so that path should prefer LightGlue". The cross-camera figure
    is the one that matters: at 1 match a second viewpoint can never register,
    so every reconstruction was permanently stuck at whatever its first frame
    saw.

    ORB stays the fallback rather than a hard requirement; LightGlue pulls in
    torch weights that a minimal checkout may not want.
    """
    from cargen import backends

    try:
        import lightglue  # noqa: F401
    except ImportError:
        return backends.build_matcher("orb")
    try:
        return backends.build_matcher("lightglue")
    except Exception as exc:            # noqa: BLE001 — weights, GPU, anything
        _report_fallback_once(
            "feature matcher", exc,
            "ORB matches ~1 point across a camera pair, so extra views of a "
            "target will not register and its model stays at one viewpoint.")
        return backends.build_matcher("orb")


# How far a match may sit from a stored landmark and still be lifted to its 3D
# point. cargen defaults to 6px, which was tuned for frames far smaller than a
# 1835x1238 CityFlow crop; at that size it yielded ~100 correspondences where
# 12px yields ~224, with reprojection error UNCHANGED at ~2px. Past ~18px the
# lift starts picking the wrong landmark and PnP fails outright, so this is
# nearer the floor of the useful band than the ceiling.
LANDMARK_RADIUS_PX = 12.0

# Registration confidence a view must reach to be fused.
#
# cargen defaults to 0.35 and nothing on this data ever reaches it: measured
# confidences run 0.16-0.25 even for poses whose reprojection error is ~2px on
# a 4px threshold — i.e. poses that are demonstrably good. The formula is the
# reason. It scores inliers/total_matches, where total_matches counts every
# match the matcher produced, so LightGlue's 688 matches drive that term to
# 0.02 and a BETTER matcher scores WORSE.
#
# This was not lowered on the argument above alone. Both settings were built
# and rendered: at 0.35 exactly one view fuses (37.1% observed); lower gates
# fuse five or six, reaching ~47%, with the silhouette, wheels and glazing
# intact from all eight orbit angles and no ghosted geometry.
#
# The VALUE is chosen for stability, not for the best single run. Measured
# confidences on this data cluster in 0.16-0.25, and registration has positive
# feedback — each accepted view enlarges the landmark store and makes the next
# easier — so a gate placed INSIDE that cluster is bistable: 0.23 produced 5 of
# 6 views (46.5%) on one run and 2 of 6 (38.3%) on the next, from nothing but
# GPU nondeterminism tipping the first borderline frame. A reconstruction whose
# quality swings on a coin flip is worse than a slightly noisier one that does
# not.
#
# 0.18 is at the LOW END of that 0.16-0.25 cluster, not below it — an
# earlier version of this comment claimed 'just below the cluster', which
# its own numbers contradict. Repeated runs at this value fused 6/6 views
# (46.7%, 46.6%), so in practice it clears the borderline frames that made
# 0.23 bistable; the residual risk is real and accepted rather than denied.
MIN_REGISTRATION_CONFIDENCE = 0.18


def _build_registrar(matcher):
    from cargen.pose_estimation.registration import PnPRegistrar

    return PnPRegistrar(matcher, neighbor_radius_px=LANDMARK_RADIUS_PX)


def _fusion_config():
    import dataclasses

    from cargen.fusion_engine.engine import FusionConfig

    return dataclasses.replace(
        FusionConfig(),
        min_registration_confidence=MIN_REGISTRATION_CONFIDENCE,
        # Traffic cameras look at a car from below the roofline, so the upper
        # surfaces are the least observed and the noisiest — they rendered as
        # black speckle floating above the roof. Pruning faint splats harder
        # and shortening the reach a new splat may be spawned from existing
        # geometry (cargen calls that "the guard against inventing depth in
        # open space") clears most of it. Measured on the same six views:
        # 0.05/8 -> 46.7% observed and visible speckle; 0.12/5 -> 46.1% and
        # visibly cleaner; 0.20/4 over-prunes and costs two whole views,
        # dropping to 43.7%. Half a point of coverage for the noise is worth it
        # — the noise reads as reconstruction failure to anyone looking.
        prune_opacity=0.12,
        densify_reach=5)


# Historical private name; the server builds one shared pipeline via the
# public name and injects it into every Target3DModel.
_build_pipeline = build_pipeline


@dataclass(frozen=True)
class FusionOutcome:
    target_id: str
    observations: int
    n_splats: int
    observed_fraction: float
    snapshot: str            # path of the pre-fusion cloud snapshot
    geometry: GeometrySignature | None
    accepted: bool


class Target3DModel:
    """Owns data/targets3d/<target_id>/: cargen asset + exports + snapshots."""

    def __init__(
        self,
        target_id: str,
        storage_root: Path | str = DEFAULT_ROOT,
        pipeline=None,
        prior_points: int | None = None,
    ):
        self.target_id = target_id
        self.dir = Path(storage_root) / target_id
        self._pipeline = pipeline
        self._prior_points = prior_points

    def _get_pipeline(self):
        if self._pipeline is None:
            self._pipeline = _build_pipeline(self._prior_points)
        return self._pipeline

    def exists(self) -> bool:
        return VehicleAsset.is_asset_dir(self.dir)

    def load(self) -> VehicleAsset:
        return VehicleAsset.load(self.dir)

    def fuse_confirmed_crop(
        self,
        crop_bgr: np.ndarray,
        event_id: str,
        reason: str,
        timestamp: float | None = None,
        device: str = "cctv",
        export: bool = True,
    ) -> FusionOutcome:
        """Fuse one CONFIRMED sighting's crop into the target's asset.

        Callers must only invoke this from the gated paths (plate-confirmed
        association or operator-accepted review) — the same rule as
        TargetProfile updates. `reason` records which gate opened; it lands
        in the asset's observation log for audit.

        `device` selects cargen's evidence weight, which encodes how
        authoritative the *imagery* is, not how trusted the association is —
        so a reference photo supplied by an operator should not be pinned to
        the CCTV tier just because the gate that admitted it was the same.

        `export` writes the .ply/.splat set inline. The server turns it off
        and regenerates on demand instead: exports are ~20 MB per fusion and
        the dossier is opened far less often than sightings arrive.
        """
        # Before any GPU time is spent: refuse subjects too small to carry
        # detail. cargen's ee-adapter does this itself; on master it does not
        # exist, and car3d.compat enforces the same bar so the protection does
        # not depend on which branch happens to be checked out.
        require_subject_detail(crop_bgr)

        pipeline = self._get_pipeline()
        asset = self.load() if self.exists() else VehicleAsset(name=self.target_id)
        snapshot = self._snapshot_cloud(asset)

        rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
        result = pipeline.ingest_photo(
            asset, rgb, device=device, timestamp=timestamp or time.time())
        # Stamp the gate that authorized this fusion onto the observation log.
        if asset.observations:
            asset.observations[-1]["gate_reason"] = reason
            asset.observations[-1]["event_id"] = event_id
        asset.save(self.dir)
        if export:
            self._export(asset)

        accepted = result.frames_fused > 0 or result.created
        sig = signature_from_cloud(asset.cloud)
        return FusionOutcome(
            target_id=self.target_id,
            observations=len(asset.observations),
            n_splats=asset.cloud.n,
            observed_fraction=sig.observed_fraction if sig else 0.0,
            snapshot=snapshot,
            geometry=sig,
            accepted=accepted,
        )

    def rollback_to(self, snapshot_path: str | Path | None) -> None:
        """Restore a pre-fusion cloud snapshot (operator rejected the chain).

        An empty snapshot path means the fusion being undone was the first
        one — there was no model before it, so the correct rollback is to
        remove the asset entirely rather than restore a placeholder that
        cannot be loaded.
        """
        if not snapshot_path:
            if self.dir.exists():
                shutil.rmtree(self.dir)
            return
        snapshot_path = Path(snapshot_path)
        if not snapshot_path.exists():
            raise FileNotFoundError(f"no snapshot at {snapshot_path}")
        shutil.copy2(snapshot_path, self.dir / "cloud.npz")
        self._export(self.load())

    def geometry(self) -> GeometrySignature | None:
        return signature_from_cloud(self.load().cloud) if self.exists() else None

    def turntable_png(self, provenance_overlay: bool = True) -> Path:
        """Write (and return) the dossier's turntable strip."""
        import cv2

        asset = self.load()
        strip = turntable_strip(asset.cloud, provenance_overlay=provenance_overlay)
        path = self.dir / "exports" / (
            "turntable_provenance.png" if provenance_overlay else "turntable.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), strip)
        return path

    # -------------------------------------------------- lazy artefacts
    #
    # Both are derived views of cloud.npz, so "up to date" is simply "newer
    # than the cloud". Generating them on request instead of on every fusion
    # takes ~20 MB of writes and six CPU renders off the ingest path; an
    # operator opening a dossier pays for what they actually look at.

    def _is_stale(self, path: Path) -> bool:
        cloud = self.dir / "cloud.npz"
        if not path.is_file():
            return True
        return not cloud.is_file() or path.stat().st_mtime < cloud.stat().st_mtime

    def ensure_exports(self) -> None:
        """Regenerate the .ply/.splat set if the cloud has moved on."""
        if self.exists() and self._is_stale(self.dir / "exports" / "model.splat"):
            self._export(self.load())

    def turntable_is_stale(self, provenance_overlay: bool = True) -> bool:
        """Would ensure_turntable() actually do work? A file stat, no render.

        Lets a request decide whether to render inline or hand the job to a
        worker, without paying six CPU renders to find out. The status
        endpoint asks this on every dossier open.
        """
        if not self.exists():
            return False
        name = ("turntable_provenance.png" if provenance_overlay else "turntable.png")
        return self._is_stale(self.dir / "exports" / name)

    def ensure_turntable(self, provenance_overlay: bool = True) -> Path | None:
        """Regenerate the dossier turntable if the cloud has moved on."""
        if not self.exists():
            return None
        name = ("turntable_provenance.png" if provenance_overlay else "turntable.png")
        path = self.dir / "exports" / name
        if self._is_stale(path):
            return self.turntable_png(provenance_overlay=provenance_overlay)
        return path

    # ------------------------------------------------------------ internals

    def _snapshot_cloud(self, asset: VehicleAsset) -> str:
        """Copy the current cloud aside so a rejected chain can be undone.

        Returns "" when there is no cloud yet. The first fusion has nothing
        to restore, and the placeholder .npz written here previously made
        rollback *corrupt* the asset rather than undo it: restoring it left a
        cloud.npz with no `positions`, and every subsequent load raised
        KeyError. "" is the honest record of "there was no model before".
        """
        existing = self.dir / "cloud.npz"
        if not existing.exists():
            return ""
        snap_dir = self.dir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        target = snap_dir / f"v{len(asset.observations):03d}.npz"
        shutil.copy2(existing, target)
        return str(target)

    def _export(self, asset: VehicleAsset) -> None:
        export_all(asset.cloud, self.dir / "exports")
