"""Rendering a target's 3D model: turntables, provenance overlay, and the
optional render-and-compare corroboration signal.

Uses cargen's CPU point renderer — no CUDA needed, honest about being a
disc-splatting approximation rather than a full Gaussian rasterizer.
"""
from __future__ import annotations

import numpy as np

from cargen.core.camera import CameraPose, Intrinsics
from cargen.core.splat import GaussianCloud, Provenance
from cargen.export.exporter import OBSERVED_TINT, PRIOR_TINT
from cargen.fusion_engine.point_renderer import PointRenderer


def _orbit_pose(cloud: GaussianCloud, azimuth_rad: float,
                elevation_rad: float = 0.35) -> CameraPose:
    center = cloud.positions.mean(axis=0)
    radius = float(np.linalg.norm(cloud.positions - center, axis=1).max())
    dist = radius * 2.6
    eye = center + dist * np.array([
        np.cos(azimuth_rad) * np.cos(elevation_rad),
        np.sin(azimuth_rad) * np.cos(elevation_rad),
        np.sin(elevation_rad)])
    return CameraPose.look_at(eye=eye, target=center)


def _provenance_colored(cloud: GaussianCloud) -> GaussianCloud:
    """Green = confirmed by real sightings, red = generative-prior guess."""
    colors = np.where(
        (cloud.provenance == Provenance.OBSERVED)[:, None],
        OBSERVED_TINT, PRIOR_TINT).astype(np.float32)
    from dataclasses import replace

    return replace(cloud, colors=colors)


def render_view(
    cloud: GaussianCloud, pose: CameraPose,
    size: tuple[int, int] = (320, 240),
) -> np.ndarray:
    """One BGR uint8 render from an arbitrary pose."""
    intr = Intrinsics.simple(size[0], size[1])
    result = PointRenderer().render(cloud, pose, intr)
    rgb = (np.clip(result.color, 0, 1) * 255).astype(np.uint8)
    return rgb[:, :, ::-1].copy()


def turntable_strip(
    cloud: GaussianCloud,
    n_views: int = 6,
    size: tuple[int, int] = (240, 180),
    provenance_overlay: bool = False,
) -> np.ndarray:
    """Horizontal strip of orbit views — the dossier's rotatable-model
    stand-in for environments without the WebGL viewer."""
    shown = _provenance_colored(cloud) if provenance_overlay else cloud
    views = [
        render_view(shown, _orbit_pose(cloud, az), size)
        for az in np.linspace(0, 2 * np.pi, n_views, endpoint=False)
    ]
    return np.hstack(views)


def render_and_compare(
    cloud: GaussianCloud,
    crop_bgr: np.ndarray,
    embed_fn,
    n_azimuths: int = 12,
) -> float:
    """OPTIONAL corroboration signal: best same-angle similarity between the
    target's renders and a query crop.

    Sweeps azimuth, embeds each render, returns max cosine vs the crop's
    embedding — comparing same-angle against same-angle instead of asking a
    2D embedding to be view-invariant. Clearly labeled experimental: the
    CPU renders are not photoreal, so this signal is advisory and is NOT
    wired into the cascade score by default (see reasoning/cascade.py —
    nothing there consumes it silently).
    """
    crop_emb = embed_fn(crop_bgr)
    best = 0.0
    for az in np.linspace(0, 2 * np.pi, n_azimuths, endpoint=False):
        view = render_view(cloud, _orbit_pose(cloud, float(az)))
        sim = float(np.dot(embed_fn(view), crop_emb))
        best = max(best, sim)
    return best
