"""Per-target persistent 3D model, gated exactly like profile updates.

THE PARALLEL, made explicit: Eyes Everywhere gates profile updates behind
plate- or operator-confirmation with reversible snapshots; cargen gates
model merges behind pending-approval (auto_merge off) with per-splat
provenance. This module unifies them — a sighting's crop fuses into the
target's 3D asset ONLY on the same events that open the profile-update
gate, and every fusion snapshots the previous cloud for rollback. The two
anti-poisoning mechanisms are one mechanism.

Uses cargen's stub backends by default (CPU, no ML installs): the prior is
then a PROCEDURAL SEDAN, so the 3D model is a placeholder shape whose
observed-vs-prior provenance is still real. Real reconstruction quality
requires a real prior backend (SF3D/TRELLIS) — see cargen's README; the
geometry attributes refuse to fire until enough of the cloud is OBSERVED
either way (see car3d/geometry.py).
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from car3d.geometry import GeometrySignature, signature_from_cloud
from car3d.render import turntable_strip
from cargen.core.asset import VehicleAsset
from cargen.export.exporter import export_all

DEFAULT_ROOT = Path("data/targets3d")


def _build_pipeline(prior_points: int):
    """cargen Pipeline honoring CARGEN_* env, falling back to CPU stubs.

    cargen's server defaults assume its ML installs (rembg segmenter, SF3D
    prior). This venv may not have them, so each heavy backend is probed and
    replaced by its stub when the import fails — the capability check the
    addendum requires, with the fallback stated on stdout rather than hidden.
    """
    from cargen import backends
    from cargen.pipeline import Pipeline

    try:
        segmenter = backends.build_segmenter()
    except ImportError:
        print("car3d: rembg unavailable, using cargen's stub segmenter "
              "(rectangle mask; fine for the demo, not for real quality)")
        segmenter = backends.build_segmenter("stub")
    try:
        prior = backends.build_prior_generator()
    except ImportError:
        print("car3d: no real image-to-3D backend installed, using cargen's "
              "procedural-sedan stub prior (geometry attrs will refuse to "
              "fire until real evidence dominates the cloud)")
        prior = backends.build_prior_generator("stub")
    return Pipeline(segmenter=segmenter, prior_generator=prior,
                    prior_points=prior_points)
# CPU stub path: 20k splats keeps a fuse under a few seconds. Real backends
# want cargen's 120k default — override via prior_points.
STUB_PRIOR_POINTS = 20_000


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
        prior_points: int = STUB_PRIOR_POINTS,
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
    ) -> FusionOutcome:
        """Fuse one CONFIRMED sighting's crop into the target's asset.

        Callers must only invoke this from the gated paths (plate-confirmed
        association or operator-accepted review) — the same rule as
        TargetProfile updates. `reason` records which gate opened; it lands
        in the asset's observation log for audit.
        """
        pipeline = self._get_pipeline()
        asset = self.load() if self.exists() else VehicleAsset(name=self.target_id)
        snapshot = self._snapshot_cloud(asset)

        rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
        result = pipeline.ingest_photo(
            asset, rgb, device="cctv", timestamp=timestamp or time.time())
        # Stamp the gate that authorized this fusion onto the observation log.
        if asset.observations:
            asset.observations[-1]["gate_reason"] = reason
            asset.observations[-1]["event_id"] = event_id
        asset.save(self.dir)
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

    def rollback_to(self, snapshot_path: str | Path) -> None:
        """Restore a pre-fusion cloud snapshot (operator rejected the chain)."""
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

    # ------------------------------------------------------------ internals

    def _snapshot_cloud(self, asset: VehicleAsset) -> str:
        snap_dir = self.dir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        name = f"v{len(asset.observations):03d}.npz"
        target = snap_dir / name
        existing = self.dir / "cloud.npz"
        if existing.exists():
            shutil.copy2(existing, target)
        else:
            np.savez_compressed(target, empty=np.zeros(0))
        return str(target)

    def _export(self, asset: VehicleAsset) -> None:
        export_all(asset.cloud, self.dir / "exports")
