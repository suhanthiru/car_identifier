"""RealPerceptor: SightingEvent-shaped inputs -> Observation, from real
CityFlow video instead of the synthetic renderer.

Parallel to perception/observe.py's Perceptor, but every field is honestly
scoped to what CityFlow actually gives us: real ground-truth boxes, real
pixels, a real color heuristic, and (optionally) a real plate OCR attempt.
There is no real make/model/body_type annotation in this dataset, so
class_attrs only ever carries color, and instance_attrs is always empty --
inventing either would misrepresent what the data supports. See K's plan
entry and DATASETS.md for the honesty constraints this follows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from datasets.cityflow_video import VideoFrameSource, bbox_for
from perception.attributes import estimate_color
from perception.embedder import ReidEmbedder
from perception.plates import FastPlateOcrReader
from perception.types import SOURCE_HEURISTIC, Observation


class RealPerceptor:
    """Stateful pipeline wrapper for one CityFlow scenario's cameras.

    `camera_dirs`: camera_id -> the per-camera directory containing
    `gt/gt.txt` and `vdo.avi` (e.g. `.../S01/c001/`).
    `camera_positions`: camera_id -> (lat, lon), from
    `CityFlowScenario.camera_gps()` -- the honest, real-center-anchored
    approximate position (see datasets/cityflow.py). Missing entries
    default to (0.0, 0.0) rather than guessing.
    `pipeline_state`: any object exposing `.enable_plate_ocr` (typically
    the server's `app.state`), checked on every call so a runtime toggle
    takes effect immediately, not just at startup.
    """

    def __init__(
        self,
        camera_dirs: dict[str, Path],
        camera_positions: dict[str, tuple[float, float]] | None = None,
        pipeline_state: object | None = None,
        embedder: ReidEmbedder | None = None,
        plate_reader: FastPlateOcrReader | None = None,
    ):
        self._camera_dirs = camera_dirs
        self._camera_positions = camera_positions or {}
        self._pipeline_state = pipeline_state
        self._embedder = embedder or ReidEmbedder()
        self._plate_reader = plate_reader or FastPlateOcrReader()
        self._video_sources: dict[str, VideoFrameSource] = {}

    def _video_source(self, camera_id: str) -> VideoFrameSource:
        if camera_id not in self._video_sources:
            self._video_sources[camera_id] = VideoFrameSource(
                self._camera_dirs[camera_id] / "vdo.avi")
        return self._video_sources[camera_id]

    def _plate_ocr_enabled(self) -> bool:
        return bool(getattr(self._pipeline_state, "enable_plate_ocr", True))

    def process(
        self, camera_id: str, vehicle_id: int, frame: int, timestamp_s: float,
    ) -> Observation | None:
        cam_dir = self._camera_dirs.get(camera_id)
        if cam_dir is None:
            return None
        bbox = bbox_for(cam_dir / "gt" / "gt.txt", frame, vehicle_id)
        if bbox is None:
            return None
        crop = self._video_source(camera_id).crop(frame, bbox)
        if crop is None or crop.size == 0:
            return None

        embedding = self._embedder.embed(crop)
        color = estimate_color(crop)
        plate = None
        if self._plate_ocr_enabled():
            try:
                plate = self._plate_reader.read(crop)
            except RuntimeError:
                # fast-plate-ocr not installed: degrade to "no read", same
                # as any other miss -- plate OCR is best-effort, never
                # load-bearing for the rest of the pipeline.
                plate = None

        lat, lon = self._camera_positions.get(camera_id, (0.0, 0.0))
        return Observation(
            event_id=f"cf-{camera_id}-{frame}-{vehicle_id}",
            camera_id=camera_id, timestamp_s=timestamp_s, lat=lat, lon=lon,
            embedding=embedding, plate=plate,
            class_attrs={"color": color}, class_attrs_source=SOURCE_HEURISTIC,
            instance_attrs={}, detection_source="cityflow-gt",
            crop=crop, eval_truth_id=str(vehicle_id))

    def close(self) -> None:
        for src in self._video_sources.values():
            src.close()
        self._video_sources.clear()
