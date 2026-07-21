"""Perceptor: SightingEvent -> Observation.

This is the seam between the simulator and everything downstream. Two modes:

- fast mode (default for the live demo): render the vehicle crop directly
  and skip YOLO. detection_source="sim-fallback". Keeps the demo real-time
  on CPU.
- full mode (use_yolo=True): render a frame burst, run the real
  YOLO+ByteTrack detector, best-crop select, and only fall back to the
  simulator box if the detector finds nothing (which, on cartoon sprites,
  is the common case — recorded honestly per observation).

Either way the ReID embedding is computed by the real OSNet model on the
actual crop pixels, and plate/class attrs go through their documented
noise channels.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from perception.attributes import (
    AttributeNoiseConfig,
    perceive_class_attrs,
    perceive_instance_attrs,
)
from perception.detector import VehicleDetector
from perception.embedder import ReidEmbedder
from perception.plates import PlateNoiseConfig, SimulatedPlateReader
from perception.types import CLIP_FRAMES, SOURCE_HEURISTIC, Observation
from sim.model import SightingEvent
from sim.render import render_passage, render_vehicle_crop
from sim.road_graph import RoadGraph


@dataclass(frozen=True)
class PerceptionConfig:
    use_yolo: bool = False
    frames_per_passage: int = 4
    miss_prob: float = 0.05  # camera misses the vehicle entirely
    plate_noise: PlateNoiseConfig = field(default_factory=PlateNoiseConfig)
    attr_noise: AttributeNoiseConfig = field(default_factory=AttributeNoiseConfig)
    keep_crops: bool = True
    clip_gap_s: float = 0.2   # spacing between the sighting-clip frames
    seed: int = 47


class Perceptor:
    """Stateful pipeline wrapper (models load lazily on first use)."""

    def __init__(self, graph: RoadGraph, config: PerceptionConfig | None = None):
        self._graph = graph
        self._cfg = config or PerceptionConfig()
        self._embedder = ReidEmbedder()
        self._detector = VehicleDetector() if self._cfg.use_yolo else None
        self._plates = SimulatedPlateReader(self._cfg.plate_noise, seed=self._cfg.seed)

    def process(self, event: SightingEvent) -> Observation | None:
        """Run one sighting through the pipeline. None = missed detection."""
        cfg = self._cfg
        rng = random.Random(f"{cfg.seed}|miss|{event.event_id}")
        if rng.random() < cfg.miss_prob:
            return None

        camera = self._graph.camera(event.camera_id)
        detection_source = "sim-fallback"
        if cfg.use_yolo and self._detector is not None:
            burst = render_passage(
                event.truth, camera, event.timestamp_s, cfg.frames_per_passage
            )
            det = self._detector.best_detection(
                [f for f, _ in burst], [b for _, b in burst]
            )
            if det is None:
                return None
            crop, detection_source = det.crop, det.source
        else:
            crop = render_vehicle_crop(event.truth, event.camera_id, event.timestamp_s)

        embedding = self._embedder.embed(crop)
        plate = self._plates.read(event.truth.plate, event.event_id)
        class_attrs = perceive_class_attrs(event.truth, crop, event.event_id, cfg.attr_noise)
        instance_attrs = perceive_instance_attrs(event.truth, event.event_id, cfg.attr_noise)

        # Short looping sighting clip: the same vehicle sprite re-rendered at
        # successive moments of this passage. Frame 0 matches `crop` in the
        # fast path (same call, same timestamp). Illustrative synthetic motion,
        # honestly labeled as such in the UI.
        clip_frames: tuple = ()
        if cfg.keep_crops:
            clip_frames = tuple(
                render_vehicle_crop(
                    event.truth, event.camera_id,
                    event.timestamp_s + i * cfg.clip_gap_s)
                for i in range(CLIP_FRAMES))

        return Observation(
            event_id=event.event_id,
            camera_id=event.camera_id,
            timestamp_s=event.timestamp_s,
            lat=event.lat,
            lon=event.lon,
            embedding=embedding,
            plate=plate,
            class_attrs=class_attrs,
            class_attrs_source=SOURCE_HEURISTIC,
            instance_attrs=instance_attrs,
            detection_source=detection_source,
            crop=crop if cfg.keep_crops else None,
            clip_frames=clip_frames,
            eval_truth_id=event.truth.vehicle_id,
        )
