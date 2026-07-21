"""Observation types: what perception hands to the reasoning layer.

An Observation is everything downstream code is allowed to know about a
sighting. Provenance is explicit on every noisy field: `source` values tell
you whether a real model produced the value or the simulator did. Nothing
in reasoning/ may touch `eval_truth_id`; it exists only so the evaluation
and calibration harnesses can score decisions after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

# Provenance labels.
SOURCE_MODEL = "model"          # a real off-the-shelf model produced this
SOURCE_HEURISTIC = "heuristic"  # simple real image processing (e.g. color)
SOURCE_SIM = "sim"              # simulator-labeled ground truth w/ injected noise

# How many frames make up the short looping sighting clip the console plays in
# place of a single still. Shared so the synthetic and real perceptors agree on
# length; each picks its own frame spacing (seconds vs frame-step).
CLIP_FRAMES = 6


@dataclass(frozen=True)
class PlateRead:
    """One plate-read attempt. May be wrong; confidence is the reader's own."""

    text: str
    confidence: float
    source: str  # SOURCE_MODEL or SOURCE_SIM


@dataclass(frozen=True)
class Detection:
    """Best-crop pick for one camera passage."""

    crop: np.ndarray = field(repr=False, compare=False)
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    source: str  # "yolo" when the real detector found it, "sim-fallback" otherwise


@dataclass(frozen=True)
class Observation:
    """A fully-processed sighting, ready for identity reasoning."""

    event_id: str
    camera_id: str
    timestamp_s: float
    lat: float
    lon: float
    # L2-normalized appearance embedding (ReID model output).
    embedding: np.ndarray = field(repr=False, compare=False)
    plate: PlateRead | None
    # make/model/body_type/color as perceived (may be wrong).
    class_attrs: Mapping[str, str]
    class_attrs_source: str
    # Distinguishing marks. Simulator-labeled; no real detector backs these.
    instance_attrs: Mapping[str, str]
    detection_source: str
    crop: np.ndarray | None = field(default=None, repr=False, compare=False)
    # A short burst of BGR frames (car crops) for this same passage, oldest
    # first — the console loops them as the sighting clip. The ACTUAL sighting
    # (rendered passage in the synthetic world, real vdo.avi footage in
    # CityFlow), never a prediction. Empty when clips are disabled.
    clip_frames: tuple[np.ndarray, ...] = field(
        default=(), repr=False, compare=False)
    # EVALUATION ONLY. The reasoning layer must never read this field —
    # it is the simulator's answer key, used to score decisions afterwards.
    eval_truth_id: str = ""
