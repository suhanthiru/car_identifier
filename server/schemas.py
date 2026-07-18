"""Request/response schemas for the API.

Observations arrive from (simulated) edge nodes already perceived: the
edge tier runs detection/embedding/OCR locally and ships compact results,
which is also how the real architecture would partition work. Crops travel
as base64 PNG so the review UI can show them.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class FlagTargetRequest(BaseModel):
    label: str = Field(min_length=1, max_length=120)
    plate: str = ""
    class_attrs: dict[str, str] = {}
    instance_attrs: dict[str, str] = {}


class PlateReadIn(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "sim"


class SightingReport(BaseModel):
    """What an edge node reports for one vehicle passage."""

    event_id: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    timestamp_s: float
    lat: float
    lon: float
    embedding: list[float] = Field(min_length=8)
    plate: PlateReadIn | None = None
    class_attrs: dict[str, str] = {}
    class_attrs_source: str = "heuristic"
    instance_attrs: dict[str, str] = {}
    detection_source: str = "sim-fallback"
    crop_png_b64: str = ""
    # Simulator ground truth for the evaluation harness. A real edge node
    # would not send this; the serving path never reads it.
    eval_truth_id: str = ""


class ReviewResolution(BaseModel):
    accept: bool


class ProfileEditRequest(BaseModel):
    """Operator-initiated profile edit (label/plate/attrs). Gated: recorded
    in profile_updates with the operator as the authority."""

    label: str | None = None
    plate: str | None = None
    class_attrs: dict[str, str] | None = None
    instance_attrs: dict[str, str] | None = None


class InspectTargetIn(BaseModel):
    """One hand-built target profile for the reasoning sandbox."""

    target_id: str = Field(default="sandbox-target", min_length=1, max_length=40)
    label: str = "Test target"
    plate: str = ""
    class_attrs: dict[str, str] = {}
    instance_attrs: dict[str, str] = {}
    last_seen_camera_id: str = ""
    last_seen_timestamp_s: float | None = None
    # Simulated ReID similarity to the sighting below, in [-1, 1]. None = the
    # target has no appearance gallery yet, so ReID is unavailable (matches
    # a freshly-flagged, never-confirmed target).
    reid_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)


class InspectSightingIn(BaseModel):
    """One hand-built sighting for the reasoning sandbox."""

    camera_id: str = Field(min_length=1)
    timestamp_s: float
    plate_text: str = ""
    plate_confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    class_attrs: dict[str, str] = {}
    instance_attrs: dict[str, str] = {}


class InspectRequest(BaseModel):
    """Reasoning-sandbox request: no DB, no tracker, no audit trail — just
    the cascade run on inputs a human constructed by hand. Up to 4 targets
    lets the ambiguity / candidate-set behavior be exercised directly."""

    sighting: InspectSightingIn
    targets: list[InspectTargetIn] = Field(min_length=1, max_length=4)
    distinctiveness_floor: float | None = Field(default=None, ge=0.0, le=1.0)
