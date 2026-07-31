"""Request/response schemas for the API.

Observations arrive from (simulated) edge nodes already perceived: the
edge tier runs detection/embedding/OCR locally and ships compact results,
which is also how the real architecture would partition work. Crops travel
as base64 PNG so the review UI can show them.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


# Attribute maps carry short keys like "color" and short values like "silver".
# Bounding the dict length alone left the door open: a single 100,000-character
# KEY was accepted, which bloats every response that echoes the profile just as
# effectively as a long label did.
AttrKey = Annotated[str, StringConstraints(strip_whitespace=True, max_length=40)]
AttrValue = Annotated[str, StringConstraints(max_length=120)]
AttrMap = Annotated[dict[AttrKey, AttrValue], Field(max_length=32)]


class StrictModel(BaseModel):
    """Request bodies that reject fields they do not recognise.

    Pydantic ignores unknown keys by default, which made a mistyped or
    misremembered request look like a success: POSTing {"action": "pause"} to
    the feed control returned 200 and changed nothing, and flagging a target
    with {"vehicle_id": 999, "body": "motorcycle"} returned 201 having silently
    dropped both. A tester reported those as "pause is broken" and "invalid
    vehicles are accepted" — reasonable conclusions from the evidence, and both
    wrong. The API had accepted a request it did not understand and said
    nothing.

    Refusing the unknown field turns four silent misunderstandings into one
    immediate 422 that names the offending key.

    `allow_inf_nan=False` closes a denial of service. Python's json module
    accepts the non-standard literals `Infinity` and `NaN`, and Pydantic let
    them through a plain `float` field, so a single accepted sighting carrying
    `timestamp_s: Infinity` set the server's clock to infinity — after which
    /api/stats and /api/audit returned 500 forever, because neither value can
    be serialised back to JSON. One unauthenticated POST permanently broke the
    console for everyone until an operator reset it. NaN was the same story:
    it silently passes `ge`/`le` bounds checks (every comparison with NaN is
    False) and crashes downstream instead.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FlagTargetRequest(StrictModel):
    # Stripped before the length check, so "   " is rejected rather than
    # creating a target with no readable name. min_length=1 alone accepted
    # whitespace, and the operator then had a row in the target list they
    # could not identify or tell apart from another blank one.
    label: Annotated[str, StringConstraints(strip_whitespace=True,
                                            min_length=1, max_length=120)]
    # Bounded like the label. Unbounded it accepted a 5 MB plate string, which
    # is both nonsense as a plate and a free amplifier for anything trying to
    # exhaust memory.
    plate: str = Field(default="", max_length=32)
    class_attrs: AttrMap = {}
    instance_attrs: AttrMap = {}
    # Optional reference photo of the vehicle being flagged (base64 PNG) —
    # e.g. the CityFlow browse thumbnail the operator clicked. The server
    # seeds honestly-derived evidence from it: the pixel-color heuristic and
    # one appearance-gallery embedding. Without it a label-only flag has no
    # evidence for the cascade to ever match against.
    reference_crop_b64: str = ""
    # Further crops of the same reference passage (base64 PNGs) — the pose
    # changes across a passage, so seeding first/mid/last frames lets the
    # capped ReID tiebreaker actually recognize the flagged car later.
    reference_gallery_b64: list[str] = []


class PlateReadIn(StrictModel):
    text: str = Field(max_length=32)
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "sim"


class SightingReport(StrictModel):
    """What an edge node reports for one vehicle passage."""

    # The event id becomes a FILENAME: the crop is stored as `{event_id}.png`
    # and the clip frames as `{event_id}.f{i}.png`. It was an unrestricted
    # 120-character string, so a single unauthenticated POST could write
    # attacker-controlled bytes anywhere the server process can write --
    # `../../x`, `C:\...\x`, `/x` were all confirmed to escape the crops
    # directory, including to a drive root. The read side already resolves and
    # checks containment; the write side had nothing.
    #
    # Restricted to what real ids actually use (`evt-00016`,
    # `cf-c001-1234-56`), which also removes the `<`, `>`, `|` and NUL
    # characters that made write_bytes() raise and 500 the request.
    event_id: Annotated[str, StringConstraints(
        min_length=1, max_length=120, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    camera_id: str = Field(min_length=1, max_length=40)
    # Bounded, not merely finite. allow_inf_nan=False rejects Infinity and NaN
    # but happily accepted 1e308, and `state.sim_now = max(sim_now, ...)` runs
    # BEFORE the work that then failed -- so one rejected request left the
    # replay clock at 1e308 permanently, for every client, exactly the outage
    # the Infinity guard was written to prevent. Range is generous (about 31
    # years of footage) and still keeps the clock representable.
    timestamp_s: float = Field(ge=0.0, le=1e9)
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    embedding: list[float] = Field(min_length=8)
    plate: PlateReadIn | None = None
    class_attrs: AttrMap = {}
    class_attrs_source: str = Field(default="heuristic", max_length=40)
    instance_attrs: AttrMap = {}
    detection_source: str = Field(default="sim-fallback", max_length=40)
    crop_png_b64: str = ""
    # Ordered base64 PNG frames of the short sighting clip (oldest first).
    # Empty when clips are disabled; the console loops them in the card.
    clip_frames_b64: list[str] = []
    # Simulator ground truth for the evaluation harness. A real edge node
    # would not send this; the serving path never reads it.
    eval_truth_id: str = Field(default="", max_length=64)


class ReviewResolution(StrictModel):
    accept: bool


class ProfileEditRequest(StrictModel):
    """Operator-initiated profile edit (label/plate/attrs). Gated: recorded
    in profile_updates with the operator as the authority.

    Bounded identically to FlagTargetRequest. It previously had no limits at
    all, so the same 5 MB label that POST /api/targets rejects with a 422 was
    accepted here with a 200 — and every target list, every dossier and every
    WebSocket snapshot then carried it. Two poisoned targets took the list
    response to 10 MB and 1.5s, and because the console's snapshot broadcast
    includes each target's full profile, a client with an ordinary 1 MB frame
    limit is disconnected with "1009 message too big" on the next broadcast.
    An edit path that can do what the create path forbids is a hole, not a
    convenience.
    """

    label: Annotated[str, StringConstraints(strip_whitespace=True,
                                            min_length=1,
                                            max_length=120)] | None = None
    plate: str | None = Field(default=None, max_length=32)
    class_attrs: AttrMap | None = None
    instance_attrs: AttrMap | None = None


class InspectTargetIn(StrictModel):
    """One hand-built target profile for the reasoning sandbox."""

    target_id: str = Field(default="sandbox-target", min_length=1, max_length=40)
    label: str = "Test target"
    plate: str = ""
    class_attrs: AttrMap = {}
    instance_attrs: AttrMap = {}
    last_seen_camera_id: str = ""
    last_seen_timestamp_s: float | None = None
    # Simulated ReID similarity to the sighting below, in [-1, 1]. None = the
    # target has no appearance gallery yet, so ReID is unavailable (matches
    # a freshly-flagged, never-confirmed target).
    reid_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)


class InspectSightingIn(StrictModel):
    """One hand-built sighting for the reasoning sandbox."""

    camera_id: str = Field(min_length=1)
    timestamp_s: float
    plate_text: str = ""
    plate_confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    class_attrs: AttrMap = {}
    instance_attrs: AttrMap = {}


class InspectRequest(StrictModel):
    """Reasoning-sandbox request: no DB, no tracker, no audit trail — just
    the cascade run on inputs a human constructed by hand. Up to 4 targets
    lets the ambiguity / candidate-set behavior be exercised directly."""

    sighting: InspectSightingIn
    targets: list[InspectTargetIn] = Field(min_length=1, max_length=4)
    distinctiveness_floor: float | None = Field(default=None, ge=0.0, le=1.0)


class PipelineConfigRequest(StrictModel):
    """Runtime toggle for real-clip mode's plate OCR (see K's live console).
    Only meaningful fields need be sent; omitted ones are left unchanged."""

    plate_ocr: bool | None = None


class FeedControlRequest(StrictModel):
    """Freeze/resume the replay clock (server.feed.FeedClock). Omit the
    field to read the current state without changing it."""

    paused: bool | None = None
