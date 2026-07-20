"""Structured decision signals: typed booleans/scalars behind the cascade.

The cascade used to infer plate_exact / attrs_support / mark counts by
substring-matching plain-English fact text. That was fragile, and the
counterfactual + distinctiveness features need reliable typed signals — in
particular the transit veto boundary (graph.min_transit_s), which no fact
carries. `compute_signals` derives the same facts the plausibility layer
emits, but as structured data, from the same inputs and the same rules.

A parity test (tests/test_signals.py) pins these against the fact-based
interpretation so the two cannot drift.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from perception.plates import CONFUSIONS
from perception.types import Observation
from reasoning.plausibility import (
    GEOM_PREFIX, PLATE_VETO_CONF, STALE_TRACK_S, _char_diffs, _partial_match,
)
from reasoning.profile import TargetProfile
from sim.road_graph import RoadGraph


@dataclass(frozen=True)
class MatchSignals:
    # plate tier
    plate_exact: bool = False
    plate_near: bool = False          # one OCR-confusable char off
    plate_contradiction: bool = False  # clean mismatch, high confidence -> veto
    plate_available: bool = False     # target has a plate AND sighting read one
    # A real ALPR partial read (some characters masked "_") whose KNOWN
    # positions all agree with the target plate. Weaker than plate_exact,
    # stronger than no read -- see reasoning/plausibility.py's _partial_match.
    plate_partial_match: bool = False
    plate_known_chars: int = 0
    plate_total_chars: int = 0
    # class/instance tier
    attrs_consistent: bool = False
    class_mismatches: tuple[str, ...] = ()
    body_veto: bool = False
    mark_match_count: int = 0
    mark_veto: bool = False
    # transit tier — the counterfactual boundary lives here
    transit_applicable: bool = False
    same_camera: bool = False
    transit_dt_s: float | None = None
    transit_fastest_s: float | None = None   # None for same-camera / no history
    transit_veto: bool = False
    no_path_veto: bool = False
    # geometry (coarse, caution/support only — never a veto)
    geometry_consistent: bool = False
    geometry_inconsistent: bool = False
    # corroboration
    corroboration_ok: bool = False
    corroboration_stale: bool = False
    # reid (filled by the cascade after gallery lookup; NEVER feeds distinctiveness)
    has_gallery: bool = False
    reid_similarity: float = 0.0
    reid_prob: float = 0.0
    # render-and-compare (car3d): separate calibrated tiebreaker; 0 to score
    render_match_p: float | None = None
    render_match_version: str = ""

    @property
    def any_veto(self) -> bool:
        return self.plate_contradiction or self.transit_veto or self.body_veto or self.mark_veto

    def with_reid(self, has_gallery: bool, similarity: float, prob: float) -> "MatchSignals":
        return replace(self, has_gallery=has_gallery, reid_similarity=similarity, reid_prob=prob)

    def with_render(self, prob: float | None, version: str) -> "MatchSignals":
        return replace(self, render_match_p=prob, render_match_version=version)


def _plate_signals(obs: Observation, profile: TargetProfile) -> dict:
    if not profile.plate or obs.plate is None:
        return {}
    seen, want = obs.plate.text.upper(), profile.plate.upper()
    out: dict = {"plate_available": True}
    if seen == want:
        out["plate_exact"] = True
        return out
    diffs = _char_diffs(seen, want)
    if diffs is not None and len(diffs) == 1 and CONFUSIONS.get(diffs[0][0]) == diffs[0][1]:
        out["plate_near"] = True
        return out
    partial = _partial_match(seen, want)
    if partial is not None:
        out["plate_partial_match"] = True
        out["plate_known_chars"], out["plate_total_chars"] = partial
        return out
    if obs.plate.confidence >= PLATE_VETO_CONF:
        out["plate_contradiction"] = True
    return out


def _transit_signals(obs: Observation, profile: TargetProfile, graph: RoadGraph) -> dict:
    last = profile.last_seen
    if last is None:
        return {}
    dt = obs.timestamp_s - last.timestamp_s
    out: dict = {"transit_applicable": True, "transit_dt_s": dt}
    if last.camera_id == obs.camera_id:
        out["same_camera"] = True
        if dt < 0:
            out["transit_veto"] = True
        return out
    fastest = graph.min_transit_s(last.camera_id, obs.camera_id)
    if fastest is None:
        out["no_path_veto"] = True
        out["transit_veto"] = True
        return out
    out["transit_fastest_s"] = fastest
    if dt < 0 or dt < fastest:
        out["transit_veto"] = True
    return out


def _attribute_signals(obs: Observation, profile: TargetProfile) -> dict:
    want, got = profile.class_attrs, obs.class_attrs
    out: dict = {}
    if want.get("body_type") and got.get("body_type") and want["body_type"] != got["body_type"]:
        out["body_veto"] = True
    matches = [k for k in ("make", "model", "color")
               if want.get(k) and got.get(k) and want[k] == got[k]]
    mismatches = tuple(k for k in ("make", "model", "color")
                       if want.get(k) and got.get(k) and want[k] != got[k])
    if want.get("body_type") and got.get("body_type") == want.get("body_type"):
        matches.append("body_type")
    if matches:
        out["attrs_consistent"] = True
    if mismatches:
        out["class_mismatches"] = mismatches

    mark_matches = 0
    for kind, val in profile.instance_attrs.items():
        if kind.startswith(GEOM_PREFIX):
            continue
        seen = obs.instance_attrs.get(kind)
        if seen is None:
            continue
        if seen == val:
            mark_matches += 1
        else:
            out["mark_veto"] = True
    if mark_matches:
        out["mark_match_count"] = mark_matches
    return out


def _geometry_signals(obs: Observation, profile: TargetProfile) -> dict:
    obs_geom = {k: v for k, v in obs.instance_attrs.items() if k.startswith(GEOM_PREFIX)}
    prof_geom = {k: v for k, v in profile.instance_attrs.items() if k.startswith(GEOM_PREFIX)}
    shared = set(obs_geom) & set(prof_geom)
    if not shared:
        return {}
    out: dict = {}
    if any(obs_geom[k] == prof_geom[k] for k in shared):
        out["geometry_consistent"] = True
    if any(obs_geom[k] != prof_geom[k] for k in shared):
        out["geometry_inconsistent"] = True
    return out


def _corroboration_signals(obs: Observation, profile: TargetProfile, graph: RoadGraph) -> dict:
    last = profile.last_seen
    if last is None:
        return {}
    if obs.timestamp_s - last.timestamp_s > STALE_TRACK_S:
        return {"corroboration_stale": True}
    expected = set(graph.neighbors(last.camera_id)) | {last.camera_id}
    return {"corroboration_ok": obs.camera_id in expected}


def compute_signals(
    obs: Observation, profile: TargetProfile, graph: RoadGraph
) -> MatchSignals:
    """Structured signals mirroring the plausibility layer's facts."""
    patch: dict = {}
    patch.update(_plate_signals(obs, profile))
    patch.update(_transit_signals(obs, profile, graph))
    patch.update(_attribute_signals(obs, profile))
    patch.update(_geometry_signals(obs, profile))
    patch.update(_corroboration_signals(obs, profile, graph))
    return MatchSignals(**patch)
