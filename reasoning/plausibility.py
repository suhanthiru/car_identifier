"""Symbolic plausibility layer: four checks, every one explainable.

These run before any similarity score is even consulted. A hard veto from
any check rejects the association no matter how good the appearance match
is — physics and logic outrank embeddings. Checks:

1. plate         — clean match / near-miss (OCR confusion) / contradiction
2. transit time  — is the hop physically possible given the road graph?
3. attributes    — hard contradictions in class/instance attributes
4. corroboration — is the sighting where the target was expected next?

Each check returns plain-English Facts; the cascade combines them.
"""
from __future__ import annotations

from perception.plates import CONFUSIONS
from perception.types import Observation
from reasoning.facts import Fact, caution, info, support, veto
from reasoning.profile import TargetProfile
from sim.road_graph import RoadGraph

# A clean plate read at or above this confidence that flatly mismatches the
# target's known plate is a hard veto.
PLATE_VETO_CONF = 0.75
# Sightings this long after the last one (seconds) carry no corroboration
# weight — the target could have gone anywhere in the meantime.
STALE_TRACK_S = 900.0


def _char_diffs(a: str, b: str) -> list[tuple[str, str]] | None:
    """Positionwise diffs, or None if formats are incomparable."""
    if len(a) != len(b):
        return None
    return [(x, y) for x, y in zip(a, b) if x != y]


def _partial_match(seen: str, want: str) -> tuple[int, int] | None:
    """(known_chars, total_chars) if `seen` has >=1 unread ('_') position,
    every OTHER position agrees with `want`, and at least one position is
    actually known -- i.e. a real ALPR partial read that is consistent with
    (but doesn't fully confirm) the target plate. None otherwise: full
    reads, incomparable lengths, no mask at all, or a known-position
    contradiction (which stays a contradiction/veto candidate, mask or not).
    """
    if len(seen) != len(want) or "_" not in seen:
        return None
    known = 0
    for s, w in zip(seen, want):
        if s == "_":
            continue
        if s != w:
            return None
        known += 1
    return (known, len(seen)) if known else None


def check_plate(obs: Observation, profile: TargetProfile) -> list[Fact]:
    if not profile.plate:
        return [info("Target plate is unknown; plate evidence unavailable.", "plate")]
    if obs.plate is None:
        return [info("No plate read on this sighting.", "plate")]
    seen, want = obs.plate.text.upper(), profile.plate.upper()
    if seen == want:
        return [support(
            f"Plate read {seen} exactly matches the target plate "
            f"(reader confidence {obs.plate.confidence:.2f}).", "plate")]
    diffs = _char_diffs(seen, want)
    if diffs is not None and len(diffs) == 1 and CONFUSIONS.get(diffs[0][0]) == diffs[0][1]:
        return [support(
            f"Plate read {seen} is one OCR-confusable character off the "
            f"target plate {want} ({diffs[0][0]}<->{diffs[0][1]}); treated as a weak match.",
            "plate")]
    partial = _partial_match(seen, want)
    if partial is not None:
        known, total = partial
        return [support(
            f"{known} of {total} characters read and consistent with the target "
            f"plate {want}; {total - known} unreadable -- treated as a weak match, "
            f"not a confirmation.", "plate")]
    if obs.plate.confidence >= PLATE_VETO_CONF:
        return [veto(
            f"Plate read {seen} (confidence {obs.plate.confidence:.2f}) "
            f"contradicts the target plate {want}.", "plate")]
    return [caution(
        f"Low-confidence plate read {seen} does not match target plate {want}; "
        f"not treated as decisive either way.", "plate")]


def check_transit(obs: Observation, profile: TargetProfile, graph: RoadGraph) -> list[Fact]:
    """The physics veto: you cannot outrun the road network."""
    last = profile.last_seen
    if last is None:
        return [info("No prior confirmed sighting; transit check not applicable.", "transit")]
    dt = obs.timestamp_s - last.timestamp_s
    if last.camera_id == obs.camera_id:
        if dt < 0:
            return [veto("Sighting predates the last confirmed sighting at the same camera.",
                         "transit")]
        return [info(f"Same camera as last sighting, {dt:.0f}s later.", "transit")]
    fastest = graph.min_transit_s(last.camera_id, obs.camera_id)
    if fastest is None:
        return [veto(
            f"No road path exists from {last.camera_id} to {obs.camera_id}.", "transit")]
    if dt < 0:
        return [veto("Sighting predates the last confirmed sighting.", "transit")]
    if dt < fastest:
        return [veto(
            f"Physically impossible transit: {last.camera_id} -> {obs.camera_id} "
            f"in {dt:.0f}s; the fastest possible route takes {fastest:.0f}s.", "transit")]
    window = graph.transit_window(last.camera_id, obs.camera_id)
    if window and window[0] <= dt <= window[1]:
        return [support(
            f"Transit {last.camera_id} -> {obs.camera_id} in {dt:.0f}s sits inside "
            f"the direct-hop window [{window[0]:.0f}s, {window[1]:.0f}s].", "transit")]
    return [info(
        f"Transit {last.camera_id} -> {obs.camera_id} in {dt:.0f}s is possible "
        f"but not a direct-hop timing; route or stops unknown.", "transit")]


def check_attributes(obs: Observation, profile: TargetProfile) -> list[Fact]:
    facts: list[Fact] = []
    want = profile.class_attrs
    got = obs.class_attrs
    if want.get("body_type") and got.get("body_type") and want["body_type"] != got["body_type"]:
        facts.append(veto(
            f"Body type contradiction: target is a {want['body_type']}, "
            f"sighting shows a {got['body_type']}.", "attributes"))
    matches = [k for k in ("make", "model", "color")
               if want.get(k) and got.get(k) and want[k] == got[k]]
    mismatches = [k for k in ("make", "model", "color")
                  if want.get(k) and got.get(k) and want[k] != got[k]]
    if want.get("body_type") and got.get("body_type") == want.get("body_type"):
        matches.append("body_type")
    if matches:
        facts.append(support(
            "Class attributes consistent: " +
            ", ".join(f"{k}={got[k]}" for k in matches) + ".", "attributes"))
    for k in mismatches:
        facts.append(caution(
            f"Class attribute mismatch on {k}: target {want[k]}, sighting {got[k]} "
            f"(classifier/camera noise is possible).", "attributes"))

    for kind, val in profile.instance_attrs.items():
        if kind.startswith(GEOM_PREFIX):
            continue  # 3D geometry has its own soft check; never a mark veto
        seen = obs.instance_attrs.get(kind)
        if seen is None:
            facts.append(info(
                f"Target's known mark '{val}' not visible in this sighting "
                f"(marks are often missed).", "attributes"))
        elif seen == val:
            facts.append(support(
                f"Distinguishing mark matches: {seen}.", "attributes"))
        else:
            facts.append(veto(
                f"Distinguishing-mark contradiction: target has '{val}', "
                f"sighting shows '{seen}'.", "attributes"))
    for kind, seen in obs.instance_attrs.items():
        if kind not in profile.instance_attrs and not kind.startswith(GEOM_PREFIX):
            facts.append(info(
                f"Sighting shows a mark not yet on the profile: {seen}.", "attributes"))
    return facts


GEOM_PREFIX = "geom3d:"


def check_geometry(obs: Observation, profile: TargetProfile) -> list[Fact]:
    """3D-geometry attribute comparison (from the car3d bridge).

    View-invariant proportion buckets measured on fused splat clouds. They
    enter at the attribute tier as support/caution only — sparse single-crop
    reconstructions mismeasure too often for geometry to carry veto power,
    and it must never act as the ReID tiebreaker. Runs only when BOTH sides
    actually carry geometry (the bridge withholds attrs until enough of the
    cloud is real evidence, so absence is common and meaningless).
    """
    obs_geom = {k: v for k, v in obs.instance_attrs.items() if k.startswith(GEOM_PREFIX)}
    prof_geom = {k: v for k, v in profile.instance_attrs.items()
                 if k.startswith(GEOM_PREFIX)}
    shared = sorted(set(obs_geom) & set(prof_geom))
    if not shared:
        return []
    matches = [k for k in shared if obs_geom[k] == prof_geom[k]]
    facts: list[Fact] = []
    if matches:
        facts.append(support(
            "3D geometry consistent: " +
            ", ".join(f"{k.removeprefix(GEOM_PREFIX)}={obs_geom[k]}" for k in matches) +
            " (view-invariant).", "geometry"))
    for k in shared:
        if obs_geom[k] != prof_geom[k]:
            facts.append(caution(
                f"3D geometry mismatch on {k.removeprefix(GEOM_PREFIX)}: target "
                f"{prof_geom[k]}, sighting {obs_geom[k]} — sparse reconstructions "
                f"mismeasure, so this cautions rather than vetoes.", "geometry"))
    return facts


def check_corroboration(obs: Observation, profile: TargetProfile, graph: RoadGraph) -> list[Fact]:
    """Was this sighting where we expected the target to appear next?"""
    last = profile.last_seen
    if last is None:
        return [info("No track yet; corroboration not applicable.", "corroboration")]
    dt = obs.timestamp_s - last.timestamp_s
    if dt > STALE_TRACK_S:
        return [caution(
            f"Last confirmed sighting was {dt:.0f}s ago; track is stale and "
            f"this sighting corroborates nothing.", "corroboration")]
    expected = set(graph.neighbors(last.camera_id)) | {last.camera_id}
    if obs.camera_id in expected:
        return [support(
            f"Sighting at {obs.camera_id} is consistent with the expected "
            f"next cameras after {last.camera_id}.", "corroboration")]
    return [caution(
        f"Sighting at {obs.camera_id} is not among the expected next cameras "
        f"after {last.camera_id}; possible but uncorroborated.", "corroboration")]


def run_all_checks(
    obs: Observation, profile: TargetProfile, graph: RoadGraph
) -> list[Fact]:
    """All checks, in the order the console displays them. Geometry is a
    no-op unless both sides carry 3D attributes from the car3d bridge."""
    return [
        *check_plate(obs, profile),
        *check_transit(obs, profile, graph),
        *check_attributes(obs, profile),
        *check_geometry(obs, profile),
        *check_corroboration(obs, profile, graph),
    ]
