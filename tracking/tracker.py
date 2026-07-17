"""FleetTracker: the per-target orchestrator behind the live console.

Wires the pieces together for every incoming observation:

    cascade (identity + plausibility vetoes)
      -> lifecycle state machine
      -> capped-additive corroboration
      -> gated profile update (with snapshot record)
      -> review queue + event stream for the console

Ambiguity guard: when two flagged targets score within AMBIGUITY_MARGIN of
each other (the look-alike signature), the sighting is NOT associated with
either — it goes straight to review listing both candidates. Guessing
between look-alikes is precisely what this system refuses to automate.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Mapping

from perception.types import Observation
from reasoning.cascade import (
    VERDICT_CONFIRMED, VERDICT_LIKELY, CascadeConfig, MatchDecision, rank_candidates,
)
from reasoning.corroboration import (
    CorroborationState, apply_decision, apply_operator_confirmation,
    apply_operator_rejection,
)
from reasoning.facts import render_facts
from reasoning.profile import TargetProfile
from reasoning.updates import UpdateRecord, apply_update
from sim.road_graph import RoadGraph
from tracking.lifecycle import Track, on_association, on_rejection, on_tick
from tracking.predictor import predict_next_cameras
from tracking.smoother import SmootherState, init_state, predict, update

AMBIGUITY_MARGIN = 0.10


@dataclass(frozen=True)
class TrackerEvent:
    """Things the server broadcasts to the console/audit log."""

    kind: str            # association | review | state_change | profile_update
                         # | anomaly | rejection
    target_id: str
    event_id: str
    timestamp_s: float
    detail: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackedTarget:
    profile: TargetProfile
    corroboration: CorroborationState
    track: Track
    smoother: SmootherState | None = None
    update_records: tuple[UpdateRecord, ...] = ()


@dataclass(frozen=True)
class PendingReview:
    review_id: str
    target_id: str
    observation: Observation
    decision: MatchDecision
    # Other candidate target ids when the review exists because of ambiguity.
    rival_target_ids: tuple[str, ...] = ()


class FleetTracker:
    """Stateful across observations; all per-target values are immutable."""

    def __init__(self, graph: RoadGraph, cascade_config: CascadeConfig | None = None):
        self._graph = graph
        self._cascade_config = cascade_config
        self._targets: dict[str, TrackedTarget] = {}
        self._reviews: dict[str, PendingReview] = {}
        self._review_seq = itertools.count(1)

    # ------------------------------------------------------------- targets

    def flag_target(self, profile: TargetProfile) -> None:
        if profile.target_id in self._targets:
            raise ValueError(f"target {profile.target_id} already flagged")
        self._targets[profile.target_id] = TrackedTarget(
            profile=profile,
            corroboration=CorroborationState(target_id=profile.target_id),
            track=Track(target_id=profile.target_id),
        )

    def unflag_target(self, target_id: str) -> None:
        self._targets.pop(target_id, None)
        self._reviews = {k: v for k, v in self._reviews.items()
                        if v.target_id != target_id}

    def targets(self) -> dict[str, TrackedTarget]:
        return dict(self._targets)

    def pending_reviews(self) -> tuple[PendingReview, ...]:
        return tuple(self._reviews.values())

    # -------------------------------------------------------- observations

    def process_observation(self, obs: Observation) -> list[TrackerEvent]:
        if not self._targets:
            return []
        ranked = rank_candidates(
            obs, [t.profile for t in self._targets.values()],
            self._graph, self._cascade_config,
        )
        best = ranked.best
        events: list[TrackerEvent] = []

        if best.anomaly:
            events.append(self._queue_review(obs, best, kind="anomaly"))
        if best.verdict not in (VERDICT_CONFIRMED, VERDICT_LIKELY):
            if best.verdict == "rejected":
                tracked = self._targets[best.target_id]
                self._targets[best.target_id] = replace(
                    tracked, track=on_rejection(tracked.track, obs.timestamp_s))
            return events

        # Ambiguity guard: two candidates too close together.
        runner_up = next(
            (d for d in ranked.all_decisions[1:] if d.verdict == VERDICT_LIKELY), None)
        if (best.verdict == VERDICT_LIKELY and runner_up is not None
                and ranked.margin < AMBIGUITY_MARGIN):
            events.append(self._queue_review(
                obs, best, kind="review",
                rivals=(runner_up.target_id,),
                note=(f"Ambiguous between {best.target_id} and "
                      f"{runner_up.target_id} (margin {ranked.margin:.2f}); "
                      f"not auto-associated."),
            ))
            return events

        events.extend(self._associate(obs, best))
        return events

    def _associate(self, obs: Observation, decision: MatchDecision) -> list[TrackerEvent]:
        tracked = self._targets[decision.target_id]
        events: list[TrackerEvent] = []
        plate_grade = decision.verdict == VERDICT_CONFIRMED

        old_state = tracked.track.state
        track = on_association(tracked.track, obs.timestamp_s, plate_grade)
        corr, corr_facts = apply_decision(
            tracked.corroboration, decision, obs.timestamp_s)
        profile, record, gate = apply_update(
            tracked.profile, obs, decision, corr)
        smoother = (
            update(tracked.smoother, obs.lat, obs.lon, obs.timestamp_s)
            if tracked.smoother is not None
            else init_state(obs.lat, obs.lon, obs.timestamp_s)
        )
        self._targets[decision.target_id] = TrackedTarget(
            profile=profile,
            corroboration=corr,
            track=track,
            smoother=smoother,
            update_records=(
                (*tracked.update_records, record) if record else tracked.update_records),
        )

        events.append(TrackerEvent(
            kind="association", target_id=decision.target_id,
            event_id=obs.event_id, timestamp_s=obs.timestamp_s,
            detail={
                "verdict": decision.verdict, "score": decision.score,
                "tier": decision.deciding_tier, "belief": corr.belief,
                "camera_id": obs.camera_id,
                "facts": render_facts(list(decision.facts) + corr_facts),
            }))
        if track.state != old_state:
            events.append(TrackerEvent(
                kind="state_change", target_id=decision.target_id,
                event_id=obs.event_id, timestamp_s=obs.timestamp_s,
                detail={"from": old_state, "to": track.state}))
        if record is not None:
            events.append(TrackerEvent(
                kind="profile_update", target_id=decision.target_id,
                event_id=obs.event_id, timestamp_s=obs.timestamp_s,
                detail={"version": profile.version, "reason": gate.reason}))
        if decision.requires_review and not decision.anomaly:
            events.append(self._queue_review(obs, decision, kind="review"))
        return events

    def _queue_review(
        self, obs: Observation, decision: MatchDecision, kind: str,
        rivals: tuple[str, ...] = (), note: str = "",
    ) -> TrackerEvent:
        review_id = f"rev-{next(self._review_seq):04d}"
        self._reviews[review_id] = PendingReview(
            review_id=review_id, target_id=decision.target_id,
            observation=obs, decision=decision, rival_target_ids=rivals,
        )
        facts_text = render_facts(list(decision.facts))
        if note:
            facts_text = note + "\n" + facts_text
        return TrackerEvent(
            kind=kind, target_id=decision.target_id,
            event_id=obs.event_id, timestamp_s=obs.timestamp_s,
            detail={"review_id": review_id, "facts": facts_text,
                    "score": decision.score, "rivals": list(rivals)})

    # ------------------------------------------------------------- reviews

    def resolve_review(
        self, review_id: str, accept: bool, now_s: float
    ) -> list[TrackerEvent]:
        review = self._reviews.pop(review_id, None)
        if review is None:
            raise KeyError(f"unknown review {review_id}")
        tracked = self._targets.get(review.target_id)
        if tracked is None:
            return []
        obs = review.observation
        if accept:
            corr, facts = apply_operator_confirmation(
                tracked.corroboration, now_s)
            old_state = tracked.track.state
            track = on_association(tracked.track, now_s, plate_grade=True)
            profile, record, gate = apply_update(
                tracked.profile, obs, review.decision, corr,
                operator_confirmed=True)
            smoother = (
                update(tracked.smoother, obs.lat, obs.lon, obs.timestamp_s)
                if tracked.smoother is not None
                else init_state(obs.lat, obs.lon, obs.timestamp_s))
            self._targets[review.target_id] = TrackedTarget(
                profile=profile, corroboration=corr, track=track,
                smoother=smoother,
                update_records=(
                    (*tracked.update_records, record) if record
                    else tracked.update_records))
            events = [TrackerEvent(
                kind="association", target_id=review.target_id,
                event_id=obs.event_id, timestamp_s=now_s,
                detail={"verdict": "operator-confirmed", "belief": corr.belief,
                        "camera_id": obs.camera_id,
                        "facts": render_facts(facts)})]
            if track.state != old_state:
                events.append(TrackerEvent(
                    kind="state_change", target_id=review.target_id,
                    event_id=obs.event_id, timestamp_s=now_s,
                    detail={"from": old_state, "to": track.state}))
            if record is not None:
                events.append(TrackerEvent(
                    kind="profile_update", target_id=review.target_id,
                    event_id=obs.event_id, timestamp_s=now_s,
                    detail={"version": profile.version, "reason": gate.reason}))
            return events
        corr, facts = apply_operator_rejection(tracked.corroboration, now_s)
        self._targets[review.target_id] = replace(
            tracked, corroboration=corr,
            track=on_rejection(tracked.track, now_s))
        return [TrackerEvent(
            kind="rejection", target_id=review.target_id,
            event_id=obs.event_id, timestamp_s=now_s,
            detail={"facts": render_facts(facts)})]

    # ---------------------------------------------------------------- time

    def tick(self, now_s: float) -> list[TrackerEvent]:
        """Advance lifecycle clocks; emits state_change events."""
        events: list[TrackerEvent] = []
        for target_id, tracked in list(self._targets.items()):
            new_track = on_tick(tracked.track, now_s)
            if new_track.state != tracked.track.state:
                self._targets[target_id] = replace(tracked, track=new_track)
                events.append(TrackerEvent(
                    kind="state_change", target_id=target_id, event_id="",
                    timestamp_s=now_s,
                    detail={"from": tracked.track.state, "to": new_track.state}))
        return events

    # ------------------------------------------------------------ snapshot

    def snapshot(self, now_s: float) -> dict:
        """Console-ready view of every flagged target."""
        out = {}
        for target_id, t in self._targets.items():
            position = None
            if t.smoother is not None:
                lat, lon = predict(t.smoother, now_s)
                position = {"lat": lat, "lon": lon}
            predictions = []
            if t.profile.last_seen is not None:
                predictions = [
                    {"camera_id": p.camera_id, "status": p.status,
                     "window": [p.window_start_s, p.window_end_s]}
                    for p in predict_next_cameras(
                        self._graph, t.profile.last_seen.camera_id,
                        t.profile.last_seen.timestamp_s, now_s)
                ]
            out[target_id] = {
                "label": t.profile.label,
                "state": t.track.state,
                "belief": round(t.corroboration.belief, 3),
                "profile_version": t.profile.version,
                "plate": t.profile.plate,
                "position": position,
                "last_seen": (
                    {"camera_id": t.profile.last_seen.camera_id,
                     "timestamp_s": t.profile.last_seen.timestamp_s}
                    if t.profile.last_seen else None),
                "next_cameras": predictions,
            }
        return out
