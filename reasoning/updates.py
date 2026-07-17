"""Gated, reversible profile updates.

A target profile only changes when the evidence gate opens, and every
change is recorded with its full before-state so an operator rejecting a
chain of automated decisions rolls the profile back losslessly.

Gate policy (in order):
- vetoed/rejected decisions never update anything;
- plate-confirmed decisions may update automatically;
- operator confirmation may update anything;
- appearance-grade ("likely") decisions may NOT update the profile, no
  matter how high the corroboration belief — and because appearance-only
  belief is capped below UPDATE_THRESHOLD (see corroboration.py), the
  belief test alone would already block them. Both conditions are checked;
  defense in depth against future constant-tweaking.
"""
from __future__ import annotations

from dataclasses import dataclass

from perception.types import Observation
from reasoning.cascade import VERDICT_CONFIRMED, MatchDecision
from reasoning.corroboration import UPDATE_THRESHOLD, CorroborationState
from reasoning.profile import LastSeen, TargetProfile


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    reason: str  # plain English, shown in the audit log either way


@dataclass(frozen=True)
class UpdateRecord:
    """Audit entry: one profile transition, reversible by construction."""

    target_id: str
    event_id: str
    timestamp_s: float
    reason: str
    before: TargetProfile
    after: TargetProfile


def evaluate_gate(
    decision: MatchDecision,
    state: CorroborationState,
    operator_confirmed: bool = False,
) -> GateResult:
    if operator_confirmed:
        return GateResult(True, "Operator explicitly confirmed this match.")
    if decision.verdict == VERDICT_CONFIRMED and state.belief >= UPDATE_THRESHOLD:
        return GateResult(True, (
            f"Plate-confirmed match with track belief {state.belief:.2f} "
            f">= {UPDATE_THRESHOLD:.2f}."))
    if decision.verdict == VERDICT_CONFIRMED:
        return GateResult(False, (
            f"Plate matched but track belief {state.belief:.2f} is below "
            f"{UPDATE_THRESHOLD:.2f}; waiting for corroboration."))
    return GateResult(False, (
        f"Evidence is appearance-grade ({decision.verdict}); profile updates "
        f"require a plate read or operator confirmation. Look-alike errors "
        f"are correlated across cameras, so repeated sightings alone never "
        f"open this gate."))


def apply_update(
    profile: TargetProfile,
    obs: Observation,
    decision: MatchDecision,
    state: CorroborationState,
    operator_confirmed: bool = False,
) -> tuple[TargetProfile, UpdateRecord | None, GateResult]:
    """Apply a gated profile update. Denied gate -> unchanged profile."""
    gate = evaluate_gate(decision, state, operator_confirmed)
    if not gate.allowed:
        return profile, None, gate
    learned_plate = ""
    if obs.plate is not None and obs.plate.text == profile.plate:
        learned_plate = obs.plate.text
    elif obs.plate is not None and not profile.plate and operator_confirmed:
        # Learning a plate the operator vouched for is allowed; automated
        # plate learning from unvetted reads is not.
        learned_plate = obs.plate.text
    after = profile.with_sighting(
        embedding=obs.embedding,
        last_seen=LastSeen(obs.camera_id, obs.timestamp_s, obs.event_id),
        observed_instance_attrs=obs.instance_attrs,
        learned_plate=learned_plate,
    )
    record = UpdateRecord(
        target_id=profile.target_id,
        event_id=obs.event_id,
        timestamp_s=obs.timestamp_s,
        reason=gate.reason,
        before=profile,
        after=after,
    )
    return after, record, gate


def rollback(record: UpdateRecord, current: TargetProfile) -> TargetProfile:
    """Undo one recorded update.

    Only valid when `current` is the direct result of `record` (no later
    updates stacked on top); otherwise the caller must roll back newest
    first, which the version counter enforces.
    """
    if current.version != record.after.version:
        raise ValueError(
            f"cannot roll back out of order: profile at v{current.version}, "
            f"record produced v{record.after.version}")
    return record.before
