"""Capped-additive corroboration: fusing repeated sightings without lying.

THE INDEPENDENCE TRAP. The textbook way to fuse repeated detections is
noisy-OR: belief = 1 - prod(1 - p_i). That formula assumes each camera's
error is independent — flip enough weak coins and certainty compounds.
For look-alike vehicles the assumption is exactly wrong: if camera A
mistakes the sibling Camry for the target, cameras B, C, and D will make
the *same* mistake, for the same reason (the vehicles genuinely look
alike; per-camera color casts add further correlated error). Under
noisy-OR, five 60%-confident sightings of the WRONG car compound to >99%
belief in a falsehood.

The fix used here:

1. Additive, diminishing increments instead of multiplicative compounding.
   Each consistent appearance-grade sighting adds base * decay^k — the
   k-th repetition of the same kind of evidence tells you less than the
   first did.
2. A hard cap on appearance-only credit (APPEARANCE_CAP), set BELOW the
   profile-update threshold. Consequence: no number of appearance-only
   corroborations — however many cameras agree — can push a track into
   auto-updating the target profile. Crossing that line requires
   *qualitatively independent* evidence: a plate read or a human.
3. Vetoed associations subtract belief; time decays it toward zero.

This is a design stance, not a learned model, and the constants are meant
to be read: the cap IS the policy "cameras agreeing is not proof".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from reasoning.cascade import (
    VERDICT_CONFIRMED, VERDICT_LIKELY, VERDICT_REJECTED, MatchDecision,
)
from reasoning.facts import Fact, caution, info, support

# Appearance-only belief can never exceed this...
APPEARANCE_CAP = 0.55
# ...which is deliberately below the threshold gating profile updates.
UPDATE_THRESHOLD = 0.75

BASE_INCREMENT = 0.25
INCREMENT_DECAY = 0.6
PLATE_BELIEF = 0.92          # a clean plate-confirmed sighting sets at least this
OPERATOR_BELIEF = 0.97       # an explicit human confirmation sets at least this
VETO_PENALTY = 0.25
BELIEF_HALF_LIFE_S = 600.0   # idle belief halves every 10 minutes


@dataclass(frozen=True)
class CorroborationState:
    """Current fused belief that we are correctly tracking a target."""

    target_id: str
    belief: float = 0.0
    appearance_credit: float = 0.0   # portion of belief owed to appearance alone
    appearance_events: int = 0
    last_update_s: float = 0.0
    contributing_events: tuple[str, ...] = ()

    @property
    def can_auto_update_profile(self) -> bool:
        return self.belief >= UPDATE_THRESHOLD


def decay(state: CorroborationState, now_s: float) -> CorroborationState:
    """Exponential decay toward zero while nothing corroborates the track."""
    dt = max(0.0, now_s - state.last_update_s)
    if dt == 0.0 or state.belief == 0.0:
        return replace(state, last_update_s=now_s)
    factor = math.pow(0.5, dt / BELIEF_HALF_LIFE_S)
    return replace(
        state,
        belief=state.belief * factor,
        appearance_credit=state.appearance_credit * factor,
        last_update_s=now_s,
    )


def apply_decision(
    state: CorroborationState, decision: MatchDecision, now_s: float
) -> tuple[CorroborationState, list[Fact]]:
    """Fold one cascade decision into the fused belief. Returns facts too —
    corroboration reasoning is shown to the operator like everything else."""
    state = decay(state, now_s)
    facts: list[Fact] = []

    if decision.verdict == VERDICT_REJECTED:
        new_belief = max(0.0, state.belief - VETO_PENALTY)
        facts.append(caution(
            f"Association vetoed; track belief reduced "
            f"{state.belief:.2f} -> {new_belief:.2f}.", "corroboration"))
        return replace(state, belief=new_belief), facts

    if decision.verdict == VERDICT_CONFIRMED:
        # Plate-grade evidence is qualitatively independent of appearance:
        # it may cross the update threshold.
        new_belief = max(state.belief, PLATE_BELIEF)
        facts.append(support(
            f"Plate-confirmed sighting raises track belief to {new_belief:.2f} "
            f"(threshold for automatic profile updates is {UPDATE_THRESHOLD:.2f}).",
            "corroboration"))
        return replace(
            state,
            belief=new_belief,
            contributing_events=(*state.contributing_events, decision.event_id),
        ), facts

    if decision.verdict == VERDICT_LIKELY:
        # Appearance-grade evidence: additive, diminishing, hard-capped.
        # Camera errors are correlated for look-alikes, so agreement across
        # cameras must NOT compound like independent evidence (no noisy-OR).
        increment = BASE_INCREMENT * (INCREMENT_DECAY ** state.appearance_events)
        headroom = max(0.0, APPEARANCE_CAP - state.appearance_credit)
        granted = min(increment, headroom)
        new_belief = min(1.0, state.belief + granted)
        capped = granted < increment
        text = (
            f"Consistent sighting adds {granted:.2f} belief "
            f"({state.appearance_events + 1} appearance-grade corroborations, "
            f"belief now {new_belief:.2f})."
        )
        if capped:
            text += (
                f" Appearance-only evidence is capped at {APPEARANCE_CAP:.2f}: "
                f"look-alike errors repeat across cameras, so more cameras "
                f"agreeing is not more proof. Plate or operator confirmation "
                f"required to go higher."
            )
        facts.append(support(text, "corroboration") if granted > 0
                     else info(text, "corroboration"))
        return replace(
            state,
            belief=new_belief,
            appearance_credit=state.appearance_credit + granted,
            appearance_events=state.appearance_events + 1,
            contributing_events=(*state.contributing_events, decision.event_id),
        ), facts

    facts.append(info("Undecided sighting; track belief unchanged.", "corroboration"))
    return state, facts


def apply_operator_confirmation(
    state: CorroborationState, now_s: float
) -> tuple[CorroborationState, list[Fact]]:
    """A human reviewed the crops and confirmed. Qualitatively independent."""
    state = decay(state, now_s)
    new_belief = max(state.belief, OPERATOR_BELIEF)
    return replace(state, belief=new_belief), [support(
        f"Operator confirmed the match; track belief set to {new_belief:.2f}.",
        "corroboration")]


def noisy_or(probabilities: list[float]) -> float:
    """The WRONG fusion rule for this problem — kept for tests and docs to
    demonstrate the trap, never used in the live pipeline."""
    p = 1.0
    for q in probabilities:
        p *= (1.0 - q)
    return 1.0 - p
