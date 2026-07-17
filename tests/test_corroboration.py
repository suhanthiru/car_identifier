"""Corroboration fusion tests, centered on the independence trap."""
import pytest

from reasoning.cascade import (
    VERDICT_CONFIRMED, VERDICT_LIKELY, VERDICT_REJECTED, VERDICT_UNDECIDED,
    MatchDecision,
)
from reasoning.corroboration import (
    APPEARANCE_CAP, UPDATE_THRESHOLD, CorroborationState,
    apply_decision, apply_operator_confirmation, decay, noisy_or,
)


def make_decision(verdict: str, event_id: str = "evt-1", score: float = 0.5) -> MatchDecision:
    return MatchDecision(
        target_id="tgt-1", event_id=event_id, verdict=verdict, score=score,
        deciding_tier="attributes", facts=(), reid_similarity=0.8,
        requires_review=verdict == VERDICT_LIKELY,
    )


def test_cap_is_below_update_threshold():
    """The structural invariant of the whole design."""
    assert APPEARANCE_CAP < UPDATE_THRESHOLD


def test_noisy_or_demonstrates_the_trap():
    """Six weak correlated sightings under noisy-OR reach near-certainty —
    the failure mode this module exists to avoid."""
    assert noisy_or([0.6] * 6) > 0.99


def test_correlated_lookalike_pair_cannot_reach_update_threshold():
    """THE Phase-4 requirement: two look-alike appearance matches (and even
    twenty) must never push belief over the auto-update threshold."""
    state = CorroborationState(target_id="tgt-1")
    for i in range(20):
        state, _ = apply_decision(state, make_decision(VERDICT_LIKELY, f"evt-{i}"), float(i))
    assert state.belief <= APPEARANCE_CAP + 1e-9
    assert not state.can_auto_update_profile


def test_increments_diminish():
    state = CorroborationState(target_id="tgt-1")
    beliefs = []
    for i in range(4):
        state, _ = apply_decision(state, make_decision(VERDICT_LIKELY, f"evt-{i}"), float(i))
        beliefs.append(state.belief)
    gains = [b - a for a, b in zip([0.0] + beliefs, beliefs)]
    assert all(g2 < g1 for g1, g2 in zip(gains, gains[1:])), gains


def test_plate_confirmation_crosses_threshold():
    state = CorroborationState(target_id="tgt-1")
    state, facts = apply_decision(state, make_decision(VERDICT_CONFIRMED), 10.0)
    assert state.belief >= UPDATE_THRESHOLD
    assert state.can_auto_update_profile
    assert any("Plate-confirmed" in f.text for f in facts)


def test_operator_confirmation_crosses_threshold():
    state = CorroborationState(target_id="tgt-1")
    state, _ = apply_operator_confirmation(state, 10.0)
    assert state.can_auto_update_profile


def test_veto_subtracts_belief():
    state = CorroborationState(target_id="tgt-1", belief=0.5, last_update_s=10.0)
    state, facts = apply_decision(state, make_decision(VERDICT_REJECTED), 10.0)
    assert state.belief == pytest.approx(0.25)
    assert any("vetoed" in f.text for f in facts)


def test_undecided_changes_nothing():
    state = CorroborationState(target_id="tgt-1", belief=0.3, last_update_s=5.0)
    after, _ = apply_decision(state, make_decision(VERDICT_UNDECIDED), 5.0)
    assert after.belief == pytest.approx(0.3)


def test_belief_decays_over_time():
    state = CorroborationState(target_id="tgt-1", belief=0.8, last_update_s=0.0)
    later = decay(state, 600.0)  # one half-life
    assert later.belief == pytest.approx(0.4, abs=0.01)
    assert decay(state, 0.0).belief == pytest.approx(0.8)


def test_capped_fact_explains_why():
    """When the cap binds, the operator must be told why in plain English."""
    state = CorroborationState(target_id="tgt-1")
    texts = []
    for i in range(6):
        state, facts = apply_decision(state, make_decision(VERDICT_LIKELY, f"evt-{i}"), float(i))
        texts.extend(f.text for f in facts)
    assert any("not more proof" in t for t in texts)
