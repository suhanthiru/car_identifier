"""Gate + reversible-update tests."""
import pytest

from reasoning.cascade import VERDICT_CONFIRMED, VERDICT_LIKELY, MatchDecision
from reasoning.corroboration import CorroborationState
from reasoning.profile import MAX_GALLERY
from reasoning.updates import apply_update, evaluate_gate, rollback
from tests.util import make_obs, make_profile, unit_vec


def decision(verdict: str) -> MatchDecision:
    return MatchDecision(
        target_id="tgt-1", event_id="evt-9", verdict=verdict, score=0.9,
        deciding_tier="plate", facts=(), reid_similarity=0.9,
        requires_review=False,
    )


def state(belief: float) -> CorroborationState:
    return CorroborationState(target_id="tgt-1", belief=belief, last_update_s=0.0)


def test_appearance_grade_never_opens_gate():
    gate = evaluate_gate(decision(VERDICT_LIKELY), state(0.99))
    assert not gate.allowed
    assert "correlated" in gate.reason


def test_plate_with_low_belief_stays_closed():
    gate = evaluate_gate(decision(VERDICT_CONFIRMED), state(0.3))
    assert not gate.allowed


def test_plate_with_belief_opens_gate():
    gate = evaluate_gate(decision(VERDICT_CONFIRMED), state(0.92))
    assert gate.allowed


def test_operator_overrides_everything():
    gate = evaluate_gate(decision(VERDICT_LIKELY), state(0.0), operator_confirmed=True)
    assert gate.allowed


def test_denied_update_leaves_profile_untouched():
    profile = make_profile()
    obs = make_obs()
    after, record, gate = apply_update(profile, obs, decision(VERDICT_LIKELY), state(0.9))
    assert after is profile
    assert record is None
    assert not gate.allowed


def test_allowed_update_appends_and_is_reversible():
    profile = make_profile()
    obs = make_obs(plate="ABC-1234", instance_attrs={"accessory": "roof rack"})
    after, record, gate = apply_update(profile, obs, decision(VERDICT_CONFIRMED), state(0.92))
    assert gate.allowed
    assert after.version == profile.version + 1
    assert len(after.gallery) == 1
    assert after.instance_attrs["accessory"] == "roof rack"
    assert after.last_seen.event_id == obs.event_id
    assert rollback(record, after) == profile


def test_rollback_out_of_order_refused():
    profile = make_profile()
    obs1 = make_obs(event_id="evt-1", plate="ABC-1234")
    obs2 = make_obs(event_id="evt-2", plate="ABC-1234", t=1200.0)
    p1, rec1, _ = apply_update(profile, obs1, decision(VERDICT_CONFIRMED), state(0.92))
    p2, rec2, _ = apply_update(p1, obs2, decision(VERDICT_CONFIRMED), state(0.92))
    with pytest.raises(ValueError):
        rollback(rec1, p2)
    assert rollback(rec2, p2) == p1


def test_plate_learning_requires_operator():
    """A target flagged without a plate must not learn one from an
    unvetted automated read."""
    profile = make_profile(plate="")
    obs = make_obs(plate="NEW-1111")
    # Simulate operator-confirmed review of a likely match.
    after, record, gate = apply_update(
        profile, obs, decision(VERDICT_LIKELY), state(0.2), operator_confirmed=True)
    assert gate.allowed
    assert after.plate == "NEW-1111"
    # Without the operator, even a confirmed-verdict update cannot invent a
    # plate for a plate-less profile (verdict would be impossible anyway).
    after2, _, gate2 = apply_update(profile, obs, decision(VERDICT_CONFIRMED), state(0.92))
    assert gate2.allowed
    assert after2.plate == ""


def test_gallery_bounded():
    profile = make_profile()
    st = state(0.92)
    for i in range(MAX_GALLERY + 5):
        obs = make_obs(event_id=f"evt-{i}", plate="ABC-1234",
                       t=1000.0 + i * 120, embedding=unit_vec(i))
        profile, _, _ = apply_update(profile, obs, decision(VERDICT_CONFIRMED), st)
    assert len(profile.gallery) == MAX_GALLERY
