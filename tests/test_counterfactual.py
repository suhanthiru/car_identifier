"""Counterfactual engine: flip points must equal the live decision boundary."""
import pytest

from reasoning.cascade import (
    VERDICT_CANDIDATE, VERDICT_CONFIRMED, VERDICT_REJECTED, CascadeConfig, evaluate,
)
from reasoning.counterfactual import counterfactuals
from reasoning.profile import LastSeen
from reasoning.signals import MatchSignals
from sim.road_graph import default_world
from tests.util import CAMRY, make_obs, make_profile


@pytest.fixture(scope="module")
def graph():
    return default_world()


def test_transit_veto_flip_point_equals_min_transit():
    # Plate matches but the hop is impossible: rejected, and the boundary is
    # exactly the fastest possible transit time.
    sig = MatchSignals(plate_exact=True, transit_applicable=True,
                       transit_dt_s=40.0, transit_fastest_s=120.0, transit_veto=True)
    cfg = CascadeConfig()
    cfs = counterfactuals(sig, cfg)
    t = next(c for c in cfs if c.signal == "transit")
    assert "120" in t.boundary
    assert t.current_outcome == VERDICT_REJECTED
    assert t.flipped_outcome == VERDICT_CONFIRMED  # plate, no veto -> confirm
    assert "120" in t.text


def test_transit_accepted_would_reject_if_faster():
    sig = MatchSignals(plate_exact=True, attrs_consistent=True,
                       transit_applicable=True, transit_dt_s=300.0,
                       transit_fastest_s=90.0, transit_veto=False)
    t = next(c for c in counterfactuals(sig) if c.signal == "transit")
    assert t.flipped_outcome == VERDICT_REJECTED
    assert "faster than" in t.text and "90" in t.text


def test_plate_flip_to_ambiguous():
    sig = MatchSignals(plate_exact=True, attrs_consistent=True)  # CONFIRMED
    p = next(c for c in counterfactuals(sig) if c.signal == "plate")
    assert p.current_outcome == VERDICT_CONFIRMED
    assert p.flipped_outcome in (VERDICT_CANDIDATE, "undecided")
    assert "without the plate" in p.text.lower()


def test_distinctiveness_counterfactual_on_refusal():
    # class + reid pushes score over the line but distinctiveness is low.
    sig = MatchSignals(attrs_consistent=True, has_gallery=True,
                       reid_similarity=0.99, reid_prob=1.0)
    cfs = counterfactuals(sig)
    d = next(c for c in cfs if c.signal == "distinctiveness")
    assert d.current_outcome == VERDICT_CANDIDATE
    assert "plate read" in d.text


def test_body_veto_counterfactual():
    sig = MatchSignals(body_veto=True, plate_exact=True)
    b = next(c for c in counterfactuals(sig) if c.signal == "body_style")
    assert b.current_outcome == VERDICT_REJECTED
    assert "body style agreed" in b.text


def test_counterfactuals_attached_to_decision(graph):
    last = LastSeen("cam-nw", 1000.0, "evt-0")
    obs = make_obs(camera_id="cam-e", t=1005.0, plate="ABC-1234")
    d = evaluate(obs, make_profile(last_seen=last), graph)
    assert d.verdict == VERDICT_REJECTED
    assert d.counterfactuals, "decision must carry counterfactuals"
    assert any(c.signal == "transit" for c in d.counterfactuals)


def test_no_transit_counterfactual_without_history(graph):
    d = evaluate(make_obs(plate="ABC-1234"), make_profile(), graph)
    assert not any(c.signal == "transit" for c in d.counterfactuals)
