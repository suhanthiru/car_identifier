"""Parity tests: structured signals must match the fact-based interpretation.

Guards the refactor that replaced substring-matching of fact text with typed
signals — the two must never drift.
"""
import pytest

from reasoning.facts import KIND_SUPPORT, has_veto
from reasoning.plausibility import run_all_checks
from reasoning.signals import compute_signals
from reasoning.profile import LastSeen
from sim.road_graph import default_world
from tests.util import CAMRY, make_obs, make_profile


@pytest.fixture(scope="module")
def graph():
    return default_world()


def _scenarios(graph):
    win = graph.transit_window("cam-n", "cam-ctr")
    mid = (win[0] + win[1]) / 2
    last = LastSeen("cam-n", 1000.0, "evt-0")
    return {
        "plate_exact": (make_obs(plate="ABC-1234"), make_profile()),
        "plate_near": (make_obs(plate="ABC-1Z34"), make_profile(plate="ABC-1234")),
        "plate_contradiction": (make_obs(plate="XYZ-9999", plate_conf=0.95), make_profile()),
        "plate_lowconf": (make_obs(plate="XYZ-9999", plate_conf=0.5), make_profile()),
        "body_veto": (make_obs(class_attrs={**CAMRY, "body_type": "pickup"}), make_profile()),
        "color_mismatch": (make_obs(class_attrs={**CAMRY, "color": "black"}), make_profile()),
        "mark_match": (make_obs(instance_attrs={"accessory": "roof rack"}),
                       make_profile(instance_attrs={"accessory": "roof rack"})),
        "mark_veto": (make_obs(instance_attrs={"accessory": "tow hitch"}),
                      make_profile(instance_attrs={"accessory": "roof rack"})),
        "transit_ok": (make_obs(camera_id="cam-ctr", t=1000.0 + mid),
                       make_profile(last_seen=last)),
        "transit_veto": (make_obs(camera_id="cam-e", t=1010.0),
                         make_profile(last_seen=LastSeen("cam-nw", 1000.0, "evt-0"))),
        "geometry": (make_obs(instance_attrs={"geom3d:body_profile": "low"}),
                     make_profile(instance_attrs={"geom3d:body_profile": "low"})),
    }


def test_signal_parity_with_facts(graph):
    for name, (obs, profile) in _scenarios(graph).items():
        facts = run_all_checks(obs, profile, graph)
        s = compute_signals(obs, profile, graph)
        plate_support = [f for f in facts if f.check == "plate" and f.kind == KIND_SUPPORT]
        assert s.plate_exact == any("exactly matches" in f.text for f in plate_support), name
        assert s.plate_near == any("OCR-confusable" in f.text for f in plate_support), name
        assert s.attrs_consistent == any(
            f.check == "attributes" and f.kind == KIND_SUPPORT
            and f.text.startswith("Class attributes") for f in facts), name
        assert s.mark_match_count == sum(
            1 for f in facts if f.check == "attributes" and f.kind == KIND_SUPPORT
            and "mark matches" in f.text), name
        assert s.geometry_consistent == any(
            f.check == "geometry" and f.kind == KIND_SUPPORT for f in facts), name
        assert s.any_veto == has_veto(facts), name


def test_transit_fastest_populated(graph):
    last = LastSeen("cam-nw", 1000.0, "evt-0")
    # adjacent hop: fastest == direct edge min
    s = compute_signals(make_obs(camera_id="cam-n", t=1400.0), make_profile(last_seen=last), graph)
    assert s.transit_fastest_s == pytest.approx(graph.min_transit_s("cam-nw", "cam-n"))
    assert s.transit_dt_s == pytest.approx(400.0)
    # multi-hop: still populated, larger than one edge
    s2 = compute_signals(make_obs(camera_id="cam-e", t=2000.0), make_profile(last_seen=last), graph)
    assert s2.transit_fastest_s == pytest.approx(graph.min_transit_s("cam-nw", "cam-e"))
    # same camera: no fastest
    s3 = compute_signals(make_obs(camera_id="cam-nw", t=1400.0), make_profile(last_seen=last), graph)
    assert s3.same_camera and s3.transit_fastest_s is None
    # no history: not applicable
    s4 = compute_signals(make_obs(), make_profile(), graph)
    assert not s4.transit_applicable and s4.transit_fastest_s is None


def test_veto_flags_specific(graph):
    s = compute_signals(make_obs(class_attrs={**CAMRY, "body_type": "pickup"}), make_profile(), graph)
    assert s.body_veto and s.any_veto
    s2 = compute_signals(make_obs(plate="XYZ-9999", plate_conf=0.95), make_profile(), graph)
    assert s2.plate_contradiction and s2.any_veto


def test_plate_partial_match_signal_populated(graph):
    s = compute_signals(make_obs(plate="ABC-1_34", plate_conf=0.95), make_profile(), graph)
    assert s.plate_partial_match
    assert not s.plate_exact and not s.plate_near and not s.plate_contradiction
    assert s.plate_known_chars == 7
    assert s.plate_total_chars == 8
    assert not s.any_veto


def test_plate_partial_match_with_contradiction_is_still_a_veto(graph):
    s = compute_signals(make_obs(plate="ABC-1_99", plate_conf=0.95), make_profile(), graph)
    assert not s.plate_partial_match
    assert s.plate_contradiction and s.any_veto
