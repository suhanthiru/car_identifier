"""Unit tests for the four symbolic plausibility checks."""
import pytest

from reasoning.facts import (
    KIND_CAUTION, KIND_INFO, KIND_SUPPORT, KIND_VETO, Fact, has_veto, render_facts,
)
from reasoning.plausibility import (
    check_attributes, check_corroboration, check_plate, check_transit, run_all_checks,
)
from reasoning.profile import LastSeen
from sim.road_graph import default_world
from tests.util import CAMRY, make_obs, make_profile


@pytest.fixture(scope="module")
def graph():
    return default_world()


# ---------------------------------------------------------------- facts core

def test_fact_validation():
    with pytest.raises(ValueError):
        Fact("bogus", "text")
    with pytest.raises(ValueError):
        Fact(KIND_INFO, "   ")


def test_render_facts_prefixes():
    out = render_facts([Fact(KIND_SUPPORT, "a"), Fact(KIND_VETO, "b"),
                        Fact(KIND_CAUTION, "c"), Fact(KIND_INFO, "d")])
    assert out.splitlines() == ["[+] a", "[X] b", "[!] c", "[i] d"]


# --------------------------------------------------------------- plate check

def test_plate_exact_match_supports():
    facts = check_plate(make_obs(plate="ABC-1234"), make_profile())
    assert [f.kind for f in facts] == [KIND_SUPPORT]
    assert "exactly matches" in facts[0].text


def test_plate_ocr_confusion_is_weak_match():
    # Profile plate ABC-1234; read ABC-1Z34: 2<->Z is a known confusion pair.
    facts = check_plate(make_obs(plate="ABC-1Z34"), make_profile(plate="ABC-1234"))
    assert facts[0].kind == KIND_SUPPORT
    assert "OCR-confusable" in facts[0].text


def test_plate_clean_mismatch_vetoes():
    facts = check_plate(make_obs(plate="XYZ-9999", plate_conf=0.95), make_profile())
    assert has_veto(facts)


def test_plate_low_conf_mismatch_only_cautions():
    facts = check_plate(make_obs(plate="XYZ-9999", plate_conf=0.5), make_profile())
    assert [f.kind for f in facts] == [KIND_CAUTION]


def test_plate_absent_cases_are_info():
    assert check_plate(make_obs(), make_profile())[0].kind == KIND_INFO
    assert check_plate(make_obs(plate="ABC-1234"), make_profile(plate=""))[0].kind == KIND_INFO


def test_plate_partial_read_with_known_agreement_is_weak_support():
    """Real ALPR on oblique footage: some characters read, some masked ('_'),
    and every character that WAS read agrees with the target plate."""
    facts = check_plate(make_obs(plate="ABC-1_34", plate_conf=0.95), make_profile())
    assert facts[0].kind == KIND_SUPPORT
    assert "7 of 8" in facts[0].text
    assert "unreadable" in facts[0].text


def test_plate_partial_read_with_known_contradiction_still_vetoes():
    """A masked character elsewhere does not launder a real contradiction
    at a KNOWN position -- this must veto exactly as a full clean mismatch."""
    facts = check_plate(make_obs(plate="ABC-1_99", plate_conf=0.95), make_profile())
    assert has_veto(facts)


# ------------------------------------------------------------- transit check

def test_transit_without_history_is_info(graph):
    facts = check_transit(make_obs(), make_profile(), graph)
    assert facts[0].kind == KIND_INFO


def test_transit_impossible_hop_vetoes(graph):
    last = LastSeen("cam-nw", 1000.0, "evt-0")
    # cam-nw -> cam-e requires at least two hops; 10 seconds is impossible.
    obs = make_obs(camera_id="cam-e", t=1010.0)
    facts = check_transit(obs, make_profile(last_seen=last), graph)
    assert has_veto(facts)
    assert "impossible" in facts[0].text.lower()


def test_transit_within_direct_window_supports(graph):
    window = graph.transit_window("cam-n", "cam-ctr")
    mid = (window[0] + window[1]) / 2
    last = LastSeen("cam-n", 1000.0, "evt-0")
    obs = make_obs(camera_id="cam-ctr", t=1000.0 + mid)
    facts = check_transit(obs, make_profile(last_seen=last), graph)
    assert facts[0].kind == KIND_SUPPORT


def test_transit_slow_but_possible_is_info(graph):
    window = graph.transit_window("cam-n", "cam-ctr")
    last = LastSeen("cam-n", 1000.0, "evt-0")
    obs = make_obs(camera_id="cam-ctr", t=1000.0 + window[1] + 500)
    facts = check_transit(obs, make_profile(last_seen=last), graph)
    assert facts[0].kind == KIND_INFO


def test_transit_negative_dt_vetoes(graph):
    last = LastSeen("cam-n", 1000.0, "evt-0")
    facts = check_transit(make_obs(camera_id="cam-n", t=900.0),
                          make_profile(last_seen=last), graph)
    assert has_veto(facts)


# ---------------------------------------------------------- attributes check

def test_body_type_contradiction_vetoes():
    obs = make_obs(class_attrs={**CAMRY, "body_type": "pickup"})
    assert has_veto(check_attributes(obs, make_profile()))


def test_color_mismatch_only_cautions():
    obs = make_obs(class_attrs={**CAMRY, "color": "black"})
    facts = check_attributes(obs, make_profile())
    kinds = {f.kind for f in facts}
    assert KIND_CAUTION in kinds and KIND_VETO not in kinds


def test_matching_mark_supports_and_conflicting_mark_vetoes():
    profile = make_profile(instance_attrs={"accessory": "roof rack"})
    match = check_attributes(make_obs(instance_attrs={"accessory": "roof rack"}), profile)
    assert any(f.kind == KIND_SUPPORT and "mark matches" in f.text for f in match)
    clash = check_attributes(make_obs(instance_attrs={"accessory": "tow hitch"}), profile)
    assert has_veto(clash)


def test_missing_mark_is_not_a_contradiction():
    profile = make_profile(instance_attrs={"sticker": "university decal"})
    facts = check_attributes(make_obs(instance_attrs={}), profile)
    assert not has_veto(facts)
    assert any("not visible" in f.text for f in facts)


# ------------------------------------------------------- corroboration check

def test_corroboration_expected_camera_supports(graph):
    last = LastSeen("cam-n", 1000.0, "evt-0")
    obs = make_obs(camera_id="cam-ctr", t=1080.0)
    facts = check_corroboration(obs, make_profile(last_seen=last), graph)
    assert facts[0].kind == KIND_SUPPORT


def test_corroboration_unexpected_camera_cautions(graph):
    last = LastSeen("cam-n", 1000.0, "evt-0")
    obs = make_obs(camera_id="cam-sw", t=1200.0)
    facts = check_corroboration(obs, make_profile(last_seen=last), graph)
    assert facts[0].kind == KIND_CAUTION


def test_corroboration_stale_track_cautions(graph):
    last = LastSeen("cam-n", 1000.0, "evt-0")
    obs = make_obs(camera_id="cam-ctr", t=1000.0 + 3600.0)
    facts = check_corroboration(obs, make_profile(last_seen=last), graph)
    assert "stale" in facts[0].text


# ---------------------------------------------------------------- aggregate

def test_run_all_checks_orders_by_check(graph):
    facts = run_all_checks(make_obs(plate="ABC-1234"), make_profile(), graph)
    checks = [f.check for f in facts]
    assert checks == sorted(checks, key=["plate", "transit", "attributes",
                                         "corroboration"].index)
