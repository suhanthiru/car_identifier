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


def test_confusable_color_pair_treated_as_consistent():
    # silver vs gray: adjacent bins of the pixel heuristic (real CityFlow
    # footage reads the same car silver at one camera, gray at the next) —
    # support, not a caution.
    obs = make_obs(class_attrs={**CAMRY, "color": "gray"})
    facts = check_attributes(obs, make_profile())
    assert any(f.kind == KIND_SUPPORT and "adjacent bins" in f.text for f in facts)
    assert KIND_CAUTION not in {f.kind for f in facts}


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


def test_cloned_plate_on_a_different_make_is_an_anomaly_not_a_confirmation():
    """A plate match must not confirm a car the attributes contradict.

    W_PLATE_EXACT (0.90) alone clears CONFIRM_THRESHOLD (0.85), and only
    body_type used to veto — so an exact plate on a Honda Civic auto-CONFIRMED
    against a Toyota Camry target (same body_type, nothing objected), merged
    that vehicle's evidence into the target's profile, and queued nothing for
    review. A cloned plate is the most likely way this system ever meets the
    wrong car; it was the one case the attribute check waved through.

    Vetoing routes it correctly rather than merely blocking it: a veto WITH an
    exact plate is what the cascade already calls an anomaly — the
    plate-clone / clock-skew case a human is supposed to see.
    """
    facts = check_attributes(
        make_obs(class_attrs={"make": "Honda", "model": "Civic",
                          "body_type": "sedan", "color": "black"}),
        make_profile(class_attrs={"make": "Toyota", "model": "Camry",
                              "body_type": "sedan", "color": "silver"}))
    assert any(f.kind == "veto" for f in facts), \
        f"make/model contradiction did not veto: {[f.text for f in facts]}"


def test_colour_difference_alone_still_only_cautions():
    """Colour must NOT join the veto list: it is a mean-pixel heuristic whose
    adjacent bins flip under lighting (CONFUSABLE_COLORS). Make and model come
    from labels or a classifier; colour does not."""
    facts = check_attributes(
        make_obs(class_attrs={"make": "Toyota", "model": "Camry",
                          "body_type": "sedan", "color": "red"}),
        make_profile(class_attrs={"make": "Toyota", "model": "Camry",
                              "body_type": "sedan", "color": "blue"}))
    assert not any(f.kind == "veto" for f in facts)


def test_a_nan_timestamp_vetoes_instead_of_disabling_the_physics_check():
    """Every transit veto is a `dt < x` test, and every comparison with NaN is
    False — so a NaN timestamp made `dt < 0` and `dt < fastest` both fail, the
    check fell through to an info fact, and a plate match then confirmed
    unopposed. The veto the cascade documents as final, disabled by one bad
    float. NaN is not exotic: a failed parse or a missing field defaulting
    through the pipeline produces it. A check that cannot run must not read as
    a pass.
    """
    graph = default_world()
    prof = make_profile(last_seen=LastSeen("cam-nw", 1000.0, "e0"))
    for bad in (float("nan"),):
        facts = check_transit(make_obs(camera_id="cam-s", t=bad), prof, graph)
        assert any(f.kind == "veto" for f in facts), \
            f"timestamp {bad} did not veto: {[f.text for f in facts]}"
