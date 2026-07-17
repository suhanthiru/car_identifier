"""Identity cascade tests: tier ordering, veto finality, ReID-as-tiebreaker."""
import numpy as np
import pytest

from reasoning.cascade import (
    VERDICT_CANDIDATE, VERDICT_CONFIRMED, VERDICT_LIKELY, VERDICT_REJECTED,
    VERDICT_UNDECIDED, evaluate, rank_candidates,
)
from reasoning.profile import LastSeen
from sim.road_graph import default_world
from tests.util import CAMRY, make_obs, make_profile, unit_vec


@pytest.fixture(scope="module")
def graph():
    return default_world()


def test_clean_plate_match_confirms(graph):
    d = evaluate(make_obs(plate="ABC-1234"), make_profile(), graph)
    assert d.verdict == VERDICT_CONFIRMED
    assert d.deciding_tier == "plate"
    assert not d.requires_review
    assert d.score >= 0.85
    assert any("exactly matches" in f.text for f in d.facts)


def test_single_common_mark_is_refused_not_individuated(graph):
    """One distinguishing mark + class attributes is not enough to name one
    vehicle (feature B): distinctiveness sits below the floor, so the system
    refuses to individuate and returns a candidate set."""
    obs = make_obs(instance_attrs={"accessory": "roof rack"})
    profile = make_profile(plate="", instance_attrs={"accessory": "roof rack"})
    d = evaluate(obs, profile, graph)
    assert d.verdict == VERDICT_CANDIDATE
    assert d.refused_to_individuate and d.requires_review
    assert d.distinctiveness < 0.30


def test_two_marks_individuate(graph):
    """Two distinguishing marks push distinctiveness over the floor: the
    system may now name an individual (LIKELY, still human-reviewed)."""
    marks = {"accessory": "roof rack", "sticker": "oval bumper sticker"}
    d = evaluate(make_obs(instance_attrs=dict(marks)),
                 make_profile(plate="", instance_attrs=dict(marks)), graph)
    assert d.verdict == VERDICT_LIKELY
    assert not d.refused_to_individuate
    assert d.distinctiveness >= 0.30


def test_reid_cannot_rescue_zero_symbolic_evidence(graph):
    """ReID is a tiebreaker only: perfect appearance similarity with no
    symbolic support must not produce a match."""
    e = unit_vec(5)
    obs = make_obs(class_attrs={"make": "Ford", "model": "F-150",
                                "body_type": "pickup", "color": "black"},
                   embedding=e)
    profile = make_profile(plate="", class_attrs=dict(CAMRY), gallery=(e,))
    d = evaluate(obs, profile, graph)
    # body-type contradiction actually vetoes here; use attr-free profile too
    profile2 = make_profile(plate="", class_attrs={}, gallery=(e,))
    d2 = evaluate(make_obs(embedding=e, class_attrs={}), profile2, graph)
    assert d.verdict == VERDICT_REJECTED
    assert d2.verdict == VERDICT_UNDECIDED
    assert d2.score == 0.0, "reid alone contributed score without symbolic support"


def test_veto_is_final_even_with_plate_match(graph):
    last = LastSeen("cam-nw", 1000.0, "evt-0")
    obs = make_obs(camera_id="cam-e", t=1005.0, plate="ABC-1234")
    d = evaluate(obs, make_profile(last_seen=last), graph)
    assert d.verdict == VERDICT_REJECTED
    assert d.score == 0.0
    assert d.anomaly, "plate match + physics veto must flag a plate-clone anomaly"
    assert d.requires_review


def test_body_type_veto_rejects_without_anomaly(graph):
    d = evaluate(make_obs(class_attrs={**CAMRY, "body_type": "pickup"}), make_profile(plate=""), graph)
    assert d.verdict == VERDICT_REJECTED
    assert not d.anomaly


def test_score_capped_at_one(graph):
    e = unit_vec(9)
    obs = make_obs(plate="ABC-1234", instance_attrs={"accessory": "roof rack",
                                                     "sticker": "oval bumper sticker"},
                   embedding=e)
    profile = make_profile(instance_attrs={"accessory": "roof rack",
                                           "sticker": "oval bumper sticker"},
                           gallery=(e,))
    d = evaluate(obs, profile, graph)
    assert d.score <= 1.0


def test_reid_breaks_tie_between_lookalikes(graph):
    """Two flagged look-alikes, identical symbolic evidence; the embedding
    gallery decides the ranking — and the thin margin is visible."""
    e_obs = unit_vec(3)
    near = (e_obs + 0.05 * unit_vec(4)) / np.linalg.norm(e_obs + 0.05 * unit_vec(4))
    far = unit_vec(17)
    a = make_profile(target_id="tgt-a", plate="", gallery=(near.astype(np.float32),))
    b = make_profile(target_id="tgt-b", plate="", gallery=(far,))
    ranked = rank_candidates(make_obs(embedding=e_obs), [a, b], graph)
    assert ranked.best.target_id == "tgt-a"
    assert ranked.margin < 0.35, "lookalike margin should be thin"
    assert len(ranked.all_decisions) == 2


def test_rank_candidates_empty():
    assert rank_candidates(make_obs(), [], default_world()) is None


def test_every_decision_carries_facts(graph):
    d = evaluate(make_obs(plate="ABC-1234"), make_profile(), graph)
    assert len(d.facts) >= 4
    assert all(f.text for f in d.facts)
