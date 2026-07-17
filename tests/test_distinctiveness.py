"""Distinctiveness floor: the system refuses to individuate on generic evidence."""
import pytest

from reasoning.cascade import (
    VERDICT_CANDIDATE, VERDICT_CONFIRMED, VERDICT_LIKELY, CascadeConfig, evaluate,
)
from reasoning.distinctiveness import distinctiveness
from reasoning.signals import MatchSignals
from sim.road_graph import default_world
from tests.util import CAMRY, make_obs, make_profile


@pytest.fixture(scope="module")
def graph():
    return default_world()


def test_distinctiveness_ordering():
    plate = distinctiveness(MatchSignals(plate_exact=True))
    two_marks = distinctiveness(MatchSignals(attrs_consistent=True, mark_match_count=2))
    one_mark = distinctiveness(MatchSignals(attrs_consistent=True, mark_match_count=1))
    class_only = distinctiveness(MatchSignals(attrs_consistent=True))
    assert plate == 1.0
    assert plate > two_marks > one_mark > class_only > 0
    assert class_only < 0.30 <= two_marks


def test_reid_never_feeds_distinctiveness():
    """A high appearance similarity between look-alikes must not inflate
    distinctiveness — that is the whole failure mode B guards against."""
    lo = distinctiveness(MatchSignals(attrs_consistent=True, has_gallery=True,
                                      reid_similarity=0.99, reid_prob=0.99))
    plain = distinctiveness(MatchSignals(attrs_consistent=True))
    assert lo == plain


def test_plated_target_individuates(graph):
    d = evaluate(make_obs(plate="ABC-1234"), make_profile(), graph)
    assert d.verdict == VERDICT_CONFIRMED
    assert not d.refused_to_individuate
    assert d.distinctiveness == 1.0


def test_generic_target_refuses(graph):
    """The sharp case: appearance similarity is high enough to want to fire,
    but the symbolic evidence is class-level only. ReID cannot buy
    distinctiveness, so the system refuses and returns a candidate set."""
    from tests.util import unit_vec
    e = unit_vec(3)
    d = evaluate(make_obs(embedding=e, class_attrs=dict(CAMRY)),
                 make_profile(plate="", class_attrs=dict(CAMRY), gallery=(e,)), graph)
    assert d.score >= 0.45, "high ReID pushed the score over the line"
    assert d.verdict == VERDICT_CANDIDATE
    assert d.refused_to_individuate
    assert any(f.check == "distinctiveness" for f in d.facts)


def test_class_only_no_appearance_is_undecided(graph):
    """Without appearance evidence, class attributes alone don't even reach
    the propose threshold — undecided, not a refusal (distinct outcomes)."""
    d = evaluate(make_obs(class_attrs=dict(CAMRY)),
                 make_profile(plate="", class_attrs=dict(CAMRY)), graph)
    assert d.verdict == "undecided"
    assert not d.refused_to_individuate


def test_floor_is_configurable(graph):
    obs = make_obs(instance_attrs={"accessory": "roof rack"})
    profile = make_profile(plate="", instance_attrs={"accessory": "roof rack"})
    # Default floor 0.30 -> one mark (0.28) refused.
    assert evaluate(obs, profile, graph).refused_to_individuate
    # Lower the floor -> the same evidence now individuates.
    cfg = CascadeConfig(distinctiveness_floor=0.2)
    d = evaluate(obs, profile, graph, cfg)
    assert d.verdict == VERDICT_LIKELY and not d.refused_to_individuate


def test_plate_never_refused_even_at_high_floor(graph):
    cfg = CascadeConfig(distinctiveness_floor=0.99)
    d = evaluate(make_obs(plate="ABC-1234"), make_profile(), graph, cfg)
    assert d.verdict == VERDICT_CONFIRMED, "a plate match is distinctiveness 1.0"
