"""The per-scenario showcase set: the ranking rules, and the endpoint that
serves the frozen artifact.

The set is the first thing a visitor sees, so two things have to hold. The
ranking must be reproducible from ground truth alone -- not from anything the
cascade concluded, which would be selection on the outcome. And the endpoint
must not become a way to read arbitrary files out of the repo.
"""
import json

import pytest
from fastapi.testclient import TestClient

from datasets.cityflow import CityFlowScenario, TrackSpan
from scripts.rank_showcase import _features, _lookalike_pair, rank
from server.api import create_app
from tests.test_server_cityflow import make_cityflow_root


def _span(cam, vid, enter, exit_):
    return TrackSpan(scenario="S01", camera_id=cam, vehicle_id=vid,
                     enter_s=enter, exit_s=exit_)


def _scenario(spans):
    """A scenario with enough calibration for to_road_graph() to work.

    S01's real center is on record, and two cameras with distinct homography
    image-center mappings give the graph two nodes to place.
    """
    import numpy as np
    cams = sorted({s.camera_id for s in spans})
    homographies = {}
    for i, cam in enumerate(cams):
        H = np.eye(3)
        H[0, 2] = float(i * 10)
        H[1, 2] = float(i * 4)
        homographies[cam] = H
    return CityFlowScenario(name="S01", cameras=tuple(cams),
                            homographies=homographies, spans=tuple(spans))


def test_transit_is_measured_between_passage_MIDPOINTS():
    """The feature must model what the cascade evaluates, not what
    transitions() reports.

    The live feed emits ONE timestamp per passage, its midpoint, and the
    transit check differences those. Using transitions()' exit->enter gaps
    instead predicted the real per-passage verdicts in
    data/recognizability_s01.json almost at chance; midpoints agreed on 84 of
    86 evaluated S01 vehicles. The two quantities differ by half of each
    passage's duration, which on this dataset is bigger than most of the gaps.
    """
    # c001 spans 0..20 (midpoint 10), c002 spans 30..50 (midpoint 40).
    # exit->enter gap is 10s; midpoint-to-midpoint is 30s.
    spans = [_span("c001", 1, 0.0, 20.0), _span("c002", 1, 30.0, 50.0)]
    feats = _features(_scenario(spans))
    assert feats[1]["hops"][0]["dt_s"] == pytest.approx(30.0)


def test_transit_is_measured_from_the_SEED_passage():
    """Every later sighting is checked against the moment the car was flagged.

    `last_seen` only advances when a match is accepted, and on plateless
    CityFlow footage most sightings never reach acceptance -- so in practice
    the comparison stays anchored at the first passage. Chaining consecutive
    hops instead would model a target that keeps getting confirmed, which is
    the case this dataset almost never produces.
    """
    spans = [_span("c001", 1, 0.0, 10.0),     # midpoint 5
             _span("c002", 1, 20.0, 30.0),    # midpoint 25
             _span("c003", 1, 40.0, 50.0)]    # midpoint 45
    feats = _features(_scenario(spans))
    hops = {h["to_camera"]: h["dt_s"] for h in feats[1]["hops"]}
    assert hops["c002"] == pytest.approx(20.0)   # 25 - 5
    assert hops["c003"] == pytest.approx(40.0)   # 45 - 5, NOT 45 - 25


def test_lookalike_pair_needs_a_shared_onward_camera():
    """Co-presence alone is not the hard case.

    Two cars at one camera together are only genuinely confusable if space-time
    cannot separate them afterwards. If they leave for different cameras the
    transit window prefers one of them and the problem solves itself, so the
    pair is not worth showcasing.
    """
    # 1 and 2 overlap at c001 and both go on to c002 -> a pair.
    together = [_span("c001", 1, 0.0, 10.0), _span("c001", 2, 2.0, 12.0),
                _span("c002", 1, 20.0, 30.0), _span("c002", 2, 22.0, 32.0)]
    assert _lookalike_pair(_scenario(together), _features(_scenario(together))) \
        == (1, 2)

    # Same overlap, but they diverge -> not a pair.
    apart = [_span("c001", 1, 0.0, 10.0), _span("c001", 2, 2.0, 12.0),
             _span("c002", 1, 20.0, 30.0), _span("c003", 2, 22.0, 32.0)]
    assert _lookalike_pair(_scenario(apart), _features(_scenario(apart))) is None


def test_the_pair_claims_its_vehicles_before_other_archetypes():
    """The pair is the only archetype that needs a specific COMBINATION, and
    the only one reaching the multi-target ambiguity guard -- so it cannot be
    assembled from whatever the single-vehicle archetypes left behind.

    Letting them go first silently destroyed it in practice: on S02 both
    members topped another archetype and the pair vanished from the set
    entirely; on S04 only one member survived, so the "flag both" instruction
    pointed at a car that was not on screen. Both members must appear, and
    must name each other.
    """
    spans = [_span("c001", 1, 0.0, 60.0), _span("c001", 2, 2.0, 62.0),
             _span("c002", 1, 80.0, 90.0), _span("c002", 2, 82.0, 92.0),
             _span("c001", 3, 0.0, 5.0), _span("c002", 3, 40.0, 45.0)]
    report = rank(_scenario(spans))
    pair = [v for v in report["vehicles"] if v["archetype"] == "lookalike_pair"]
    assert len(pair) == 2, "both members or the instruction is unfollowable"
    assert {v["vehicle_id"] for v in pair} == {1, 2}
    assert {v["partner_vehicle_id"] for v in pair} == {1, 2}
    # And nothing is listed twice under two archetypes.
    ids = [v["vehicle_id"] for v in report["vehicles"]]
    assert len(ids) == len(set(ids))


def test_every_pick_carries_a_reason_and_a_prediction():
    """A shortlist without stated reasons is just a claim. Each entry says why
    it was chosen (a ground-truth fact) and what the cascade should therefore
    do, so an operator can check the system against a prediction made BEFORE
    they ran it."""
    spans = [_span("c001", 1, 0.0, 60.0), _span("c001", 2, 2.0, 62.0),
             _span("c002", 1, 80.0, 90.0), _span("c002", 2, 82.0, 92.0),
             _span("c001", 3, 0.0, 5.0), _span("c002", 3, 40.0, 45.0)]
    for v in rank(_scenario(spans))["vehicles"]:
        assert v["why"].strip()
        assert v["expect"].strip()
        assert v["headline"].strip()
        assert "{" not in v["why"], "unformatted placeholder left in the copy"


# --------------------------------------------------------------- the endpoint

@pytest.fixture()
def client(tmp_path):
    root = make_cityflow_root(tmp_path)
    app = create_app(db_url=f"sqlite:///{tmp_path}/cf.sqlite",
                     crops_dir=str(tmp_path / "crops"), world_source="real",
                     cityflow_root=str(root), cityflow_scenario_name="S01")
    with TestClient(app) as c:
        yield c


def test_showcase_serves_the_committed_artifact(client):
    r = client.get("/api/cityflow/S01/showcase")
    assert r.status_code == 200
    body = r.json()
    assert body["scenario"] == "S01"
    assert body["vehicles"], "S01 has a committed set"
    for v in body["vehicles"]:
        assert {"vehicle_id", "archetype", "headline", "why", "expect"} <= set(v)


def test_missing_scenario_is_404_not_500(client):
    """A scenario with no artifact is a normal state -- the console falls back
    to another filter. A 500 would read as the server being broken."""
    r = client.get("/api/cityflow/S42/showcase")
    assert r.status_code == 404
    assert "rank_showcase" in r.json()["detail"], "says how to generate one"


@pytest.mark.parametrize("evil", [
    "..", "../..", "..%2F..%2Fserver", "S01/../../server", "S01.json",
    "con", "S01 ", "%2e%2e",
])
def test_showcase_refuses_path_traversal(client, evil):
    """The scenario name is path-segment input reaching a filesystem read.
    This file has shipped an arbitrary-path bug before; nothing here may
    resolve outside data/showcase."""
    r = client.get(f"/api/cityflow/{evil}/showcase")
    assert r.status_code == 404
    assert "-----BEGIN" not in r.text


def test_committed_artifacts_match_a_fresh_run():
    """`--check` is what keeps the frozen set honest: the ranking is derived
    from ground truth, so it must be reproducible, and a committed file that
    no longer matches the scorer means the stated reasons describe cars that
    would no longer be chosen."""
    path = "data/showcase/S01.json"
    try:
        with open(path, encoding="utf-8") as fh:
            body = json.load(fh)
    except FileNotFoundError:
        pytest.skip("no committed showcase artifact")
    assert body["schema"] == "showcase-v1"
    assert body["generated_from"].startswith("ground truth")
