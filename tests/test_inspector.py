"""Reasoning sandbox endpoint: runs the real cascade on hand-built inputs
with no DB/tracker/audit side effects."""
import pytest
from fastapi.testclient import TestClient

from server.api import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(db_url=f"sqlite:///{tmp_path}/test.sqlite",
                     crops_dir=str(tmp_path / "crops"))
    with TestClient(app) as c:
        yield c


def base_target(**kw):
    t = {"target_id": "t1", "label": "Test target", "plate": "ABC-1234",
         "class_attrs": {"make": "Toyota", "model": "Camry",
                         "body_type": "sedan", "color": "silver"}}
    t.update(kw)
    return t


def base_sighting(**kw):
    s = {"camera_id": "cam-ctr", "timestamp_s": 1000.0, "plate_text": "ABC-1234",
         "plate_confidence": 0.95,
         "class_attrs": {"make": "Toyota", "model": "Camry",
                         "body_type": "sedan", "color": "silver"}}
    s.update(kw)
    return s


def test_plate_match_confirms(client):
    resp = client.post("/api/inspect/evaluate", json={
        "sighting": base_sighting(), "targets": [base_target()]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["best"]["verdict"] == "confirmed"
    assert body["best"]["distinctiveness"] == 1.0
    assert not body["best"]["refused_to_individuate"]
    assert any(f["check"] == "plate" for f in body["best"]["facts"])
    assert body["best"]["signals"]["plate_exact"] is True


def test_impossible_transit_produces_counterfactual(client):
    target = base_target(plate="ABC-1234", last_seen_camera_id="cam-nw",
                         last_seen_timestamp_s=1000.0)
    sighting = base_sighting(camera_id="cam-e", timestamp_s=1005.0)
    resp = client.post("/api/inspect/evaluate", json={
        "sighting": sighting, "targets": [target]})
    body = resp.json()["best"]
    assert body["verdict"] == "rejected"
    assert body["anomaly"] is True
    cf_signals = {c["signal"] for c in body["counterfactuals"]}
    assert "transit" in cf_signals
    transit_cf = next(c for c in body["counterfactuals"] if c["signal"] == "transit")
    assert transit_cf["boundary"]


def test_distinctiveness_floor_refuses_generic_lookalike(client):
    marks = {}
    generic = base_target(plate="", instance_attrs=marks, reid_similarity=0.99)
    sighting = base_sighting(plate_text="", instance_attrs=marks)
    resp = client.post("/api/inspect/evaluate", json={
        "sighting": sighting, "targets": [generic]})
    body = resp.json()["best"]
    assert body["verdict"] == "candidate"
    assert body["refused_to_individuate"] is True
    assert "t1" in body["candidate_ids"]


def test_configurable_floor_changes_outcome(client):
    target = base_target(plate="", instance_attrs={"accessory": "roof rack"},
                         reid_similarity=None)
    sighting = base_sighting(plate_text="", instance_attrs={"accessory": "roof rack"})
    default_resp = client.post("/api/inspect/evaluate", json={
        "sighting": sighting, "targets": [target]}).json()
    lowered_resp = client.post("/api/inspect/evaluate", json={
        "sighting": sighting, "targets": [target], "distinctiveness_floor": 0.2}).json()
    assert default_resp["best"]["refused_to_individuate"] is True
    assert lowered_resp["best"]["refused_to_individuate"] is False
    assert lowered_resp["best"]["verdict"] == "likely"


def test_multiple_targets_ranked(client):
    a = base_target(target_id="a", label="A", plate="", reid_similarity=0.9,
                    instance_attrs={"accessory": "roof rack"})
    b = base_target(target_id="b", label="B", plate="", reid_similarity=0.2,
                    instance_attrs={"accessory": "roof rack"})
    sighting = base_sighting(plate_text="", instance_attrs={"accessory": "roof rack"})
    resp = client.post("/api/inspect/evaluate", json={
        "sighting": sighting, "targets": [a, b]}).json()
    assert resp["best"]["target_id"] == "a"
    assert len(resp["all_decisions"]) == 2
    assert resp["all_decisions"][0]["score"] >= resp["all_decisions"][1]["score"]


def test_unknown_camera_rejected(client):
    resp = client.post("/api/inspect/evaluate", json={
        "sighting": base_sighting(camera_id="cam-nowhere"),
        "targets": [base_target()]})
    assert resp.status_code == 422


def test_render_returns_png(client):
    import json
    import urllib.parse

    payload = urllib.parse.quote(json.dumps({
        "plate": "ABC-1234", "make": "Toyota", "model": "Camry",
        "body_type": "sedan", "color": "silver", "instance_attrs": {}}))
    resp = client.get(f"/api/inspect/render?camera_id=cam-ctr&timestamp_s=1000&payload={payload}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_unknown_color_falls_back(client):
    import json
    import urllib.parse

    payload = urllib.parse.quote(json.dumps({"color": "chartreuse"}))
    resp = client.get(f"/api/inspect/render?camera_id=cam-ctr&timestamp_s=1000&payload={payload}")
    assert resp.status_code == 200  # falls back to gray rather than 500ing


def test_render_unknown_camera_rejected(client):
    resp = client.get("/api/inspect/render?camera_id=nope&timestamp_s=1000&payload=%7B%7D")
    assert resp.status_code == 422


def test_render_bad_payload_rejected(client):
    resp = client.get("/api/inspect/render?camera_id=cam-ctr&timestamp_s=1000&payload=not-json")
    assert resp.status_code == 422


def test_no_side_effects(client):
    """The sandbox must never create a real target or audit entry."""
    client.post("/api/inspect/evaluate", json={
        "sighting": base_sighting(), "targets": [base_target()]})
    assert client.get("/api/targets").json() == {}
    audit = client.get("/api/audit").json()
    assert audit["length"] == 0
