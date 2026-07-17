"""API integration tests over a temp SQLite DB (no ML models involved —
observations are hand-built and posted like an edge node would)."""
import base64

import pytest
from fastapi.testclient import TestClient

from server.api import create_app
from tests.util import unit_vec

PLATE = "ABC-1234"
CAMRY = {"make": "Toyota", "model": "Camry", "body_type": "sedan", "color": "silver"}

# 1x1 white PNG.
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFBQIA"
    "X8jx0gAAAABJRU5ErkJggg==")


@pytest.fixture()
def client(tmp_path):
    app = create_app(db_url=f"sqlite:///{tmp_path}/test.sqlite",
                     crops_dir=str(tmp_path / "crops"))
    with TestClient(app) as c:
        yield c


def flag(client, **kw) -> str:
    body = {"label": "test flag", "plate": PLATE, "class_attrs": CAMRY}
    body.update(kw)
    resp = client.post("/api/targets", json=body)
    assert resp.status_code == 201
    return resp.json()["target_id"]


def sighting(event_id="evt-1", camera_id="cam-ctr", t=1000.0, plate=PLATE,
             seed=1, crop=False, **kw):
    body = {
        "event_id": event_id, "camera_id": camera_id, "timestamp_s": t,
        "lat": 40.73, "lon": -89.61,
        "embedding": [float(x) for x in unit_vec(seed, 16)],
        "class_attrs": dict(CAMRY), "instance_attrs": {},
        "crop_png_b64": base64.b64encode(TINY_PNG).decode() if crop else "",
    }
    if plate:
        body["plate"] = {"text": plate, "confidence": 0.95, "source": "sim"}
    body.update(kw)
    return body


def test_world_endpoints(client):
    cameras = client.get("/api/cameras").json()
    assert len(cameras) == 8
    adjacency = client.get("/api/adjacency").json()
    assert all(e["min_s"] < e["max_s"] for e in adjacency)


def test_flag_and_confirm_flow(client):
    target_id = flag(client)
    resp = client.post("/api/sightings", json=sighting(crop=True))
    assert resp.status_code == 202
    kinds = resp.json()["events"]
    assert "association" in kinds and "profile_update" in kinds

    snap = client.get("/api/targets").json()
    assert snap[target_id]["state"] == "confirmed"
    assert snap[target_id]["belief"] > 0.9

    alerts = client.get("/api/alerts").json()
    assert any(a["kind"] == "association" for a in alerts)

    dossier = client.get(f"/api/targets/{target_id}").json()
    assert dossier["gallery_size"] == 1
    assert dossier["profile_updates"]
    assert dossier["corroboration_chain"]
    facts = dossier["corroboration_chain"][0]["facts"]
    assert "exactly matches" in facts, "plain-english facts must be persisted"
    assert dossier["reference_crop"] == "evt-1.png"


def test_review_flow_accept(client):
    target_id = flag(client, plate="", instance_attrs={"accessory": "roof rack"})
    resp = client.post("/api/sightings", json=sighting(
        plate=None, instance_attrs={"accessory": "roof rack"}))
    assert "review" in resp.json()["events"]
    reviews = client.get("/api/reviews").json()
    assert len(reviews) == 1
    assert "[+]" in reviews[0]["facts"]

    resolve = client.post(f"/api/reviews/{reviews[0]['review_id']}/resolve",
                          json={"accept": True})
    assert resolve.status_code == 200
    assert client.get("/api/reviews").json() == []
    dossier = client.get(f"/api/targets/{target_id}").json()
    assert dossier["live"]["state"] == "confirmed"
    assert dossier["live"]["profile_version"] == 1


def test_review_flow_reject_leaves_profile(client):
    target_id = flag(client, plate="", instance_attrs={"accessory": "roof rack"})
    client.post("/api/sightings", json=sighting(
        plate=None, instance_attrs={"accessory": "roof rack"}))
    review_id = client.get("/api/reviews").json()[0]["review_id"]
    client.post(f"/api/reviews/{review_id}/resolve", json={"accept": False})
    dossier = client.get(f"/api/targets/{target_id}").json()
    assert dossier["live"]["profile_version"] == 0
    rows = client.get("/api/reviews", params={"status": "rejected"}).json()
    assert len(rows) == 1
    # Resolving twice is a 404, not a double-apply.
    assert client.post(f"/api/reviews/{review_id}/resolve",
                       json={"accept": True}).status_code == 404


def test_anomaly_on_impossible_transit(client):
    flag(client)
    client.post("/api/sightings", json=sighting(event_id="evt-1", camera_id="cam-nw",
                                                t=1000.0))
    resp = client.post("/api/sightings", json=sighting(event_id="evt-2",
                                                       camera_id="cam-e", t=1005.0,
                                                       seed=2))
    assert "anomaly" in resp.json()["events"]
    reviews = client.get("/api/reviews").json()
    anomaly = [r for r in reviews if r["kind"] == "anomaly"]
    assert anomaly and "impossible" in anomaly[0]["facts"].lower()


def test_crop_roundtrip_and_traversal_block(client):
    flag(client)
    client.post("/api/sightings", json=sighting(crop=True))
    got = client.get("/api/crops/evt-1.png")
    assert got.status_code == 200
    assert got.content == TINY_PNG
    assert client.get("/api/crops/../eyes.sqlite").status_code in (404, 403)


def test_profile_edit_is_audited(client):
    target_id = flag(client)
    resp = client.patch(f"/api/targets/{target_id}", json={"label": "renamed"})
    assert resp.json()["version"] == 1
    dossier = client.get(f"/api/targets/{target_id}").json()
    assert dossier["label"] == "renamed"
    assert any("Operator edited" in u["reason"] for u in dossier["profile_updates"])
    assert client.patch(f"/api/targets/{target_id}", json={}).status_code == 422


def test_invalid_payloads(client):
    flag(client)
    bad = sighting()
    bad["embedding"] = [0.0] * 16
    assert client.post("/api/sightings", json=bad).status_code == 422
    bad2 = sighting(crop=True)
    bad2["crop_png_b64"] = "not-base64!!"
    assert client.post("/api/sightings", json=bad2).status_code == 422
    assert client.get("/api/targets/tgt-999").status_code == 404


def test_websocket_stream(client):
    flag(client)
    with client.websocket_connect("/ws/console") as ws:
        first = ws.receive_json()
        assert first["type"] == "snapshot"
        client.post("/api/sightings", json=sighting())
        types = [ws.receive_json()["type"] for _ in range(3)]
        assert "contact" in types
        assert "association" in types
