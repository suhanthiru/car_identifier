"""Server-side 3D bridge tests: gated fusion on confirmed sightings only,
model3d endpoints, and path traversal guarding."""
import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("cargen")
cv2 = pytest.importorskip("cv2")

from server.api import create_app
from sim.emitter import build_default_world
from sim.render import render_vehicle_crop
from tests.util import CAMRY, unit_vec

PLATE = "ABC-1234"


@pytest.fixture()
def client(tmp_path):
    app = create_app(db_url=f"sqlite:///{tmp_path}/t.sqlite",
                     crops_dir=str(tmp_path / "crops"),
                     targets3d_dir=str(tmp_path / "t3d"),
                     enable_3d=True)
    with TestClient(app) as c:
        yield c


def crop_b64():
    world = build_default_world()
    crop = render_vehicle_crop(world.fleet[0], "cam-ctr", 5.0)
    ok, png = cv2.imencode(".png", crop)
    assert ok
    return base64.b64encode(png.tobytes()).decode()


def sighting(event_id, plate=PLATE, with_crop=True):
    body = {
        "event_id": event_id, "camera_id": "cam-ctr", "timestamp_s": 1000.0,
        "lat": 40.73, "lon": -89.61,
        "embedding": [float(x) for x in unit_vec(3, 16)],
        "class_attrs": dict(CAMRY),
        "crop_png_b64": crop_b64() if with_crop else "",
    }
    if plate:
        body["plate"] = {"text": plate, "confidence": 0.95, "source": "sim"}
    return body


def test_confirmed_sighting_builds_3d_model(client):
    target_id = client.post("/api/targets", json={
        "label": "t", "plate": PLATE, "class_attrs": CAMRY}).json()["target_id"]
    assert client.get(f"/api/targets/{target_id}/model3d").json()["exists"] is False

    resp = client.post("/api/sightings", json=sighting("evt-1"))
    assert "profile_update" in resp.json()["events"]

    status = client.get(f"/api/targets/{target_id}/model3d").json()
    assert status["exists"] is True
    assert status["observations"] == 1
    assert status["n_splats"] > 0
    png = client.get(status["turntable"])
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    splat = client.get(status["exports"]["splat"])
    assert splat.status_code == 200 and len(splat.content) > 0


def test_unconfirmed_sighting_never_fuses(client):
    """The anti-poisoning parallel: appearance-grade sightings must no more
    touch the 3D model than they may touch the profile."""
    target_id = client.post("/api/targets", json={
        "label": "t", "plate": "", "class_attrs": CAMRY,
        "instance_attrs": {"accessory": "roof rack"}}).json()["target_id"]
    body = sighting("evt-9", plate=None)
    body["instance_attrs"] = {"accessory": "roof rack"}
    resp = client.post("/api/sightings", json=body)
    assert "review" in resp.json()["events"]
    assert client.get(f"/api/targets/{target_id}/model3d").json()["exists"] is False


def test_operator_accept_fuses(client):
    target_id = client.post("/api/targets", json={
        "label": "t", "plate": "", "class_attrs": CAMRY,
        "instance_attrs": {"accessory": "roof rack"}}).json()["target_id"]
    body = sighting("evt-5", plate=None)
    body["instance_attrs"] = {"accessory": "roof rack"}
    client.post("/api/sightings", json=body)
    review = client.get("/api/reviews").json()[0]
    client.post(f"/api/reviews/{review['review_id']}/resolve", json={"accept": True})
    status = client.get(f"/api/targets/{target_id}/model3d").json()
    assert status["exists"] is True, "operator confirmation opens the 3D gate too"


def test_pipeline_is_built_once_across_fusions(client, monkeypatch):
    """The integration blocker: a Target3DModel built per event with no
    injected pipeline reconstructs the whole cargen stack every fusion. With
    a real prior that is ~4 GB of weights and most of an 8 GB GPU, per crop."""
    import car3d.profile_model as pm

    builds = []
    real_build = pm.build_pipeline

    def counting_build(*args, **kwargs):
        builds.append(1)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(pm, "build_pipeline", counting_build)

    target_id = client.post("/api/targets", json={
        "label": "t", "plate": PLATE, "class_attrs": CAMRY}).json()["target_id"]
    for i in range(3):
        client.post("/api/sightings", json=sighting(f"evt-multi-{i}"))
        # the status endpoint settles the queued fusion before reporting
        assert client.get(f"/api/targets/{target_id}/model3d").json()["exists"]

    assert sum(builds) == 1, f"pipeline rebuilt {sum(builds)}x across 3 fusions"


def test_fusion_does_not_block_the_ingest_response(client):
    """Reconstruction is queued, not run inline: /api/sightings must return
    without waiting for it. Correctness of the queued work is covered by the
    tests above, which settle it via the status endpoint."""
    client.post("/api/targets", json={
        "label": "t", "plate": PLATE, "class_attrs": CAMRY})
    resp = client.post("/api/sightings", json=sighting("evt-async"))
    assert resp.status_code == 202
    assert "profile_update" in resp.json()["events"]


def test_render_verifier_stays_out_of_the_cascade_by_default(tmp_path):
    """Showing a reconstruction and letting it vote on identity are different
    claims. The measured ablation (scripts/ablate_3d_cityflow.py) found the
    geometry channel vetoing correct matches and no incorrect ones, so the 3D
    panel is on with `enable_3d` while its identification path needs the
    separate opt-in."""
    app = create_app(db_url=f"sqlite:///{tmp_path}/w.sqlite",
                     crops_dir=str(tmp_path / "crops"),
                     targets3d_dir=str(tmp_path / "t3d"), enable_3d=True)
    with TestClient(app):
        assert app.state.enable_3d is True
        assert app.state.tracker._cascade_config.shortlist_verifier is None


def test_render_verifier_is_wired_when_3d_identification_enabled(tmp_path):
    """reasoning/ declares the shortlist_verifier hook but never imports
    car3d; the server is the only place the two can be joined. Before this
    wiring the verifier was built, tested, and unreachable in production."""
    app = create_app(db_url=f"sqlite:///{tmp_path}/w.sqlite",
                     crops_dir=str(tmp_path / "crops"),
                     targets3d_dir=str(tmp_path / "t3d"), enable_3d=True,
                     enable_3d_identification=True)
    with TestClient(app):
        verifier = app.state.tracker._cascade_config.shortlist_verifier
        assert verifier is not None

        # Abstains rather than guessing when there is no crop to compare.
        class NoCrop:
            crop = None

        assert verifier("tgt-1", NoCrop()) is None
        # ...and when the target has no model yet, without raising.
        class WithCrop:
            crop = np.zeros((48, 64, 3), np.uint8)

        assert verifier("tgt-missing", WithCrop()) is None


def test_no_render_verifier_when_3d_disabled(tmp_path):
    app = create_app(db_url=f"sqlite:///{tmp_path}/n.sqlite",
                     crops_dir=str(tmp_path / "crops"),
                     targets3d_dir=str(tmp_path / "t3d"), enable_3d=False)
    with TestClient(app):
        cfg = app.state.tracker._cascade_config
        assert cfg is None or cfg.shortlist_verifier is None


def test_model3d_traversal_blocked(client):
    target_id = client.post("/api/targets", json={
        "label": "t", "plate": PLATE, "class_attrs": CAMRY}).json()["target_id"]
    client.post("/api/sightings", json=sighting("evt-1"))
    bad = client.get(f"/api/targets/{target_id}/model3d/..%2F..%2Fcloud.npz")
    assert bad.status_code == 404
