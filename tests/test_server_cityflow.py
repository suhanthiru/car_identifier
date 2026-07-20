"""GET /api/cityflow/scenarios + /api/cityflow/{scenario}/vehicles:
presence-gated exactly like every other real-dataset endpoint."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from server.api import create_app


def make_cityflow_root(tmp_path):
    root = tmp_path / "CityFlow"
    cam_dir = root / "train" / "S01" / "c001"
    (cam_dir / "gt").mkdir(parents=True)
    (cam_dir / "gt" / "gt.txt").write_text(
        "0,7,2,2,10,8,1,-1,-1,-1\n10,7,3,3,10,8,1,-1,-1,-1\n")
    (cam_dir / "calibration.txt").write_text(
        "Homography matrix: 1 0 -90.72;0 1 42.52;0 0 1")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(cam_dir / "vdo.avi"), fourcc, 10.0, (32, 24))
    for i in range(11):
        writer.write(np.full((24, 32, 3), i * 20, dtype=np.uint8))
    writer.release()
    return root


@pytest.fixture()
def bare_client(tmp_path):
    app = create_app(db_url=f"sqlite:///{tmp_path}/bare.sqlite",
                     crops_dir=str(tmp_path / "crops"))
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def cityflow_client(tmp_path):
    root = make_cityflow_root(tmp_path)
    app = create_app(db_url=f"sqlite:///{tmp_path}/cf.sqlite",
                     crops_dir=str(tmp_path / "crops"), world_source="real",
                     cityflow_root=str(root), cityflow_scenario_name="S01")
    with TestClient(app) as c:
        yield c


def test_scenarios_empty_when_no_cityflow_configured(bare_client):
    assert bare_client.get("/api/cityflow/scenarios").json() == []


def test_vehicles_404_when_no_cityflow_configured(bare_client):
    resp = bare_client.get("/api/cityflow/S01/vehicles")
    assert resp.status_code == 404


def test_scenarios_lists_the_active_scenario(cityflow_client):
    assert cityflow_client.get("/api/cityflow/scenarios").json() == ["S01"]


def test_vehicles_returns_real_shape_with_thumbnail(cityflow_client):
    resp = cityflow_client.get("/api/cityflow/S01/vehicles")
    assert resp.status_code == 200
    vehicles = resp.json()
    assert len(vehicles) == 1
    v = vehicles[0]
    assert set(v.keys()) == {"vehicle_id", "first_camera", "first_time_s", "thumbnail_b64"}
    assert v["vehicle_id"] == 7
    assert v["first_camera"] == "c001"
    assert v["thumbnail_b64"]


def test_vehicles_404_for_a_different_scenario_name(cityflow_client):
    resp = cityflow_client.get("/api/cityflow/S02/vehicles")
    assert resp.status_code == 404


def test_vehicles_cached_after_first_build(cityflow_client, monkeypatch):
    first = cityflow_client.get("/api/cityflow/S01/vehicles").json()
    import server.real_feed as real_feed_module

    def _boom(*a, **kw):
        raise AssertionError("build_vehicle_index should not run twice")
    monkeypatch.setattr(real_feed_module, "build_vehicle_index", _boom)
    second = cityflow_client.get("/api/cityflow/S01/vehicles").json()
    assert first == second
