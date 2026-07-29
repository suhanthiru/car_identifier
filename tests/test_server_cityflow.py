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



def _vehicles(client, path="/api/cityflow/S01/vehicles", tries=60):
    """Fetch the browse list, waiting out the background index build.

    The endpoint answers 503 while building rather than blocking a threadpool
    worker for the length of a video decode — see the route for why building
    inline let a polling client take the whole API down. Tests therefore poll
    exactly like the console does.
    """
    import time

    for _ in range(tries):
        resp = client.get(path)
        if resp.status_code == 200:
            return resp.json()
        assert resp.status_code == 503, resp.status_code
        time.sleep(0.1)
    raise AssertionError("vehicle index never finished building")


def client_gallery(client, vehicle_id, scenario="S01"):
    resp = client.get(f"/api/cityflow/{scenario}/vehicles/{vehicle_id}/gallery")
    assert resp.status_code == 200, resp.status_code
    return resp.json()["gallery_b64"]


def test_vehicles_returns_real_shape_with_thumbnail(cityflow_client):
    vehicles = _vehicles(cityflow_client)
    assert len(vehicles) == 1
    v = vehicles[0]
    # gallery_b64 is deliberately ABSENT: three full-resolution crops per
    # vehicle made this response 31 MB for 95 vehicles, to draw thumbnails.
    # It is fetched per-vehicle at flag time instead.
    assert set(v.keys()) == {"vehicle_id", "first_camera", "first_time_s",
                             "thumbnail_b64",
                             "n_cameras", "n_passages", "cameras",
                             "visible_s", "span_s", "last_time_s", "journeys"}
    # Watchability metrics the browse panel filters and sorts on.
    assert v["visible_s"] >= 0 and v["span_s"] >= 0
    assert v["last_time_s"] >= v["first_time_s"]
    assert v["vehicle_id"] == 7
    assert v["first_camera"] == "c001"
    assert v["thumbnail_b64"]
    # Reference-gallery seeds come from the per-vehicle endpoint: >=1 real
    # passage crop, with the thumbnail among them.
    g = client_gallery(cityflow_client, v["vehicle_id"])
    assert g and v["thumbnail_b64"] in g
    assert v["n_cameras"] == len(v["cameras"]) >= 1


def test_vehicles_can_exclude_single_camera_vehicles(cityflow_client):
    """`min_cameras=2` hides vehicles that cannot be re-identified.

    A vehicle the ground truth only ever saw once has no second sighting to
    associate, so flagging it leaves the review queue empty and makes a working
    run look broken. The fixture's lone vehicle is single-camera, so the
    filtered list must come back empty rather than fall back to everything.
    """
    unfiltered = _vehicles(cityflow_client)
    filtered = _vehicles(cityflow_client,
                         "/api/cityflow/S01/vehicles?min_cameras=2")
    assert len(unfiltered) == 1 and unfiltered[0]["n_cameras"] == 1
    assert filtered == []
    # min_cameras=1 is the unfiltered default, not a special case.
    assert _vehicles(
        cityflow_client,
        "/api/cityflow/S01/vehicles?min_cameras=1") == unfiltered


def test_vehicles_404_for_a_different_scenario_name(cityflow_client):
    resp = cityflow_client.get("/api/cityflow/S02/vehicles")
    assert resp.status_code == 404


def test_vehicles_cached_after_first_build(cityflow_client, monkeypatch):
    first = _vehicles(cityflow_client)
    import server.real_feed as real_feed_module

    def _boom(*a, **kw):
        raise AssertionError("build_vehicle_index should not run twice")
    monkeypatch.setattr(real_feed_module, "build_vehicle_index", _boom)
    second = _vehicles(cityflow_client)
    assert first == second


def test_concurrent_requests_start_only_one_index_build(cityflow_client,
                                                        monkeypatch):
    """Polling while the index builds must not start a second build.

    Regression: the endpoint used to build inline. It is a sync route, so
    FastAPI ran it in the threadpool, and a cold build takes tens of seconds of
    video decoding. A client polling for the list therefore started a fresh
    concurrent build on every poll until the threadpool and the CPU were both
    exhausted and the whole API stopped answering — the console "froze" on any
    scenario switch whose index was not already cached.
    """
    import server.real_feed as real_feed_module

    builds = []
    real_build = real_feed_module.build_vehicle_index

    def counting_build(scenario, camera_dirs, *a, **kw):
        builds.append(1)
        return real_build(scenario, camera_dirs, *a, **kw)

    monkeypatch.setattr(real_feed_module, "build_vehicle_index", counting_build)

    # Hammer it the way the console did, then let the single build finish.
    for _ in range(8):
        cityflow_client.get("/api/cityflow/S01/vehicles")
    _vehicles(cityflow_client)
    assert sum(builds) == 1, f"started {sum(builds)} concurrent index builds"


def test_journeys_come_from_ground_truth_and_need_a_real_gap(cityflow_client):
    """`journeys` selects on the FOOTAGE, never on anything we computed.

    A hop only counts when the vehicle left one camera and arrived at another
    LATER — a positive gap, an interval where nothing observed it. That gap is
    the re-identification problem. Vehicles simultaneously in several views
    (54 of S01's 95) pose no such question, which is why flagging one can sit
    there producing nothing.

    Deliberately not derived from cascade output: selecting on what the system
    concluded would be selection on the outcome, and the console would be
    answering a question it had already rigged.
    """
    v = _vehicles(cityflow_client)[0]
    assert isinstance(v["journeys"], list)
    # The fixture's vehicle is seen at one camera only, so it cannot have one.
    assert v["n_cameras"] == 1 and v["journeys"] == []
    for hop in v["journeys"]:
        assert hop["gap_s"] > 0
        assert hop["from_camera"] != hop["to_camera"]
