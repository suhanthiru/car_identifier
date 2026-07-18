"""Real-graph construction from CityFlow: real GPS + real observed transit
windows, as opposed to the synthetic world's speed-envelope guess."""
import pytest

from datasets.cityflow import CityFlow


def make_scenario_root(tmp_path):
    """Two cameras ~1.1km apart (real-ish GPS), several observed hops with
    varying elapsed times so min/typical/max differ meaningfully."""
    root = tmp_path / "CityFlow"
    # c001 sees vehicle 1 leave at frame 0; c002 sees it enter at frames
    # giving elapsed times of 20s, 30s, 40s (@10fps) across three vehicles.
    c001_rows = ["0,1,10,10,50,50,1,-1,-1,-1", "0,2,10,10,50,50,1,-1,-1,-1",
                "0,3,10,10,50,50,1,-1,-1,-1", "0,4,10,10,50,50,1,-1,-1,-1"]
    c002_rows = ["200,1,10,10,50,50,1,-1,-1,-1",  # 20.0s
                "300,2,10,10,50,50,1,-1,-1,-1",   # 30.0s
                "400,3,10,10,50,50,1,-1,-1,-1",   # 40.0s
                # vehicle 4 never appears at c002: no transition for it
                ]
    for cam, rows in {"c001": c001_rows, "c002": c002_rows}.items():
        d = root / "train" / "S01" / cam / "gt"
        d.mkdir(parents=True)
        (d / "gt.txt").write_text("\n".join(rows))
    (root / "train" / "S01" / "c001" / "calibration.txt").write_text(
        "Homography matrix: 1 0 -89.62;0 1 40.72;0 0 1")
    (root / "train" / "S01" / "c002" / "calibration.txt").write_text(
        "Homography matrix: 1 0 -89.61;0 1 40.73;0 0 1")
    return root


def test_real_graph_places_cameras_at_real_gps(tmp_path):
    scen = CityFlow(make_scenario_root(tmp_path)).load_scenario("S01")
    graph = scen.to_road_graph()
    gps = scen.camera_gps()
    for cam_id, (lat, lon) in gps.items():
        cam = graph.camera(cam_id)
        assert cam.lat == pytest.approx(lat)
        assert cam.lon == pytest.approx(lon)


def test_real_graph_windows_from_observed_samples(tmp_path):
    scen = CityFlow(make_scenario_root(tmp_path)).load_scenario("S01")
    graph = scen.to_road_graph()
    edge = graph.edge("c001", "c002")
    assert edge is not None
    assert edge.min_s == pytest.approx(20.0)
    assert edge.typical_s == pytest.approx(30.0)  # median of [20, 30, 40]
    assert edge.max_s == pytest.approx(40.0 + 5.0)  # widest + buffer
    assert edge.distance_m > 0  # real haversine distance between real GPS


def test_real_graph_no_edge_without_observed_transitions(tmp_path):
    scen = CityFlow(make_scenario_root(tmp_path)).load_scenario("S01")
    graph = scen.to_road_graph()
    # No vehicle was ever observed going c002 -> c001 in this fixture.
    assert graph.edge("c002", "c001") is None


def test_real_graph_min_transit_matches_observed_min(tmp_path):
    scen = CityFlow(make_scenario_root(tmp_path)).load_scenario("S01")
    graph = scen.to_road_graph()
    assert graph.min_transit_s("c001", "c002") == pytest.approx(20.0)


def test_real_graph_rejects_uncalibrated_scenario(tmp_path):
    from datasets.cityflow import CityFlowScenario, TrackSpan

    scen = CityFlowScenario(
        name="S99", cameras=("c001",), homographies={},
        spans=(TrackSpan("S99", "c001", 1, 0.0, 5.0),))
    with pytest.raises(ValueError, match="no usable camera calibration"):
        scen.to_road_graph()


def test_real_graph_is_a_usable_road_graph(tmp_path):
    """Sanity: the built graph passes the same validation and supports the
    same queries a live server/tracker would make on it."""
    scen = CityFlow(make_scenario_root(tmp_path)).load_scenario("S01")
    graph = scen.to_road_graph()
    assert set(graph.camera_ids()) == {"c001", "c002"}
    assert graph.neighbors("c001") == ("c002",)
    assert graph.transit_window("c001", "c002") == (20.0, 45.0)
