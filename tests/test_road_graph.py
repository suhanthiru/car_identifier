import math

import pytest

from sim.model import CameraSpec, TransitEdge
from sim.road_graph import RoadGraph, default_world, haversine_m, make_edge


def test_default_world_shape():
    g = default_world()
    assert len(g.cameras) == 8
    # Every listed pair is bidirectional.
    for e in g.edges:
        assert g.edge(e.dst, e.src) is not None


def test_transit_windows_are_ordered():
    g = default_world()
    for e in g.edges:
        assert 0 < e.min_s <= e.typical_s <= e.max_s


def test_window_derivation_matches_speed_envelope():
    a = CameraSpec("a", "A", 40.0, -89.0)
    b = CameraSpec("b", "B", 40.009, -89.0)  # ~1 km north
    e = make_edge(a, b, road_factor=1.0)
    dist = haversine_m(a.lat, a.lon, b.lat, b.lon)
    assert math.isclose(e.distance_m, dist, rel_tol=1e-3)
    assert e.min_s < e.typical_s < e.max_s
    assert e.min_s == pytest.approx(dist / 20.0, rel=1e-2)


def test_neighbors_and_window_lookup():
    g = default_world()
    assert "cam-ctr" in g.neighbors("cam-n")
    assert g.transit_window("cam-n", "cam-ctr") is not None
    # NW and SE-ish corners are not directly adjacent.
    assert g.edge("cam-nw", "cam-e") is None
    assert g.transit_window("cam-nw", "cam-e") is None


def test_min_transit_multi_hop():
    g = default_world()
    direct = g.edge("cam-nw", "cam-n").min_s
    assert g.min_transit_s("cam-nw", "cam-n") == pytest.approx(direct)
    # Two hops minimum: nw -> n -> ne (or nw -> w -> ... ), must exceed one hop.
    two_hop = g.min_transit_s("cam-nw", "cam-ne")
    assert two_hop > direct
    assert g.min_transit_s("cam-ctr", "cam-ctr") == 0.0


def test_graph_rejects_bad_edges():
    a = CameraSpec("a", "A", 40.0, -89.0)
    with pytest.raises(ValueError):
        RoadGraph(
            cameras=(a,),
            edges=(TransitEdge("a", "ghost", 100.0, 5.0, 8.0, 25.0),),
        )
    with pytest.raises(ValueError):
        RoadGraph(
            cameras=(a, CameraSpec("b", "B", 40.01, -89.0)),
            edges=(TransitEdge("a", "b", 100.0, 9.0, 8.0, 25.0),),  # min > typical
        )
