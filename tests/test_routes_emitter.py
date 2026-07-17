from sim.emitter import build_default_world, iter_sightings
from sim.fleet import generate_fleet
from sim.road_graph import default_world
from sim.routes import RouteConfig, generate_routes


def test_every_route_hop_is_physically_plausible():
    """The simulator must never violate its own transit windows —
    otherwise the reasoning layer's veto would (correctly) reject truth."""
    world = build_default_world()
    for route in world.routes:
        for a, b in zip(route.stops, route.stops[1:]):
            edge = world.graph.edge(a.camera_id, b.camera_id)
            assert edge is not None, "routes must follow graph edges"
            dt = b.arrival_s - a.arrival_s
            assert edge.contains(dt), (
                f"{route.vehicle_id}: {a.camera_id}->{b.camera_id} took {dt}s, "
                f"window is [{edge.min_s}, {edge.max_s}]"
            )


def test_routes_cover_all_vehicles():
    world = build_default_world()
    assert {r.vehicle_id for r in world.routes} == {v.vehicle_id for v in world.fleet}


def test_lookalikes_overlap_in_time():
    """Cluster members must be on the road concurrently, or there is
    nothing for cross-camera reasoning to disambiguate."""
    world = build_default_world()
    by_vehicle = {r.vehicle_id: r for r in world.routes}
    clusters: dict[str, list[str]] = {}
    for v in world.fleet:
        if v.lookalike_group:
            clusters.setdefault(v.lookalike_group, []).append(v.vehicle_id)
    for members in clusters.values():
        spans = [
            (by_vehicle[m].stops[0].arrival_s, by_vehicle[m].stops[-1].arrival_s)
            for m in members
        ]
        latest_start = max(s for s, _ in spans)
        earliest_end = min(e for _, e in spans)
        assert latest_start < earliest_end, "cluster routes must overlap in time"


def test_sightings_time_ordered_and_complete():
    world = build_default_world()
    events = list(iter_sightings(world))
    times = [e.timestamp_s for e in events]
    assert times == sorted(times)
    assert len(events) == sum(len(r.stops) for r in world.routes)
    assert len({e.event_id for e in events}) == len(events)
    # Event GPS matches the emitting camera.
    for e in events[:20]:
        cam = world.graph.camera(e.camera_id)
        assert (e.lat, e.lon) == (cam.lat, cam.lon)


def test_route_generation_deterministic():
    g = default_world()
    fleet = generate_fleet()
    r1 = generate_routes(g, fleet, RouteConfig(seed=5))
    r2 = generate_routes(g, fleet, RouteConfig(seed=5))
    assert r1 == r2
