"""Event emitter: turns realized routes into a time-ordered sighting stream.

Each emitted SightingEvent is what a (simulated) edge camera node would
report: where, when, and the ground-truth vehicle that passed. Observation
noise (missed detections, plate misreads, embedding jitter) is injected
later, in the perception layer — the emitter itself is noiseless truth.

In the live demo each camera is a local process/task consuming its slice of
this stream; that is the "edge tier" and it is simulated, not real hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from sim.model import Route, SightingEvent, VehicleIdentity
from sim.road_graph import RoadGraph


@dataclass(frozen=True)
class SimWorld:
    """Bundle of everything the simulator produced for one scenario run."""

    graph: RoadGraph
    fleet: tuple[VehicleIdentity, ...]
    routes: tuple[Route, ...]

    def vehicle(self, vehicle_id: str) -> VehicleIdentity:
        for v in self.fleet:
            if v.vehicle_id == vehicle_id:
                return v
        raise KeyError(vehicle_id)


def iter_sightings(world: SimWorld) -> Iterator[SightingEvent]:
    """Yield all sightings across all routes in global time order."""
    flat: list[tuple[float, str, str]] = []  # (t, camera_id, vehicle_id)
    for route in world.routes:
        for stop in route.stops:
            flat.append((stop.arrival_s, stop.camera_id, route.vehicle_id))
    flat.sort(key=lambda x: (x[0], x[1], x[2]))

    for seq, (t, camera_id, vehicle_id) in enumerate(flat):
        cam = world.graph.camera(camera_id)
        yield SightingEvent(
            event_id=f"evt-{seq:05d}",
            camera_id=camera_id,
            timestamp_s=t,
            lat=cam.lat,
            lon=cam.lon,
            truth=world.vehicle(vehicle_id),
        )


def build_default_world(fleet_seed: int = 7, route_seed: int = 11) -> SimWorld:
    """Convenience: default graph + fleet + routes, fully deterministic."""
    from sim.fleet import FleetConfig, generate_fleet
    from sim.road_graph import default_world
    from sim.routes import RouteConfig, generate_routes

    graph = default_world()
    fleet = generate_fleet(FleetConfig(seed=fleet_seed))
    routes = generate_routes(graph, fleet, RouteConfig(seed=route_seed))
    return SimWorld(graph=graph, fleet=fleet, routes=routes)
