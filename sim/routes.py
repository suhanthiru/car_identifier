"""Route generation: vehicles walking the camera graph on realistic timings.

Travel times are sampled inside each edge's transit window (triangular
around the typical time), so every generated route is — by construction —
physically plausible. The reasoning layer's transit veto and this module
share the same physics via RoadGraph.

Look-alike clusters are routed to overlap in space and time on purpose:
members start near each other within a small time offset, so downstream
association genuinely has to disambiguate them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from sim.fleet import lookalike_groups
from sim.model import Route, RouteStop, VehicleIdentity
from sim.road_graph import RoadGraph


@dataclass(frozen=True)
class RouteConfig:
    min_stops: int = 4
    max_stops: int = 8
    # Vehicles enter the world spread over this many seconds.
    spawn_spread_s: float = 240.0
    # Look-alike cluster members spawn within this offset of each other.
    cluster_spawn_offset_s: float = 45.0
    seed: int = 11


def _sample_transit_s(rng: random.Random, min_s: float, typical_s: float, max_s: float) -> float:
    """Triangular sample inside the window; guaranteed plausible."""
    return rng.triangular(min_s, max_s, typical_s)


def _random_walk(rng: random.Random, graph: RoadGraph, start: str, n_stops: int) -> list[str]:
    """Walk the graph without immediately backtracking (when avoidable)."""
    path = [start]
    prev: str | None = None
    while len(path) < n_stops:
        options = list(graph.neighbors(path[-1]))
        if not options:
            break
        forward = [o for o in options if o != prev]
        prev = path[-1]
        path.append(rng.choice(forward or options))
    return path


def _realize(rng: random.Random, graph: RoadGraph, vehicle_id: str,
             cameras: list[str], start_s: float) -> Route:
    stops = [RouteStop(camera_id=cameras[0], arrival_s=round(start_s, 1))]
    t = start_s
    for src, dst in zip(cameras, cameras[1:]):
        edge = graph.edge(src, dst)
        if edge is None:
            raise ValueError(f"route hop {src}->{dst} is not a graph edge")
        t += _sample_transit_s(rng, edge.min_s, edge.typical_s, edge.max_s)
        stops.append(RouteStop(camera_id=dst, arrival_s=round(t, 1)))
    return Route(vehicle_id=vehicle_id, stops=tuple(stops))


def generate_routes(
    graph: RoadGraph,
    fleet: tuple[VehicleIdentity, ...],
    config: RouteConfig | None = None,
) -> tuple[Route, ...]:
    """One physically-plausible route per vehicle.

    Cluster members share a start camera and a tight spawn window so their
    sightings interleave; background vehicles spawn anywhere, anytime.
    """
    cfg = config or RouteConfig()
    rng = random.Random(cfg.seed)
    cam_ids = list(graph.camera_ids())
    routes: list[Route] = []

    clustered = lookalike_groups(fleet)
    cluster_anchor: dict[str, tuple[str, float]] = {}
    for group in clustered:
        cluster_anchor[group] = (
            rng.choice(cam_ids),
            rng.uniform(0.0, cfg.spawn_spread_s),
        )

    for v in fleet:
        n_stops = rng.randint(cfg.min_stops, cfg.max_stops)
        if v.lookalike_group:
            start_cam, anchor_t = cluster_anchor[v.lookalike_group]
            start_t = anchor_t + rng.uniform(0.0, cfg.cluster_spawn_offset_s)
        else:
            start_cam = rng.choice(cam_ids)
            start_t = rng.uniform(0.0, cfg.spawn_spread_s)
        cameras = _random_walk(rng, graph, start_cam, n_stops)
        routes.append(_realize(rng, graph, v.vehicle_id, cameras, start_t))
    return tuple(routes)
