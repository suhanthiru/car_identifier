"""Road graph: camera nodes joined by directed edges with transit windows.

The default world is a fictional 8-camera grid ("Gridville"). Coordinates are
fabricated. Transit windows are derived from edge length and a speed envelope,
so the physical-plausibility veto in the reasoning layer has real teeth: the
simulator and the veto share the same physics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from sim.model import CameraSpec, TransitEdge

# Speed envelope used to turn distances into transit windows (m/s).
# ~72 km/h flat out, ~47 km/h typical urban flow, ~14 km/h worst-case crawl.
V_MAX_MS = 20.0
V_TYPICAL_MS = 13.0
V_MIN_MS = 4.0
# Grace buffer on the slow end: signals, brief stops.
MAX_WINDOW_BUFFER_S = 30.0

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def make_edge(src: CameraSpec, dst: CameraSpec, road_factor: float = 1.25) -> TransitEdge:
    """Build a directed edge whose window follows from geometry.

    `road_factor` inflates straight-line distance to approximate actual
    road distance.
    """
    dist = haversine_m(src.lat, src.lon, dst.lat, dst.lon) * road_factor
    return TransitEdge(
        src=src.camera_id,
        dst=dst.camera_id,
        distance_m=round(dist, 1),
        min_s=round(dist / V_MAX_MS, 1),
        typical_s=round(dist / V_TYPICAL_MS, 1),
        max_s=round(dist / V_MIN_MS + MAX_WINDOW_BUFFER_S, 1),
    )


@dataclass(frozen=True)
class RoadGraph:
    """Immutable camera graph with transit-window lookups."""

    cameras: tuple[CameraSpec, ...]
    edges: tuple[TransitEdge, ...]
    _by_id: dict[str, CameraSpec] = field(init=False, repr=False)
    _edge_map: dict[tuple[str, str], TransitEdge] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        by_id = {c.camera_id: c for c in self.cameras}
        if len(by_id) != len(self.cameras):
            raise ValueError("duplicate camera_id in graph")
        edge_map: dict[tuple[str, str], TransitEdge] = {}
        for e in self.edges:
            if e.src not in by_id or e.dst not in by_id:
                raise ValueError(f"edge {e.src}->{e.dst} references unknown camera")
            if e.min_s <= 0 or e.max_s < e.typical_s or e.typical_s < e.min_s:
                raise ValueError(f"edge {e.src}->{e.dst} has an inconsistent window")
            edge_map[(e.src, e.dst)] = e
        object.__setattr__(self, "_by_id", by_id)
        object.__setattr__(self, "_edge_map", edge_map)

    def camera(self, camera_id: str) -> CameraSpec:
        return self._by_id[camera_id]

    def camera_ids(self) -> tuple[str, ...]:
        return tuple(c.camera_id for c in self.cameras)

    def edge(self, src: str, dst: str) -> TransitEdge | None:
        return self._edge_map.get((src, dst))

    def neighbors(self, camera_id: str) -> tuple[str, ...]:
        return tuple(dst for (src, dst) in self._edge_map if src == camera_id)

    def transit_window(self, src: str, dst: str) -> tuple[float, float] | None:
        """(min_s, max_s) for a direct hop, or None if not adjacent."""
        e = self.edge(src, dst)
        return (e.min_s, e.max_s) if e else None

    def min_transit_s(self, src: str, dst: str) -> float | None:
        """Fastest physically possible time from src to dst over any path.

        Dijkstra over edge min_s. Used by the transit-time veto for
        non-adjacent camera pairs: even multi-hop, you cannot beat this.
        """
        if src == dst:
            return 0.0
        if src not in self._by_id or dst not in self._by_id:
            return None
        best: dict[str, float] = {src: 0.0}
        frontier = {src}
        while frontier:
            node = min(frontier, key=lambda n: best[n])
            frontier.discard(node)
            if node == dst:
                return best[node]
            for (s, d), e in self._edge_map.items():
                if s != node:
                    continue
                cand = best[node] + e.min_s
                if cand < best.get(d, math.inf):
                    best[d] = cand
                    frontier.add(d)
        return best.get(dst)


def _bidirectional(a: CameraSpec, b: CameraSpec) -> tuple[TransitEdge, TransitEdge]:
    return make_edge(a, b), make_edge(b, a)


def default_world() -> RoadGraph:
    """The fictional Gridville deployment: 8 cameras on a rough 3x3 grid.

    Layout (N up):

        NW ---- N ---- NE
        |       |       |
        W ---- CTR ---- E
        |       |       |
        SW ---- S ------+
    """
    # ~0.009 deg lat ≈ 1 km; grid spacing ~1.1 km. Entirely fictional coords.
    c = 0.010
    lat0, lon0 = 40.7300, -89.6100
    specs = [
        CameraSpec("cam-nw", "Northwest & 1st", lat0 + c, lon0 - c, 135.0),
        CameraSpec("cam-n", "North Gate", lat0 + c, lon0, 180.0),
        CameraSpec("cam-ne", "Northeast & 1st", lat0 + c, lon0 + c, 225.0),
        CameraSpec("cam-w", "West Ave", lat0, lon0 - c, 90.0),
        CameraSpec("cam-ctr", "Center Square", lat0, lon0, 0.0),
        CameraSpec("cam-e", "East Ave", lat0, lon0 + c, 270.0),
        CameraSpec("cam-sw", "Southwest & Main", lat0 - c, lon0 - c, 45.0),
        CameraSpec("cam-s", "South Bridge", lat0 - c, lon0, 0.0),
    ]
    by_id = {s.camera_id: s for s in specs}
    pairs = [
        ("cam-nw", "cam-n"), ("cam-n", "cam-ne"),
        ("cam-nw", "cam-w"), ("cam-n", "cam-ctr"), ("cam-ne", "cam-e"),
        ("cam-w", "cam-ctr"), ("cam-ctr", "cam-e"),
        ("cam-w", "cam-sw"), ("cam-ctr", "cam-s"), ("cam-e", "cam-s"),
        ("cam-sw", "cam-s"),
    ]
    edges: list[TransitEdge] = []
    for a, b in pairs:
        fwd, back = _bidirectional(by_id[a], by_id[b])
        edges.extend((fwd, back))
    return RoadGraph(cameras=tuple(specs), edges=tuple(edges))
