"""Constant-velocity alpha-beta smoother for map display.

Purely cosmetic: it turns discrete camera sightings into a smoothly moving
marker between cameras. It carries no evidential weight — association and
belief live in the reasoning layer, and the console labels dead-reckoned
positions as predictions, not observations.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

# Standard alpha-beta gains; responsive but not jittery for sparse updates.
ALPHA = 0.85
BETA = 0.005


@dataclass(frozen=True)
class SmootherState:
    lat: float
    lon: float
    vlat: float = 0.0  # deg/s
    vlon: float = 0.0
    t: float = 0.0


def init_state(lat: float, lon: float, t: float) -> SmootherState:
    return SmootherState(lat=lat, lon=lon, t=t)


def update(state: SmootherState, lat: float, lon: float, t: float) -> SmootherState:
    """Fold in a new observed position at time t."""
    dt = t - state.t
    if dt <= 0:
        # Out-of-order or duplicate timestamp: snap without velocity change.
        return replace(state, lat=lat, lon=lon)
    pred_lat = state.lat + state.vlat * dt
    pred_lon = state.lon + state.vlon * dt
    r_lat, r_lon = lat - pred_lat, lon - pred_lon
    return SmootherState(
        lat=pred_lat + ALPHA * r_lat,
        lon=pred_lon + ALPHA * r_lon,
        vlat=state.vlat + BETA * r_lat / dt,
        vlon=state.vlon + BETA * r_lon / dt,
        t=t,
    )


def predict(state: SmootherState, t: float, max_extrapolation_s: float = 120.0) -> tuple[float, float]:
    """Dead-reckoned display position at time t (clamped extrapolation)."""
    dt = min(max(0.0, t - state.t), max_extrapolation_s)
    return state.lat + state.vlat * dt, state.lon + state.vlon * dt
