"""CV smoother + next-camera predictor tests."""
import pytest

from sim.road_graph import default_world
from tracking.predictor import (
    STATUS_EXPECTED, STATUS_OVERDUE, STATUS_UPCOMING, all_overdue, predict_next_cameras,
)
from tracking.smoother import init_state, predict, update


def test_smoother_converges_on_constant_velocity():
    # Vehicle moving at a constant 1e-4 deg/s north.
    s = init_state(40.0, -89.0, 0.0)
    for i in range(1, 30):
        s = update(s, 40.0 + i * 1e-4 * 10, -89.0, i * 10.0)
    lat, lon = predict(s, 300.0)
    assert lat == pytest.approx(40.0 + 300 * 1e-4, abs=2e-3)
    assert lon == pytest.approx(-89.0, abs=1e-6)


def test_smoother_handles_duplicate_timestamps():
    s = init_state(40.0, -89.0, 100.0)
    s2 = update(s, 40.001, -89.0, 100.0)  # same t: snap, no velocity blowup
    assert s2.lat == pytest.approx(40.001)
    assert s2.vlat == 0.0


def test_prediction_extrapolation_is_clamped():
    s = init_state(40.0, -89.0, 0.0)
    s = update(s, 40.01, -89.0, 10.0)
    far = predict(s, 100000.0)
    clamped = predict(s, s.t + 120.0)
    assert far == clamped


def test_next_camera_statuses():
    g = default_world()
    window = g.transit_window("cam-n", "cam-ctr")
    last_seen = 1000.0
    early = predict_next_cameras(g, "cam-n", last_seen, last_seen + window[0] / 2)
    ctr_early = next(p for p in early if p.camera_id == "cam-ctr")
    assert ctr_early.status == STATUS_UPCOMING

    inside = predict_next_cameras(g, "cam-n", last_seen, last_seen + window[0] + 5)
    ctr_inside = next(p for p in inside if p.camera_id == "cam-ctr")
    assert ctr_inside.status == STATUS_EXPECTED

    # Far beyond every neighbor's max window.
    max_end = max(g.transit_window("cam-n", n)[1] for n in g.neighbors("cam-n"))
    late = predict_next_cameras(g, "cam-n", last_seen, last_seen + max_end + 60)
    assert all(p.status == STATUS_OVERDUE for p in late)
    assert all_overdue(late)


def test_windows_are_absolute_times():
    g = default_world()
    preds = predict_next_cameras(g, "cam-n", 5000.0, 5000.0)
    for p in preds:
        w = g.transit_window("cam-n", p.camera_id)
        assert p.window_start_s == pytest.approx(5000.0 + w[0])
        assert p.window_end_s == pytest.approx(5000.0 + w[1])
