"""Detector wrapper tests. Marked slow: loads real YOLO weights.

On cartoon sprites YOLO usually finds nothing, so the honest-fallback path
is the main behavior to pin down. The candidate-selection logic is tested
directly without invoking the model.
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

from perception.detector import VehicleDetector, _crop
from sim.emitter import build_default_world
from sim.render import render_passage


def test_crop_bounds_clamped():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    c = _crop(frame, (-10, -10, 50, 50))
    assert c.shape == (50, 50, 3)
    with pytest.raises(ValueError):
        _crop(frame, (60, 60, 60, 90))


def test_pick_best_prefers_most_seen_track():
    det = VehicleDetector()
    frames = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(3)]
    candidates = [
        (0, 0.9, (0, 0, 10, 10), 1),      # track 1: seen once, high conf
        (0, 0.5, (0, 0, 40, 40), 2),      # track 2: seen three times
        (1, 0.6, (0, 0, 40, 40), 2),
        (2, 0.55, (0, 0, 40, 40), 2),
    ]
    best = det._pick_best(frames, candidates)
    assert best.source == "yolo"
    assert best.confidence == pytest.approx(0.6)


@pytest.mark.slow
def test_fallback_on_cartoon_sprites():
    """Real YOLO runs and (expectedly) misses the sprite; the wrapper must
    fall back to the sim box and say so."""
    world = build_default_world()
    v = world.fleet[0]
    burst = render_passage(v, world.graph.camera("cam-ctr"), 50.0, n_frames=2)
    det = VehicleDetector().best_detection(
        [f for f, _ in burst], [b for _, b in burst]
    )
    assert det is not None
    assert det.source in ("yolo", "sim-fallback")
    assert det.crop.size > 0
    if det.source == "sim-fallback":
        assert det.confidence == 0.0
