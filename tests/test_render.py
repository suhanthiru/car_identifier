import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from sim.emitter import build_default_world
from sim.render import CROP_H, CROP_W, render_frame, render_vehicle_crop


@pytest.fixture(scope="module")
def world():
    return build_default_world()


def test_crop_shape_and_determinism(world):
    v = world.fleet[0]
    a = render_vehicle_crop(v, "cam-ctr", 42.0)
    b = render_vehicle_crop(v, "cam-ctr", 42.0)
    assert a.shape == (CROP_H, CROP_W, 3)
    assert np.array_equal(a, b), "same (vehicle, camera, t) must render identical pixels"


def test_lookalikes_render_nearly_identically(world):
    """Unmarked cluster members must be closer to each other in pixel space
    than either is to a different-class vehicle — that is the whole trap."""
    cluster = [v for v in world.fleet if v.lookalike_group == "cluster-1"]
    unmarked = [v for v in cluster if not v.instance_attrs]
    marked = [v for v in cluster if v.instance_attrs]
    other = next(v for v in world.fleet if not v.lookalike_group)

    def dist(v1, v2):
        a = render_vehicle_crop(v1, "cam-ctr", 10.0).astype(np.float32)
        b = render_vehicle_crop(v2, "cam-ctr", 10.0).astype(np.float32)
        return float(np.mean(np.abs(a - b)))

    within = dist(unmarked[0], marked[0])
    across = dist(unmarked[0], other)
    assert within < across


def test_frame_contains_vehicle(world):
    v = world.fleet[0]
    cam = world.graph.camera("cam-n")
    frame = render_frame(v, cam, 5.0)
    assert frame.shape[2] == 3
    # The composited crop should make the frame less uniform than blank road.
    assert frame.std() > 20
