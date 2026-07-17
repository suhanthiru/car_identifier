"""Render-and-compare tests with cargen stub backends (CPU, no GPU).

Uses a lightweight fake Target3DModel so the abstain gates and comparison
channels are unit-tested without the slow real fusion pipeline.
"""
from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("cargen")
cv2 = pytest.importorskip("cv2")

from car3d.calibration import (
    RenderMatchModel, fit, labeled_version, load_model, save,
)
from car3d.geometry import GeometrySignature
from car3d.match import (
    VerifyConfig, domain_match, pose_search, robust_compare, verify_match,
)
from cargen.core.splat import GaussianCloud, Provenance


def box_cloud(l=2.0, w=0.8, h=0.6, n=4000, observed_frac=1.0, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform([-l / 2, -w / 2, 0], [l / 2, w / 2, h], size=(n, 3))
    prov = np.full(n, Provenance.OBSERVED, np.uint8)
    prov[: int(n * (1 - observed_frac))] = Provenance.PRIOR
    return GaussianCloud.create(pts.astype(np.float32),
                                np.full((n, 3), 0.5, np.float32), provenance=prov)


@dataclass
class FakeAsset:
    cloud: object
    observations: list


class FakeModel:
    def __init__(self, cloud, n_obs, observed_frac):
        self._cloud = cloud
        self._n_obs = n_obs
        self._frac = observed_frac

    def exists(self):
        return self._cloud.n > 0

    def load(self):
        return FakeAsset(self._cloud, [{"i": i} for i in range(self._n_obs)])

    def geometry(self):
        return GeometrySignature(2.0, 0.8, 0.6, 2.5, 0.3, "low", "compact",
                                 self._cloud.n, self._frac)


def fake_embed(bgr):
    v = bgr[::8, ::8].astype(np.float32).mean(axis=(0, 1))
    v = np.concatenate([v, [bgr.mean(), bgr.std()]]) + 1.0
    return v / np.linalg.norm(v)


def crop_of(cloud, az=0.7):
    from car3d.match import _render, _orbit_pose
    render, _ = _render(cloud, _orbit_pose(cloud, az), (96, 72))
    return render


# ------------------------------------------------------------- domain match

def test_domain_match_reaches_crop_size_and_degrades():
    cloud = box_cloud()
    crop = np.zeros((72, 96, 3), np.uint8)
    render = crop_of(cloud)
    out = domain_match(render, crop)
    assert out.shape == crop.shape and out.dtype == np.uint8


def test_robust_compare_channels():
    cloud = box_cloud()
    render = crop_of(cloud, az=0.7)
    _, alpha = __import__("car3d.match", fromlist=["_render"])._render(
        cloud, __import__("car3d.match", fromlist=["_orbit_pose"])._orbit_pose(cloud, 0.7),
        (render.shape[1], render.shape[0]))
    degraded = domain_match(render, render)
    iou, cos = robust_compare(degraded, render, alpha, fake_embed)
    assert 0.0 <= iou <= 1.0 and -1.0 <= cos <= 1.0
    # identical target render vs itself: strong silhouette agreement
    assert iou > 0.3


# -------------------------------------------------------------- pose search

def test_pose_search_returns_candidate():
    cloud = box_cloud()
    crop = crop_of(cloud, az=1.2)
    est = pose_search(cloud, crop, fake_embed, VerifyConfig(azimuths=6))
    assert 0.0 <= est.confidence <= 1.0
    assert est.method == "search" and est.pose is not None


# ------------------------------------------------------------ verify + gates

def test_maturity_gate_abstains():
    model = FakeModel(box_cloud(observed_frac=0.05), n_obs=5, observed_frac=0.05)
    r = verify_match(model, crop_of(box_cloud()), fake_embed)
    assert r.abstained and r.score is None
    assert "immature" in r.reason


def test_too_few_observations_abstains():
    model = FakeModel(box_cloud(observed_frac=0.9), n_obs=1, observed_frac=0.9)
    r = verify_match(model, crop_of(box_cloud()), fake_embed)
    assert r.abstained and "obs" in r.reason


def test_pose_gate_abstains_when_configured_strict():
    cloud = box_cloud(observed_frac=0.9)
    model = FakeModel(cloud, n_obs=5, observed_frac=0.9)
    cfg = VerifyConfig(pose_min_confidence=0.99, azimuths=6)  # impossibly strict
    r = verify_match(model, crop_of(cloud), fake_embed, config=cfg)
    assert r.abstained and "pose" in r.reason


def test_mature_model_produces_score():
    cloud = box_cloud(observed_frac=0.9)
    model = FakeModel(cloud, n_obs=5, observed_frac=0.9)
    r = verify_match(model, crop_of(cloud), fake_embed, config=VerifyConfig(azimuths=6))
    assert not r.abstained
    assert r.score is not None and 0.0 <= r.raw <= 1.0
    assert r.calibration_version == "rendercmp-uncalibrated"  # no calibrator supplied
    assert r.degraded_render is not None


def test_calibrator_maps_to_probability():
    cloud = box_cloud(observed_frac=0.9)
    model = FakeModel(cloud, n_obs=5, observed_frac=0.9)
    # monotone calibrator: high raw -> high P(same)
    cal = fit([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              [False, False, False, False, True, True, True, True, True, True])
    r = verify_match(model, crop_of(cloud), fake_embed, calibrator=cal,
                     config=VerifyConfig(azimuths=6))
    assert r.calibration_version == cal.version
    assert 0.0 <= r.score <= 1.0


# ------------------------------------------------------------- calibration

def test_calibration_roundtrip_and_label(tmp_path):
    cal = fit([i / 20 for i in range(20)], [i >= 10 for i in range(20)])
    path = tmp_path / "rc.json"
    save(cal, path)
    loaded = load_model(path)
    assert loaded.version == cal.version
    assert loaded.predict(0.9) == pytest.approx(cal.predict(0.9))
    assert labeled_version(cal).startswith("rendercmp-")
    assert labeled_version(cal) != f"isotonic-{cal.version}"  # distinct namespace
    assert "renderer" in loaded.note


def test_calibration_rejects_tiny():
    with pytest.raises(ValueError):
        fit([0.1, 0.2], [True, False])
