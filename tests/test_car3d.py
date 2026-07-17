"""car3d bridge tests: geometry invariance, gated fusion, rendering, and
the cascade's geometry check. Uses cargen's CPU stub backends only."""
import numpy as np
import pytest

pytest.importorskip("cargen")
cv2 = pytest.importorskip("cv2")

from car3d.geometry import (
    GeometrySignature, compare_signatures, signature_from_cloud, signature_to_attrs,
)
from car3d.profile_model import Target3DModel
from car3d.render import render_and_compare, turntable_strip
from cargen.core.splat import GaussianCloud, Provenance
from reasoning.cascade import evaluate
from reasoning.plausibility import check_geometry
from sim.road_graph import default_world
from tests.util import make_obs, make_profile


def box_cloud(l=2.0, w=0.8, h=0.6, n=4000, observed_frac=1.0, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform([-l / 2, -w / 2, 0], [l / 2, w / 2, h], size=(n, 3))
    prov = np.full(n, Provenance.OBSERVED, np.uint8)
    prov[: int(n * (1 - observed_frac))] = Provenance.PRIOR
    return GaussianCloud.create(
        pts.astype(np.float32), np.full((n, 3), 0.5, np.float32), provenance=prov)


def rotate(cloud, yaw=0.8, pitch=0.2):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ \
        np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    from dataclasses import replace

    return replace(cloud, positions=(cloud.positions @ R.T).astype(np.float32))


def test_signature_measures_known_box():
    sig = signature_from_cloud(box_cloud())
    assert sig.lw_ratio == pytest.approx(2.5, rel=0.06)
    assert sig.hl_ratio == pytest.approx(0.3, rel=0.1)
    assert sig.body_profile == "low"
    assert sig.trustworthy


def test_signature_is_view_invariant():
    a = signature_from_cloud(box_cloud())
    b = signature_from_cloud(rotate(box_cloud()))
    assert b.lw_ratio == pytest.approx(a.lw_ratio, rel=0.08)
    assert b.hl_ratio == pytest.approx(a.hl_ratio, rel=0.12)
    assert compare_signatures(a, b).verdict == "consistent"


def test_untrusted_geometry_emits_no_attrs():
    guessy = signature_from_cloud(box_cloud(observed_frac=0.05))
    assert not guessy.trustworthy
    assert signature_to_attrs(guessy) == {}
    assert compare_signatures(guessy, signature_from_cloud(box_cloud())).verdict \
        == "insufficient"


def test_tiny_cloud_has_no_signature():
    assert signature_from_cloud(box_cloud(n=50)) is None


def test_geometry_check_supports_and_cautions():
    graph = default_world()
    attrs = {"geom3d:body_profile": "low", "geom3d:length_class": "standard"}
    obs = make_obs(instance_attrs=dict(attrs))
    profile = make_profile(plate="", instance_attrs=dict(attrs))
    facts = check_geometry(obs, profile)
    assert any(f.kind == "support" for f in facts)
    # Mismatch must caution, never veto — and must not trip the mark veto.
    obs2 = make_obs(instance_attrs={**attrs, "geom3d:body_profile": "tall"})
    d = evaluate(obs2, profile, graph)
    kinds = {f.kind for f in d.facts if f.check == "geometry"}
    assert "caution" in kinds and "veto" not in kinds
    assert d.verdict != "rejected"
    # Consistent geometry contributes score at the attribute tier.
    d_match = evaluate(make_obs(instance_attrs=dict(attrs)), profile, graph)
    d_none = evaluate(make_obs(), make_profile(plate=""), graph)
    assert d_match.score > d_none.score


def test_fusion_is_snapshotted_and_reversible(tmp_path):
    from sim.emitter import build_default_world
    from sim.render import render_vehicle_crop

    world = build_default_world()
    crop = render_vehicle_crop(world.fleet[0], "cam-ctr", 10.0)
    model = Target3DModel("tgt-x", storage_root=tmp_path, prior_points=3000)
    out1 = model.fuse_confirmed_crop(crop, "evt-1", "operator confirmed")
    assert out1.accepted and model.exists()
    assert out1.n_splats > 0
    assert (tmp_path / "tgt-x" / "exports" / "model.splat").exists()
    assert (tmp_path / "tgt-x" / "exports" / "model_provenance.ply").exists()

    crop2 = render_vehicle_crop(world.fleet[0], "cam-e", 400.0)
    out2 = model.fuse_confirmed_crop(crop2, "evt-2", "plate confirmed")
    assert out2.observations == 2
    asset = model.load()
    assert asset.observations[-1]["gate_reason"] == "plate confirmed"

    # Roll back the second fusion; observation log remains (audit trail),
    # geometry reverts to the snapshotted cloud.
    model.rollback_to(out2.snapshot)
    assert model.load().cloud.n == out1.n_splats


def test_turntable_and_render_compare(tmp_path):
    cloud = box_cloud(n=2000)
    strip = turntable_strip(cloud, n_views=4, size=(80, 60))
    assert strip.shape == (60, 320, 3)
    overlay = turntable_strip(cloud, n_views=4, size=(80, 60), provenance_overlay=True)
    assert not np.array_equal(strip, overlay)

    def fake_embed(img_bgr):
        v = img_bgr.astype(np.float32).mean(axis=(0, 1)) + 1.0
        return v / np.linalg.norm(v)

    sim = render_and_compare(cloud, strip[:, :80], fake_embed, n_azimuths=4)
    assert 0.0 <= sim <= 1.0
