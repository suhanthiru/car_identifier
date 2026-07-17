"""Perception glue tests.

The embedder/detector tests exercise real models (OSNet, YOLO) on synthetic
sprites, so they are slower than the pure-logic suites. The properties they
pin down are structural: look-alikes must collide in embedding space, and
provenance labels must be honest.
"""
import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("torch")

from perception.attributes import AttributeNoiseConfig, estimate_color, perceive_instance_attrs
from perception.embedder import ReidEmbedder, cosine_similarity, max_similarity
from perception.observe import PerceptionConfig, Perceptor
from perception.plates import PlateNoiseConfig, SimulatedPlateReader
from sim.emitter import build_default_world, iter_sightings
from sim.render import render_vehicle_crop


@pytest.fixture(scope="module")
def world():
    return build_default_world()


@pytest.fixture(scope="module")
def embedder():
    return ReidEmbedder()


def test_embeddings_normalized(world, embedder):
    v = world.fleet[0]
    e = embedder.embed(render_vehicle_crop(v, "cam-ctr", 5.0))
    assert e.ndim == 1
    assert np.isclose(np.linalg.norm(e), 1.0, atol=1e-5)


def test_lookalikes_collide_in_embedding_space(world, embedder):
    """The premise of the whole project: appearance embeddings cannot
    separate look-alikes, so similarity must rank a look-alike sibling
    above a different-class vehicle."""
    cluster = [v for v in world.fleet if v.lookalike_group == "cluster-1"]
    a, b = cluster[0], cluster[-1]
    other = next(v for v in world.fleet if not v.lookalike_group
                 and v.body_type != a.body_type)
    ea = embedder.embed(render_vehicle_crop(a, "cam-ctr", 5.0))
    eb = embedder.embed(render_vehicle_crop(b, "cam-e", 90.0))
    eo = embedder.embed(render_vehicle_crop(other, "cam-e", 90.0))
    assert cosine_similarity(ea, eb) > cosine_similarity(ea, eo)


def test_max_similarity_over_gallery(embedder, world):
    v = world.fleet[0]
    e1 = embedder.embed(render_vehicle_crop(v, "cam-ctr", 5.0))
    e2 = embedder.embed(render_vehicle_crop(v, "cam-n", 300.0))
    assert max_similarity(e1, []) == 0.0
    assert max_similarity(e1, [e2]) == pytest.approx(cosine_similarity(e1, e2))


def test_plate_reader_deterministic_and_noisy():
    reader = SimulatedPlateReader(PlateNoiseConfig(read_prob=1.0, char_error_prob=0.15))
    a = reader.read("ABC-1234", "evt-1")
    b = reader.read("ABC-1234", "evt-1")
    assert a == b, "same event must always read the same"
    reads = [reader.read("ABC-1234", f"evt-{i}") for i in range(300)]
    wrong = [r for r in reads if r.text != "ABC-1234"]
    assert 0 < len(wrong) < 300, "noise must corrupt some but not all reads"
    for r in reads:
        assert r.source == "sim"


def test_plate_reader_missed_reads():
    reader = SimulatedPlateReader(PlateNoiseConfig(read_prob=0.5))
    reads = [reader.read("ABC-1234", f"evt-{i}") for i in range(200)]
    misses = sum(r is None for r in reads)
    assert 60 < misses < 140  # ~50% within tolerance


def test_color_heuristic_reads_sprite_color(world):
    hits = 0
    for v in world.fleet[:8]:
        crop = render_vehicle_crop(v, "cam-ctr", 5.0)
        if estimate_color(crop) == v.color:
            hits += 1
    # Camera tint makes this imperfect by design, but it must mostly work.
    assert hits >= 5


def test_instance_attr_misses(world):
    cfg = AttributeNoiseConfig(instance_attr_miss_prob=1.0)
    marked = next(v for v in world.fleet if v.instance_attrs)
    assert perceive_instance_attrs(marked, "evt-1", cfg) == {}
    cfg2 = AttributeNoiseConfig(instance_attr_miss_prob=0.0)
    assert dict(perceive_instance_attrs(marked, "evt-1", cfg2)) == dict(marked.instance_attrs)


def test_perceptor_end_to_end_fast_mode(world):
    perceptor = Perceptor(world.graph, PerceptionConfig(miss_prob=0.0))
    events = list(iter_sightings(world))[:6]
    obs = [perceptor.process(e) for e in events]
    assert all(o is not None for o in obs)
    for e, o in zip(events, obs):
        assert o.event_id == e.event_id
        assert o.detection_source == "sim-fallback"
        assert o.eval_truth_id == e.truth.vehicle_id
        assert np.isclose(np.linalg.norm(o.embedding), 1.0, atol=1e-5)
        assert set(o.class_attrs) == {"make", "model", "body_type", "color"}


def test_perceptor_misses_some(world):
    perceptor = Perceptor(world.graph, PerceptionConfig(miss_prob=1.0))
    event = next(iter(iter_sightings(world)))
    assert perceptor.process(event) is None
