"""FleetTracker integration tests: association gating, ambiguity, reviews."""
import numpy as np
import pytest

from reasoning.profile import LastSeen
from sim.road_graph import default_world
from tracking.lifecycle import CONFIRMED, TENTATIVE
from tracking.tracker import FleetTracker
from tests.util import CAMRY, make_obs, make_profile, unit_vec


@pytest.fixture()
def graph():
    return default_world()


def flagged_tracker(graph, **profile_kwargs):
    tracker = FleetTracker(graph)
    tracker.flag_target(make_profile(**profile_kwargs))
    return tracker


def test_plate_sighting_confirms_and_updates_profile(graph):
    tracker = flagged_tracker(graph)
    events = tracker.process_observation(make_obs(plate="ABC-1234"))
    kinds = [e.kind for e in events]
    assert "association" in kinds
    assert "state_change" in kinds
    assert "profile_update" in kinds
    tracked = tracker.targets()["tgt-1"]
    assert tracked.track.state == CONFIRMED
    assert tracked.profile.version == 1
    assert tracked.profile.last_seen.event_id == "evt-1"


def test_likely_sighting_reviews_but_never_updates_profile(graph):
    tracker = flagged_tracker(graph, plate="",
                              instance_attrs={"accessory": "roof rack"})
    obs = make_obs(instance_attrs={"accessory": "roof rack"})
    events = tracker.process_observation(obs)
    kinds = [e.kind for e in events]
    assert "review" in kinds
    assert "profile_update" not in kinds
    tracked = tracker.targets()["tgt-1"]
    assert tracked.profile.version == 0
    assert tracked.track.state == TENTATIVE
    assert len(tracker.pending_reviews()) == 1


def test_transit_veto_blocks_association(graph):
    profile = make_profile(last_seen=LastSeen("cam-nw", 1000.0, "evt-0"))
    tracker = FleetTracker(graph)
    tracker.flag_target(profile)
    # Impossible: across the grid 5 seconds later, plate matching.
    events = tracker.process_observation(
        make_obs(camera_id="cam-e", t=1005.0, plate="ABC-1234"))
    kinds = [e.kind for e in events]
    assert "anomaly" in kinds, "plate match + impossible transit = anomaly review"
    assert "association" not in kinds
    assert tracker.targets()["tgt-1"].profile.version == 0


def test_ambiguous_lookalikes_go_to_review_not_association(graph):
    e = unit_vec(3)
    tracker = FleetTracker(graph)
    tracker.flag_target(make_profile(
        target_id="tgt-a", plate="", gallery=(e,),
        instance_attrs={"accessory": "roof rack"}))
    tracker.flag_target(make_profile(
        target_id="tgt-b", plate="", gallery=(e,),
        instance_attrs={"accessory": "roof rack"}))
    obs = make_obs(embedding=e, instance_attrs={"accessory": "roof rack"})
    events = tracker.process_observation(obs)
    assert [e.kind for e in events] == ["review"]
    review = tracker.pending_reviews()[0]
    assert review.rival_target_ids
    assert "Ambiguous" in events[0].detail["facts"]
    for t in tracker.targets().values():
        assert t.profile.version == 0
        assert t.track.consecutive_sightings == 0


def test_operator_accept_confirms_and_updates(graph):
    tracker = flagged_tracker(graph, plate="",
                              instance_attrs={"accessory": "roof rack"})
    obs = make_obs(instance_attrs={"accessory": "roof rack"})
    tracker.process_observation(obs)
    review = tracker.pending_reviews()[0]
    events = tracker.resolve_review(review.review_id, accept=True, now_s=1100.0)
    kinds = [e.kind for e in events]
    assert "association" in kinds and "profile_update" in kinds
    tracked = tracker.targets()["tgt-1"]
    assert tracked.track.state == CONFIRMED
    assert tracked.profile.version == 1
    assert tracked.corroboration.belief >= 0.95
    assert not tracker.pending_reviews()


def test_operator_reject_penalizes_belief(graph):
    tracker = flagged_tracker(graph, plate="",
                              instance_attrs={"accessory": "roof rack"})
    tracker.process_observation(make_obs(instance_attrs={"accessory": "roof rack"}))
    before = tracker.targets()["tgt-1"].corroboration.belief
    review = tracker.pending_reviews()[0]
    events = tracker.resolve_review(review.review_id, accept=False, now_s=1100.0)
    assert [e.kind for e in events] == ["rejection"]
    after = tracker.targets()["tgt-1"].corroboration.belief
    assert after < before
    assert tracker.targets()["tgt-1"].track.consecutive_sightings == 0


def test_tick_emits_state_changes(graph):
    tracker = flagged_tracker(graph)
    tracker.process_observation(make_obs(plate="ABC-1234", t=1000.0))
    events = tracker.tick(1000.0 + 300)
    assert [e.kind for e in events] == ["state_change"]
    assert events[0].detail["to"] == "coasting"


def test_snapshot_shape(graph):
    tracker = flagged_tracker(graph)
    tracker.process_observation(make_obs(plate="ABC-1234", t=1000.0))
    snap = tracker.snapshot(1010.0)
    entry = snap["tgt-1"]
    assert entry["state"] == CONFIRMED
    assert entry["position"] is not None
    assert entry["last_seen"]["camera_id"] == "cam-ctr"
    assert entry["next_cameras"], "predictions must be present after a sighting"
    statuses = {p["status"] for p in entry["next_cameras"]}
    assert statuses <= {"upcoming", "expected-now", "overdue"}


def test_unknown_review_raises(graph):
    tracker = flagged_tracker(graph)
    with pytest.raises(KeyError):
        tracker.resolve_review("rev-9999", True, 0.0)


def test_no_targets_no_events(graph):
    tracker = FleetTracker(graph)
    assert tracker.process_observation(make_obs()) == []
