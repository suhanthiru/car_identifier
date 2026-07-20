"""Passage selection: one representative sighting per (vehicle, camera)
passage, at the midpoint frame, in real time-order. The async HTTP replay
loop itself is exercised live via scripts/run_cityflow_console.py, not
unit-tested here (it's a thin asyncio.sleep/httpx.post wrapper)."""
import cv2
import numpy as np

from datasets.cityflow import CityFlowScenario, TrackSpan
from server.real_feed import _passages_by_camera, build_vehicle_index


def make_gt(path, rows):
    path.write_text("\n".join(rows))


def test_passages_use_midpoint_frame_and_real_timestamp(tmp_path):
    cam_dir = tmp_path / "c001"
    (cam_dir / "gt").mkdir(parents=True)
    # vehicle 7 spans frames 0..40 at this camera (10 fps -> 0.0s..4.0s)
    make_gt(cam_dir / "gt" / "gt.txt", [
        "0,7,0,0,5,5,1,-1,-1,-1",
        "40,7,0,0,5,5,1,-1,-1,-1",
    ])
    scenario = CityFlowScenario(
        name="S01", cameras=("c001",), homographies={},
        spans=(TrackSpan("S01", "c001", 7, enter_s=0.0, exit_s=4.0),))
    out = _passages_by_camera(scenario, {"c001": cam_dir})
    assert len(out["c001"]) == 1
    p = out["c001"][0]
    assert p.vehicle_id == 7
    assert p.frame == 20          # midpoint of frames 0 and 40
    assert p.timestamp_s == 2.0   # midpoint of the real TrackSpan seconds


def test_passages_sorted_by_timestamp_across_vehicles(tmp_path):
    cam_dir = tmp_path / "c001"
    (cam_dir / "gt").mkdir(parents=True)
    make_gt(cam_dir / "gt" / "gt.txt", [
        "0,1,0,0,5,5,1,-1,-1,-1", "10,1,0,0,5,5,1,-1,-1,-1",
        "0,2,0,0,5,5,1,-1,-1,-1", "10,2,0,0,5,5,1,-1,-1,-1",
    ])
    scenario = CityFlowScenario(
        name="S01", cameras=("c001",), homographies={},
        spans=(
            TrackSpan("S01", "c001", 2, enter_s=50.0, exit_s=51.0),
            TrackSpan("S01", "c001", 1, enter_s=5.0, exit_s=6.0),
        ))
    passages = _passages_by_camera(scenario, {"c001": cam_dir})["c001"]
    assert [p.vehicle_id for p in passages] == [1, 2]


def test_vehicle_with_no_gt_rows_at_that_camera_is_skipped(tmp_path):
    cam_dir = tmp_path / "c001"
    (cam_dir / "gt").mkdir(parents=True)
    make_gt(cam_dir / "gt" / "gt.txt", ["0,1,0,0,5,5,1,-1,-1,-1"])
    scenario = CityFlowScenario(
        name="S01", cameras=("c001",), homographies={},
        spans=(TrackSpan("S01", "c001", 999, enter_s=0.0, exit_s=1.0),))
    assert _passages_by_camera(scenario, {"c001": cam_dir})["c001"] == []


def test_camera_without_a_directory_is_dropped(tmp_path):
    scenario = CityFlowScenario(
        name="S01", cameras=("c001",), homographies={},
        spans=(TrackSpan("S01", "c001", 1, enter_s=0.0, exit_s=1.0),))
    assert _passages_by_camera(scenario, {}) == {}


def _make_camera(tmp_path, name, rows):
    cam_dir = tmp_path / name
    (cam_dir / "gt").mkdir(parents=True)
    (cam_dir / "gt" / "gt.txt").write_text("\n".join(rows))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(cam_dir / "vdo.avi"), fourcc, 10.0, (32, 24))
    for i in range(3):
        writer.write(np.full((24, 32, 3), i * 50, dtype=np.uint8))
    writer.release()
    return cam_dir


def test_vehicle_index_picks_earliest_appearance_with_a_real_thumbnail(tmp_path):
    cam1 = _make_camera(tmp_path, "c001", ["0,7,2,2,10,8,1,-1,-1,-1"])
    cam2 = _make_camera(tmp_path, "c002", ["0,7,1,1,10,8,1,-1,-1,-1"])
    scenario = CityFlowScenario(
        name="S01", cameras=("c001", "c002"), homographies={},
        spans=(
            TrackSpan("S01", "c002", 7, enter_s=5.0, exit_s=6.0),   # earlier
            TrackSpan("S01", "c001", 7, enter_s=50.0, exit_s=51.0),
        ))
    index = build_vehicle_index(scenario, {"c001": cam1, "c002": cam2})
    assert len(index) == 1
    entry = index[0]
    assert entry["vehicle_id"] == 7
    assert entry["first_camera"] == "c002"
    assert entry["first_time_s"] == 5.0
    assert entry["thumbnail_b64"]  # a real crop was encoded


def test_vehicle_index_covers_every_vehicle_sorted_by_id(tmp_path):
    cam = _make_camera(tmp_path, "c001", [
        "0,2,0,0,5,5,1,-1,-1,-1", "0,1,0,0,5,5,1,-1,-1,-1",
    ])
    scenario = CityFlowScenario(
        name="S01", cameras=("c001",), homographies={},
        spans=(
            TrackSpan("S01", "c001", 2, enter_s=0.0, exit_s=1.0),
            TrackSpan("S01", "c001", 1, enter_s=0.0, exit_s=1.0),
        ))
    index = build_vehicle_index(scenario, {"c001": cam})
    assert [e["vehicle_id"] for e in index] == [1, 2]
