"""bbox_for + VideoFrameSource against a tiny generated fixture video."""
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from datasets.cityflow_video import (
    VideoFrameSource, bbox_for, discover_camera_dirs, vehicle_frame_spans,
)

FRAME_W, FRAME_H = 64, 48


def make_video(path, n_frames=5):
    """A few solid-color frames, distinguishable by frame index."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (FRAME_W, FRAME_H))
    for i in range(n_frames):
        frame = np.full((FRAME_H, FRAME_W, 3), i * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def make_gt(path):
    # frame,id,left,top,width,height,1,-1,-1,-1
    path.write_text(
        "0,7,5,5,20,15,1,-1,-1,-1\n"
        "2,7,8,8,20,15,1,-1,-1,-1\n"
        "2,9,1,1,10,10,1,-1,-1,-1\n"
    )


def test_bbox_for_returns_matching_frame_and_vehicle(tmp_path):
    gt = tmp_path / "gt.txt"
    make_gt(gt)
    assert bbox_for(gt, 0, 7) == (5, 5, 20, 15)
    assert bbox_for(gt, 2, 7) == (8, 8, 20, 15)
    assert bbox_for(gt, 2, 9) == (1, 1, 10, 10)


def test_bbox_for_missing_frame_or_vehicle_returns_none(tmp_path):
    gt = tmp_path / "gt.txt"
    make_gt(gt)
    assert bbox_for(gt, 99, 7) is None
    assert bbox_for(gt, 0, 999) is None


def test_video_frame_source_crops_correct_region(tmp_path):
    video_path = tmp_path / "vdo.avi"
    make_video(video_path)
    src = VideoFrameSource(video_path)
    crop = src.crop(2, (8, 8, 20, 15))
    assert crop is not None
    assert crop.shape[:2] == (15, 20)
    src.close()


def test_video_frame_source_reuses_one_capture_handle(tmp_path):
    video_path = tmp_path / "vdo.avi"
    make_video(video_path)
    src = VideoFrameSource(video_path)
    with patch("cv2.VideoCapture", wraps=cv2.VideoCapture) as spy:
        src.crop(0, (5, 5, 20, 15))
        src.crop(2, (8, 8, 20, 15))
        src.crop(1, (0, 0, 10, 10))
        assert spy.call_count == 1
    src.close()


def test_video_frame_source_missing_file_raises(tmp_path):
    src = VideoFrameSource(tmp_path / "nope.avi")
    with pytest.raises(FileNotFoundError):
        src.crop(0, (0, 0, 5, 5))


def test_vehicle_frame_spans_first_and_last(tmp_path):
    gt = tmp_path / "gt.txt"
    make_gt(gt)  # vehicle 7 at frames 0 and 2; vehicle 9 at frame 2 only
    spans = vehicle_frame_spans(gt)
    assert spans[7] == (0, 2)
    assert spans[9] == (2, 2)


def test_discover_camera_dirs_finds_scenario_under_a_split(tmp_path):
    root = tmp_path / "CityFlow"
    cam_dir = root / "train" / "S01" / "c001"
    (cam_dir / "gt").mkdir(parents=True)
    (cam_dir / "gt" / "gt.txt").write_text("0,1,0,0,5,5,1,-1,-1,-1\n")
    # A camera dir without gt.txt must not be picked up.
    (root / "train" / "S01" / "c002").mkdir(parents=True)
    dirs = discover_camera_dirs(root, "S01")
    assert dirs == {"c001": cam_dir}


def test_discover_camera_dirs_missing_scenario_returns_empty(tmp_path):
    root = tmp_path / "CityFlow"
    (root / "train").mkdir(parents=True)
    assert discover_camera_dirs(root, "S99") == {}
