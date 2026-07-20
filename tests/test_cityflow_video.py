"""bbox_for + VideoFrameSource against a tiny generated fixture video."""
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from datasets.cityflow_video import VideoFrameSource, bbox_for

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
