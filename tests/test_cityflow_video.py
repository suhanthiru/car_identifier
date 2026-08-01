"""bbox_for + VideoFrameSource against a tiny generated fixture video."""
import threading
import time
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


def test_frame_at_repeated_request_returns_the_same_frame(tmp_path):
    """Asking twice for the frame you just got must not answer "no frame".

    The forward-read fast path skipped its loop entirely when the requested
    frame was the one already decoded, returned None, and the frame endpoint
    turned that into a 404 the console shows as "this camera's footage has
    ended". A paused replay polls the SAME clock value every 500 ms, so every
    poll after the first reported all five cameras dead.
    """
    video_path = tmp_path / "vdo.avi"
    make_video(video_path)
    src = VideoFrameSource(video_path)
    first = src.frame_at(2)
    assert first is not None
    again = src.frame_at(2)
    assert again is not None
    assert np.array_equal(first, again)
    # And a third time, plus after a backwards jump that clears the cache.
    assert src.frame_at(2) is not None
    assert src.frame_at(0) is not None
    assert src.frame_at(0) is not None
    src.close()


def test_frame_at_past_end_then_valid_frame_recovers(tmp_path):
    """One out-of-range request must not poison the capture for later ones."""
    video_path = tmp_path / "vdo.avi"
    make_video(video_path, n_frames=5)
    src = VideoFrameSource(video_path)
    assert src.frame_at(1) is not None
    assert src.frame_at(999) is None
    assert src.frame_at(1) is not None
    src.close()


def test_crop_then_frame_at_same_frame_still_decodes(tmp_path):
    """crop() moves the decoder and drops the cache; frame_at must re-seek
    rather than trust a position with no pixels behind it."""
    video_path = tmp_path / "vdo.avi"
    make_video(video_path)
    src = VideoFrameSource(video_path)
    assert src.crop(2, (8, 8, 20, 15)) is not None
    img = src.frame_at(2)
    assert img is not None
    assert img.shape[:2] == (FRAME_H, FRAME_W)
    src.close()


def test_close_racing_a_reader_does_not_crash(tmp_path):
    """close() must not release the capture out from under a reader.

    Releasing a cv2.VideoCapture while another thread is inside set()/read()
    is an access violation: the process dies with no Python exception and no
    traceback. A replay restart does exactly that — it tears down the feed's
    RealPerceptor and closes its sources while `asyncio.to_thread` workers
    from the outgoing run are still inside crop(). Caught with the fault
    handler armed after repeated resets:

        Windows fatal exception: access violation
          File "datasets/cityflow_video.py", line 176 in crop
          File "perception/real_observe.py", line 124 in process

    This cannot assert "did not segfault" from inside the process it would
    kill — if the lock is removed, this test takes the whole pytest run down
    rather than failing. That is the honest shape of the check.
    """
    video_path = tmp_path / "vdo.avi"
    make_video(video_path, n_frames=5)
    src = VideoFrameSource(video_path)
    src.crop(0, (2, 2, 10, 8))          # open the handle first

    stop = threading.Event()
    errors: list[BaseException] = []

    def reader():
        while not stop.is_set():
            try:
                src.crop(1, (2, 2, 10, 8))
                src.frame_at(2)
            except Exception as exc:     # noqa: BLE001 — recorded, not raised
                errors.append(exc)
                return

    threads = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    for _ in range(25):
        src.close()                      # racing the readers, repeatedly
        time.sleep(0.002)
    stop.set()
    for t in threads:
        t.join(timeout=5)
    src.close()
    assert not errors, f"reader raised while close() raced it: {errors[:3]}"


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
