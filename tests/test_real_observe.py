"""RealPerceptor: real crops from a fixture video, real color heuristic,
plate OCR gated by a runtime toggle. Real ReidEmbedder (small OSNet model,
same pattern as tests/test_perception.py), plate reader stubbed."""
import cv2
import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("torch")

from perception.embedder import ReidEmbedder
from perception.real_observe import RealPerceptor
from perception.types import PlateRead, SOURCE_MODEL

FRAME_W, FRAME_H = 64, 48


class _State:
    def __init__(self, enable_plate_ocr: bool):
        self.enable_plate_ocr = enable_plate_ocr


class _StubPlateReader:
    def __init__(self, result):
        self._result = result
        self.calls = 0

    def read(self, crop_bgr):
        self.calls += 1
        return self._result


def make_scenario_dir(tmp_path):
    cam_dir = tmp_path / "c001"
    (cam_dir / "gt").mkdir(parents=True)
    (cam_dir / "gt" / "gt.txt").write_text("0,7,5,5,20,15,1,-1,-1,-1\n")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(cam_dir / "vdo.avi"), fourcc, 10.0, (FRAME_W, FRAME_H))
    writer.write(np.full((FRAME_H, FRAME_W, 3), 180, dtype=np.uint8))
    writer.release()
    return {"c001": cam_dir}


@pytest.fixture(scope="module")
def embedder():
    return ReidEmbedder()


def test_process_returns_color_only_no_marks(tmp_path, embedder):
    dirs = make_scenario_dir(tmp_path)
    perceptor = RealPerceptor(dirs, embedder=embedder,
                              plate_reader=_StubPlateReader(None))
    obs = perceptor.process("c001", vehicle_id=7, frame=0, timestamp_s=1.0)
    assert obs is not None
    assert set(obs.class_attrs.keys()) == {"color"}
    assert obs.instance_attrs == {}
    assert obs.eval_truth_id == "7"
    assert np.isclose(np.linalg.norm(obs.embedding), 1.0, atol=1e-5)


def test_plate_ocr_off_forces_no_read_even_if_mocked_ocr_would_return_one(tmp_path, embedder):
    dirs = make_scenario_dir(tmp_path)
    stub = _StubPlateReader(PlateRead("ABC1234", 0.8, SOURCE_MODEL))
    perceptor = RealPerceptor(dirs, embedder=embedder, plate_reader=stub,
                              pipeline_state=_State(enable_plate_ocr=False))
    obs = perceptor.process("c001", vehicle_id=7, frame=0, timestamp_s=1.0)
    assert obs.plate is None
    assert stub.calls == 0


def test_plate_ocr_on_flows_partial_read_through_unmodified(tmp_path, embedder):
    dirs = make_scenario_dir(tmp_path)
    stub = _StubPlateReader(PlateRead("AB__1234", 0.65, SOURCE_MODEL))
    perceptor = RealPerceptor(dirs, embedder=embedder, plate_reader=stub,
                              pipeline_state=_State(enable_plate_ocr=True))
    obs = perceptor.process("c001", vehicle_id=7, frame=0, timestamp_s=1.0)
    assert obs.plate.text == "AB__1234"
    assert obs.plate.confidence == pytest.approx(0.65)
    assert stub.calls == 1


def test_missing_camera_or_frame_returns_none(tmp_path, embedder):
    dirs = make_scenario_dir(tmp_path)
    perceptor = RealPerceptor(dirs, embedder=embedder, plate_reader=_StubPlateReader(None))
    assert perceptor.process("c999", vehicle_id=7, frame=0, timestamp_s=1.0) is None


def test_midpoint_annotation_hole_uses_nearest_boxed_frame(tmp_path, embedder):
    # GT tracks can be un-annotated exactly at a passage's midpoint frame
    # (occlusion). The perceptor must fall back to the nearest annotated
    # frame instead of silently dropping the whole passage.
    cam_dir = tmp_path / "c001"
    (cam_dir / "gt").mkdir(parents=True)
    (cam_dir / "gt" / "gt.txt").write_text(
        "\n".join(f"{f},7,5,5,20,15,1,-1,-1,-1" for f in (0, 1, 2, 8, 9)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(cam_dir / "vdo.avi"), fourcc, 10.0, (FRAME_W, FRAME_H))
    for i in range(10):
        writer.write(np.full((FRAME_H, FRAME_W, 3), 60 + i * 15, dtype=np.uint8))
    writer.release()
    perceptor = RealPerceptor({"c001": cam_dir}, embedder=embedder,
                              plate_reader=_StubPlateReader(None))
    # Frame 5 has no box; frames 2 and 8 are equidistant-ish (2 is 3 away,
    # 8 is 3 away -- the search finds 2 first, scanning -off before +off).
    obs = perceptor.process("c001", vehicle_id=7, frame=5, timestamp_s=1.0)
    assert obs is not None
    assert obs.event_id == "cf-c001-2-7"    # nearest annotated frame, not 5
    # A vehicle with no boxes anywhere nearby still returns None.
    assert perceptor.process("c001", vehicle_id=42, frame=5, timestamp_s=1.0) is None


def test_clip_collapses_to_the_only_boxed_frame(tmp_path, embedder):
    # Multi-frame video but a box at frame 0 only: the clip window skips the
    # boxless frames and collapses to that one real crop -- honestly shorter,
    # still a real ndarray. (Uses a multi-frame video so cv2 can seek.)
    cam_dir = tmp_path / "c001"
    (cam_dir / "gt").mkdir(parents=True)
    (cam_dir / "gt" / "gt.txt").write_text("0,7,5,5,20,15,1,-1,-1,-1\n")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(cam_dir / "vdo.avi"), fourcc, 10.0, (FRAME_W, FRAME_H))
    for i in range(6):
        writer.write(np.full((FRAME_H, FRAME_W, 3), 40 + i * 20, dtype=np.uint8))
    writer.release()
    perceptor = RealPerceptor({"c001": cam_dir}, embedder=embedder,
                              plate_reader=_StubPlateReader(None))
    obs = perceptor.process("c001", vehicle_id=7, frame=0, timestamp_s=1.0)
    assert len(obs.clip_frames) == 1
    assert all(isinstance(f, np.ndarray) and f.size for f in obs.clip_frames)


def test_clip_samples_multiple_frames_across_the_passage(tmp_path, embedder):
    cam_dir = tmp_path / "c001"
    (cam_dir / "gt").mkdir(parents=True)
    # Boxes at frames 0,6,12,18 — the clip window around center=18 samples
    # these (offsets 0/6/12/18; 24/30 have no box and are skipped).
    (cam_dir / "gt" / "gt.txt").write_text(
        "\n".join(f"{f},7,5,5,20,15,1,-1,-1,-1" for f in (0, 6, 12, 18)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(cam_dir / "vdo.avi"), fourcc, 10.0, (FRAME_W, FRAME_H))
    for i in range(20):
        writer.write(np.full((FRAME_H, FRAME_W, 3), 40 + i * 8, dtype=np.uint8))
    writer.release()
    perceptor = RealPerceptor({"c001": cam_dir}, embedder=embedder,
                              plate_reader=_StubPlateReader(None))
    obs = perceptor.process("c001", vehicle_id=7, frame=18, timestamp_s=2.0)
    assert len(obs.clip_frames) == 4
    assert perceptor.process("c001", vehicle_id=999, frame=0, timestamp_s=1.0) is None
