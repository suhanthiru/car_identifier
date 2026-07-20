"""Loader tests over tiny fake dataset trees (no real data required —
the real datasets are presence-gated and manually downloaded)."""
import numpy as np
import pytest

from datasets.cityflow import CityFlow, parse_homography
from datasets.vehicleid import VehicleID
from datasets.veri776 import Veri776, parse_name

VERI_XML = """<?xml version="1.0" encoding="utf-8"?>
<TrainingImages><Items>
<Item imageName="0001_c001_00016450_0.jpg" vehicleID="0001" cameraID="c001" colorID="5" typeID="1"/>
<Item imageName="0001_c002_00016460_0.jpg" vehicleID="0001" cameraID="c002" colorID="5" typeID="1"/>
<Item imageName="0002_c001_00016470_0.jpg" vehicleID="0002" cameraID="c001" colorID="5" typeID="1"/>
</Items></TrainingImages>"""


def make_veri(tmp_path):
    root = tmp_path / "VeRi"
    names = ["0001_c001_00016450_0.jpg", "0001_c002_00016460_0.jpg",
             "0002_c001_00016470_0.jpg"]
    for d in ("image_train", "image_query", "image_test"):
        (root / d).mkdir(parents=True)
    (root / "name_train.txt").write_text("")
    (root / "name_query.txt").write_text(names[0] + "\n")
    (root / "name_test.txt").write_text("\n".join(names[1:]) + "\n")
    (root / "train_label.xml").write_text(VERI_XML)
    for n in names:
        (root / "image_query" / n).write_bytes(b"")
        (root / "image_test" / n).write_bytes(b"")
    return root


def test_veri_name_parsing():
    assert parse_name("0002_c002_00030600_0.jpg") == ("0002", "c002", 30600)
    assert parse_name("garbage.jpg") is None


def test_veri_loader_and_labels(tmp_path):
    ds = Veri776.load(make_veri(tmp_path))
    assert len(ds.query) == 1 and len(ds.gallery) == 2
    q = ds.query[0]
    assert (q.vehicle_id, q.camera_id, q.color, q.body_type) == \
        ("0001", "c001", "red", "sedan")
    assert ds.identities() == {"0001", "0002"}


def test_veri_absent_raises(tmp_path):
    assert not Veri776.exists(tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="DATASETS.md"):
        Veri776.load(tmp_path / "nope")


def make_vehicleid(tmp_path):
    root = tmp_path / "VehicleID"
    (root / "image").mkdir(parents=True)
    (root / "train_test_split").mkdir()
    lines = [f"{i:07d} {vid}" for vid, count in [("10", 3), ("11", 2), ("12", 4)]
             for i in range(int(vid) * 100, int(vid) * 100 + count)]
    (root / "train_test_split" / "test_list_800.txt").write_text("\n".join(lines))
    return root


def test_vehicleid_split_protocol(tmp_path):
    ds = VehicleID(make_vehicleid(tmp_path))
    split = ds.test_split(800, seed=1)
    assert len(split.gallery) == 3, "one gallery image per identity"
    assert len(split.query) == 9 - 3
    gallery_ids = [g.vehicle_id for g in split.gallery]
    assert sorted(gallery_ids) == ["10", "11", "12"]
    # Deterministic under a fixed seed.
    again = ds.test_split(800, seed=1)
    assert [g.path for g in again.gallery] == [g.path for g in split.gallery]


def make_cityflow(tmp_path):
    root = tmp_path / "CityFlow"
    for cam, rows in {
        "c001": ["1,7,10,10,50,50,1,-1,-1,-1", "40,7,12,10,50,50,1,-1,-1,-1"],
        "c002": ["120,7,10,10,50,50,1,-1,-1,-1", "150,7,10,10,50,50,1,-1,-1,-1",
                 "10,9,10,10,50,50,1,-1,-1,-1"],
    }.items():
        d = root / "train" / "S01" / cam / "gt"
        d.mkdir(parents=True)
        (d / "gt.txt").write_text("\n".join(rows))
        (root / "train" / "S01" / cam / "calibration.txt").write_text(
            "Homography matrix: 1 0 42.5;0 1 -90.25;0 0 1")
    return root


def test_cityflow_spans_and_transitions(tmp_path):
    ds = CityFlow(make_cityflow(tmp_path))
    scen = ds.load_scenario("S01")
    assert scen.cameras == ("c001", "c002")
    trans = scen.transitions()
    hop = next(t for t in trans if t.vehicle_id == 7)
    # exit c001 at frame 40 (4.0s @10fps), enter c002 at frame 120 (12.0s)
    assert hop.from_camera == "c001" and hop.to_camera == "c002"
    assert hop.elapsed_s == pytest.approx(8.0)
    # vehicle 9 appears at one camera only: no transitions
    assert not [t for t in trans if t.vehicle_id == 9]
    # Both fixture cameras share one homography, so their (degenerate,
    # identical) local layout collapses to the scenario's real documented
    # center — camera_gps() never fabricates per-camera lat/lon from the
    # homography's raw (non-georeferenced) output. See DATASETS.md.
    gps = scen.camera_gps()
    assert gps["c001"] == pytest.approx((42.525678, -90.723601))
    assert gps["c002"] == pytest.approx((42.525678, -90.723601))


def test_cityflow_applies_camera_timing_offsets(tmp_path):
    """AIC22 cameras start at different wall-clock times; without the offset
    the c001->c002 transit time is wrong."""
    root = make_cityflow(tmp_path)
    # c002 actually starts 100s after c001 on the shared clock.
    (root / "cam_timing").mkdir()
    (root / "cam_timing" / "S01.txt").write_text("c001 0.0 10\nc002 100.0 10\n")
    scen = CityFlow(root).load_scenario("S01")
    hop = next(t for t in scen.transitions() if t.vehicle_id == 7)
    # enter c002 at 12.0s local + 100s offset - exit c001 at 4.0s = 108.0s
    assert hop.elapsed_s == pytest.approx(108.0)


def test_homography_parser_variants():
    H = parse_homography("Homography matrix: 1 2 3;4 5 6;7 8 9")
    assert H is not None and H[2, 2] == 9
    assert parse_homography("no matrix here") is None
    assert parse_homography("Homography matrix: 1 2;3 4") is None
