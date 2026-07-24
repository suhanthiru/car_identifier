"""One-command real-data live console: any vehicle in a CityFlow scenario,
followed by the real reasoning pipeline.

    python scripts/run_cityflow_console.py [--scenario S01] [--time-scale 8]
        [--port 8010] [--no-browser] [--no-plate-ocr] [--keep-db]

Loads the given scenario's REAL road graph (real camera GPS-anchored
positions, real observed transit windows -- see datasets/cityflow.py),
starts the server in "real" world-source mode, and replays every vehicle
in the scenario through server/real_feed.py against real video crops.
Unlike demo.py, nothing is pre-flagged: open the printed URL, browse the
real vehicle thumbnails, and click any car to start tracking it.
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import threading
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import uvicorn

from datasets.cityflow import CityFlow
from datasets.cityflow_video import discover_camera_dirs
from datasets.config import cityflow_root
from server.real_feed import CityFlowFeedConfig, run_cityflow_feed

DB_PATH = Path("data/eyes-cityflow.sqlite")
CROPS_DIR = Path("data/crops-cityflow")


def reset_demo_data() -> Path:
    db_path = DB_PATH
    if DB_PATH.exists():
        try:
            DB_PATH.unlink()
        except PermissionError:
            db_path = DB_PATH.with_name(f"eyes-cityflow-{int(time.time())}.sqlite")
            print(f"note: {DB_PATH} is in use elsewhere; using {db_path}")
    if CROPS_DIR.exists():
        shutil.rmtree(CROPS_DIR, ignore_errors=True)
    return db_path


def start_server(
    port: int, db_path: Path, root: Path, scenario_name: str, enable_plate_ocr: bool,
    graph,
) -> tuple[uvicorn.Server, object]:
    from server.api import create_app

    # Calibration is per-deployment: the default artifact was fitted on
    # SYNTHETIC sprite pairs and maps real same-car similarities (~0.8) to
    # p~0, and the VeRi curve answers a different dataset's question. Use
    # this scenario's own fitted curve when it exists (see
    # scripts/calibrate_cityflow.py); otherwise the honest fallback -- every
    # fact then cites "uncalibrated-linear" instead of silently transferring
    # a foreign curve.
    calibration = Path(f"calibration/artifacts/cityflow_{scenario_name.lower()}.json")
    calibration_path = str(calibration) if calibration.is_file() else ""
    fallback_note = ("uncalibrated-linear fallback "
                     "(run scripts/calibrate_cityflow.py to fit this scenario)")
    print(f"reid calibration: {calibration_path or fallback_note}")

    app = create_app(
        graph=graph, db_url=f"sqlite:///{db_path.as_posix()}",
        crops_dir=str(CROPS_DIR), world_source="real",
        calibration_path=calibration_path,
        enable_plate_ocr=enable_plate_ocr,
        cityflow_root=str(root), cityflow_scenario_name=scenario_name)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, app.state


def wait_for_server(base: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/api/cameras", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.3)
    raise RuntimeError("server did not come up in time")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="S01")
    parser.add_argument("--time-scale", type=float, default=8.0,
                        help="real seconds of footage per wall second (default 8)")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-plate-ocr", action="store_true",
                        help="start with plate OCR off (still toggleable live "
                             "via POST /api/pipeline_config)")
    parser.add_argument("--keep-db", action="store_true")
    args = parser.parse_args()

    root = cityflow_root()
    if not CityFlow.exists(root):
        raise SystemExit(
            f"CityFlow not found at {root}. See DATASETS.md for how to get it.")
    cf = CityFlow(root)
    if args.scenario not in cf.scenario_names():
        raise SystemExit(
            f"scenario {args.scenario!r} not found under {root}. "
            f"Available: {', '.join(cf.scenario_names())}")
    scenario = cf.load_scenario(args.scenario)
    graph = scenario.to_road_graph()
    camera_dirs = discover_camera_dirs(root, args.scenario)
    print(f"loaded {args.scenario}: {len(graph.cameras)} cameras, "
          f"{len(scenario.spans)} real ground-truth tracks, "
          f"{len(graph.edges)} real observed transit edges")

    db_path = reset_demo_data() if not args.keep_db else DB_PATH

    base = f"http://127.0.0.1:{args.port}"
    server, app_state = start_server(args.port, db_path, root, args.scenario,
                                     enable_plate_ocr=not args.no_plate_ocr, graph=graph)
    wait_for_server(base)
    print(f"server up at {base} (world_source=real, plate_ocr="
          f"{'on' if not args.no_plate_ocr else 'off'})")
    if not args.no_browser:
        webbrowser.open(base)
    print(f"replaying every vehicle in {args.scenario} at {args.time_scale}x "
          f"real-time -- browse and flag any car in the console...")

    # The feed runs in this same process, so RealPerceptor can read the
    # live app.state.enable_plate_ocr directly -- POST /api/pipeline_config
    # mutates that same object, so a toggle takes effect on the very next
    # sighting, no polling needed.
    counts = asyncio.run(run_cityflow_feed(
        scenario, camera_dirs, scenario.camera_gps(), app_state,
        CityFlowFeedConfig(base_url=base, time_scale=args.time_scale)))
    print(f"feed complete: {sum(counts.values())} sightings across "
          f"{len(counts)} cameras")
    print("console stays live for review-queue work — ctrl+c to quit")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.should_exit = True
        print("bye")


if __name__ == "__main__":
    main()
