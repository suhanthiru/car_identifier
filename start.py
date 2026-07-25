"""One command to run everything. Start here.

    python start.py

Looks at what is actually available on this machine and runs the best
demo it can, rather than failing on a missing prerequisite:

- CityFlow present  -> real-data console (real footage, real cameras)
- CityFlow absent   -> synthetic world (always works, no downloads)
- cargen present    -> 3D reconstruction panel enabled
- cargen absent     -> everything else still runs

Nothing here needs a GPU. Override any of it with the flags below; run
`python start.py --help` to see them.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import threading
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx
import uvicorn

BANNER = "=" * 66


def _detect_cityflow() -> tuple[bool, str]:
    """(available, human-readable reason)."""
    try:
        from datasets.cityflow import CityFlow
        from datasets.config import cityflow_root
    except Exception as exc:                      # noqa: BLE001 - report, never crash
        return False, f"loader unavailable ({exc})"
    root = cityflow_root()
    if not CityFlow.exists(root):
        return False, f"not found at {root}"
    return True, str(root)


def _detect_cargen() -> tuple[bool, str]:
    try:
        import cargen  # noqa: F401
    except Exception as exc:                      # noqa: BLE001
        return False, f"not installed ({type(exc).__name__})"
    # Present, but the heavy generative backends are a separate install and
    # need CUDA; say so plainly rather than implying a real reconstruction.
    try:
        import trimesh  # noqa: F401
        return True, "installed, real 3D prior backend available"
    except ImportError:
        return True, "installed (stub prior: procedural shape, provenance is real)"


def _device_note() -> str:
    """torch is installed as a CPU-only build in some environments, so
    report what will actually be used rather than what is hoped for."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return "torch unavailable"
    if torch.cuda.is_available():
        return f"GPU - {torch.cuda.get_device_name(0)}"
    return f"CPU (torch {torch.__version__}; a CUDA build would use the GPU)"


def _print_plan(mode: str, cityflow: tuple[bool, str], cargen: tuple[bool, str],
                port: int, time_scale: float) -> None:
    print(BANNER)
    print("  EYES EVERYWHERE")
    print(BANNER)
    print(f"  mode        : {mode}")
    print(f"  CityFlow    : {'yes - ' if cityflow[0] else 'no  - '}{cityflow[1]}")
    print(f"  3D (cargen) : {'yes - ' if cargen[0] else 'no  - '}{cargen[1]}")
    print(f"  compute     : {_device_note()}")
    print(f"  speed       : {time_scale}x real time")
    print(f"  console     : http://127.0.0.1:{port}")
    print(f"  inspector   : http://127.0.0.1:{port}/inspector.html")
    print(BANNER)


def _wait_for_server(base: str, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/api/cameras", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError as exc:
            last = exc
            time.sleep(0.3)
    raise RuntimeError(f"server did not come up within {timeout_s:.0f}s ({last})")


def _reset(db_path: Path, crops_dir: Path) -> Path:
    """Clear the previous run so each launch starts clean. Falls back to a
    timestamped DB if another process still holds the file open."""
    if db_path.exists():
        try:
            db_path.unlink()
        except PermissionError:
            db_path = db_path.with_name(f"{db_path.stem}-{int(time.time())}.sqlite")
            print(f"  note: previous database still in use; using {db_path.name}")
    if crops_dir.exists():
        shutil.rmtree(crops_dir, ignore_errors=True)
    return db_path


def run_synthetic(port: int, time_scale: float, open_browser: bool) -> None:
    from demo import flag_demo_targets
    from server.api import create_app
    from server.feed import FeedConfig, run_feed
    from sim.emitter import build_default_world

    db_path = _reset(Path("data/eyes.sqlite"), Path("data/crops"))
    app = create_app(db_url=f"sqlite:///{db_path.as_posix()}")
    server = _serve(app, port)
    base = f"http://127.0.0.1:{port}"
    _wait_for_server(base)
    flag_demo_targets(base)
    if open_browser:
        webbrowser.open(base)
    print(f"\n  replaying the synthetic world at {time_scale}x - watch the console\n")

    counts = asyncio.run(run_feed(build_default_world(),
                                  FeedConfig(base_url=base, time_scale=time_scale)))
    _idle(server, sum(counts.values()), len(counts))


def run_cityflow(port: int, time_scale: float, open_browser: bool, scenario: str,
                 root: str) -> None:
    from datasets.cityflow import CityFlow
    from datasets.cityflow_video import discover_camera_dirs
    from server.api import create_app
    from server.real_feed import CityFlowFeedConfig, run_cityflow_feed

    scen = CityFlow(Path(root)).load_scenario(scenario)
    graph = scen.to_road_graph()
    camera_dirs = discover_camera_dirs(Path(root), scenario)
    print(f"  loaded {scenario}: {len(graph.cameras)} real cameras, "
          f"{len(scen.spans)} ground-truth tracks, "
          f"{len(graph.edges)} observed transit routes")

    # Calibration is per-deployment AND per-embedder; if the artifact for this
    # scenario is missing the cascade falls back to an uncalibrated curve and
    # says so, rather than silently borrowing another deployment's numbers.
    calibration = Path(f"calibration/artifacts/cityflow_{scenario.lower()}.json")
    if not calibration.is_file():
        print(f"  note: no calibration for {scenario}; using the uncalibrated "
              f"fallback (run scripts/calibrate_cityflow.py to fit one)")

    db_path = _reset(Path("data/eyes-cityflow.sqlite"), Path("data/crops-cityflow"))
    app = create_app(
        graph=graph, db_url=f"sqlite:///{db_path.as_posix()}",
        crops_dir="data/crops-cityflow", world_source="real",
        calibration_path=str(calibration) if calibration.is_file() else "",
        cityflow_root=root, cityflow_scenario_name=scenario)
    server = _serve(app, port)
    base = f"http://127.0.0.1:{port}"
    _wait_for_server(base)
    if open_browser:
        webbrowser.open(base)
    print(f"\n  replaying {scenario} at {time_scale}x - browse the thumbnails "
          f"and click any car to track it\n")

    counts = asyncio.run(run_cityflow_feed(
        scen, camera_dirs, scen.camera_gps(), app.state,
        CityFlowFeedConfig(base_url=base, time_scale=time_scale)))
    _idle(server, sum(counts.values()), len(counts))


def _serve(app, port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    return server


def _idle(server: uvicorn.Server, sightings: int, cameras: int) -> None:
    print(f"\n  replay complete: {sightings} sightings across {cameras} cameras.")
    print("  the console stays live - keep working the review queue.")
    print("  ctrl+c to quit.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.should_exit = True
        print("  bye")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["auto", "synthetic", "cityflow"],
                        default="auto", help="default: auto-detect")
    parser.add_argument("--scenario", default="S01", help="CityFlow scenario")
    parser.add_argument("--time-scale", type=float, default=None,
                        help="seconds of footage per wall second "
                             "(default 8 synthetic, 4 CityFlow)")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-3d", action="store_true",
                        help="skip the 3D panel even if cargen is installed")
    args = parser.parse_args()

    cityflow = _detect_cityflow()
    cargen = _detect_cargen()

    mode = args.mode
    if mode == "auto":
        mode = "cityflow" if cityflow[0] else "synthetic"
    if mode == "cityflow" and not cityflow[0]:
        raise SystemExit(
            f"CityFlow requested but unavailable: {cityflow[1]}.\n"
            f"See DATASETS.md, or just run `python start.py --mode synthetic`.")

    # 3D is opt-out rather than opt-in here: the whole point of one command is
    # that you see everything the machine can actually do.
    enable_3d = cargen[0] and not args.no_3d
    os.environ["EYES_ENABLE_3D"] = "1" if enable_3d else "0"
    if not enable_3d and cargen[0]:
        cargen = (True, "installed, disabled via --no-3d")

    time_scale = args.time_scale
    if time_scale is None:
        time_scale = 4.0 if mode == "cityflow" else 8.0

    _print_plan("real CityFlow footage" if mode == "cityflow"
                else "synthetic world (no datasets needed)",
                cityflow, cargen, args.port, time_scale)

    if mode == "cityflow":
        run_cityflow(args.port, time_scale, not args.no_browser,
                     args.scenario, cityflow[1])
    else:
        run_synthetic(args.port, time_scale, not args.no_browser)


if __name__ == "__main__":
    main()
