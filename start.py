"""One command to run everything. Start here.

    python start.py

On a fresh checkout this installs what it needs first (CUDA torch if the
machine has an NVIDIA GPU, the requirements, the cargen 3D bridge if it can
find it) and then runs. See setup_env.py for the details, or run
`python start.py --check` to see what it would do without doing it.

Once the environment is there, it looks at what is actually available on
this machine and runs the best demo it can, rather than failing on a
missing prerequisite:

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
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Only stdlib and setup_env (also stdlib-only) at module scope: on a fresh
# checkout the third-party imports below would be the first thing to fail,
# and a traceback is a worse answer than an offer to install them.
import setup_env

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
    """(available, what the 3D panel will actually produce).

    `import trimesh` used to stand in for "real backend available", which was
    wrong in the direction that matters: trimesh is a mesh library, not a
    reconstructor, so the banner claimed real geometry while both generative
    backends were quietly falling back to stubs. Probe the two backends cargen
    would really build — cheap, since each fails on a missing import or a
    missing checkout rather than by loading weights.
    """
    try:
        import cargen  # noqa: F401
    except Exception as exc:                      # noqa: BLE001
        return False, f"not installed ({type(exc).__name__})"

    from cargen import backends

    def _probe(build) -> bool:
        try:
            build()
            return True
        except Exception:                         # noqa: BLE001 — absent == stub
            return False

    prior = _probe(backends.build_prior_generator)
    seg = _probe(backends.build_segmenter)
    if prior and seg:
        return True, "installed, real prior + segmenter"
    if prior or seg:
        got = "prior" if prior else "segmenter"
        return True, f"installed, real {got} only (the other falls back to a stub)"
    return True, ("installed, STUB geometry only — procedural sedan, not a "
                  "reconstruction (provenance and fusion are real)")


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
    import httpx

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


def _reset(db_path: Path, crops_dir: Path, targets3d_dir: Path | None = None) -> Path:
    """Clear the previous run so each launch starts clean. Falls back to a
    timestamped DB if another process still holds the file open.

    The 3D models must go too. Target ids restart at `tgt-001` every launch,
    so a surviving `data/targets3d/tgt-001` is silently adopted by whatever
    vehicle happens to be flagged first next time — the new run fuses into the
    old run's asset and reports its splat count, which is how a session with a
    real SF3D backend produced a 20k-splat cloud built by a previous session's
    stub. Same-name reuse across runs is contamination, not caching.
    """
    if db_path.exists():
        try:
            db_path.unlink()
        except PermissionError:
            db_path = db_path.with_name(f"{db_path.stem}-{int(time.time())}.sqlite")
            print(f"  note: previous database still in use; using {db_path.name}")
    for stale in (crops_dir, targets3d_dir):
        if stale is not None and stale.exists():
            shutil.rmtree(stale, ignore_errors=True)
    return db_path


def run_synthetic(port: int, time_scale: float, open_browser: bool) -> None:
    from demo import flag_demo_targets
    from server.api import create_app
    from server.feed import FeedConfig, run_feed
    from sim.emitter import build_default_world

    db_path = _reset(Path("data/eyes.sqlite"), Path("data/crops"),
                     Path("data/targets3d"))
    app = create_app(db_url=f"sqlite:///{db_path.as_posix()}")
    server = _serve(app, port)
    base = f"http://127.0.0.1:{port}"
    _wait_for_server(base)
    flag_demo_targets(base)
    if open_browser:
        webbrowser.open(base)
    print(f"\n  replaying the synthetic world at {time_scale}x - watch the console\n")

    counts = asyncio.run(run_feed(
        build_default_world(),
        FeedConfig(base_url=base, time_scale=time_scale),
        pipeline_state=app.state))
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

    # Cameras in a scenario start recording at different wall-clock times.
    # Without the offsets every cross-camera elapsed time is wrong in a way
    # that still looks reasonable, so say which case this run is in.
    from datasets.cityflow import timing_offsets_found
    if not timing_offsets_found(Path(root), scenario):
        print(f"  WARNING: no camera timing offsets for {scenario} "
              f"(expected {root}\\cam_timestamp\\{scenario}.txt).\n"
              f"           Cameras do not share a clock; transit times will be "
              f"wrong without them.")

    # Calibration is per-deployment AND per-embedder; if the artifact for this
    # scenario is missing the cascade falls back to an uncalibrated curve and
    # says so, rather than silently borrowing another deployment's numbers.
    calibration = Path(f"calibration/artifacts/cityflow_{scenario.lower()}.json")
    if not calibration.is_file():
        print(f"  note: no calibration for {scenario}; using the uncalibrated "
              f"fallback (run scripts/calibrate_cityflow.py to fit one)")

    # Separate 3D directory per mode, for the same reason crops are separate:
    # a synthetic sprite's model and a real vehicle's model must never share
    # a target id's directory.
    db_path = _reset(Path("data/eyes-cityflow.sqlite"), Path("data/crops-cityflow"),
                     Path("data/targets3d-cityflow"))
    app = create_app(
        graph=graph, db_url=f"sqlite:///{db_path.as_posix()}",
        crops_dir="data/crops-cityflow", world_source="real",
        targets3d_dir="data/targets3d-cityflow",
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


def _serve(app, port: int) -> "uvicorn.Server":
    import uvicorn

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


def _ensure_environment(*, force: bool, refuse: bool) -> bool:
    """Install missing dependencies before the demo needs them.

    Returns False when the caller should stop. Installing several GB without
    asking would be rude, so an interactive run confirms first; a
    non-interactive one (CI, a pipe) refuses and prints the command instead
    of silently blocking on a prompt nobody can answer.
    """
    if refuse or not (force or setup_env.needs_setup()):
        missing = setup_env.missing_core()
        if missing and refuse:
            print(f"missing: {', '.join(missing)}. Drop --no-setup, or:\n"
                  f"  {Path(sys.executable).name} -m pip install -r requirements.txt")
            return False
        return True

    print(BANNER)
    print("  EYES EVERYWHERE - first-run setup")
    print(BANNER)
    setup_env.print_report()
    print(BANNER)
    print("\n  Setting this up means a multi-GB download (torch and friends).")

    if not sys.stdin.isatty():
        print(f"  Not an interactive terminal, so not starting it unasked. Run:\n"
              f"    python setup_env.py\n")
        return False
    try:
        if input("  Install now? [Y/n] ").strip().lower() in ("n", "no"):
            print("  Skipped. `python setup_env.py` when you're ready.")
            return False
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return False

    if not setup_env.bootstrap():
        print("\n  Setup did not complete; see the pip output above.")
        return False

    # The freshly-installed packages are invisible to this process — it has
    # already resolved (and in torch's case cached) the old import state. Hand
    # off to a new interpreter rather than importing into a stale one.
    print("\n  Setup complete — restarting.\n")
    argv = [a for a in sys.argv[1:] if a not in ("--setup",)]
    raise SystemExit(subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                     *argv]).returncode)


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
    parser.add_argument("--3d-identification", dest="three_d_identification",
                        action="store_true",
                        help="let 3D geometry vote on identity (OFF by default: "
                             "the measured ablation found it vetoes correct "
                             "matches — see RESULTS.md)")
    parser.add_argument("--check", action="store_true",
                        help="report the environment and exit, installing nothing")
    parser.add_argument("--setup", action="store_true",
                        help="run the install step even if nothing looks missing")
    parser.add_argument("--no-setup", action="store_true",
                        help="never install; fail instead if something is missing")
    args = parser.parse_args()

    if args.check:
        print(BANNER)
        print("  EYES EVERYWHERE - environment")
        print(BANNER)
        setup_env.print_report()
        cargen = _detect_cargen()
        print(f"  3D backend  : {cargen[1]}")
        cityflow = _detect_cityflow()
        print(f"  CityFlow    : {'yes - ' if cityflow[0] else 'no  - '}{cityflow[1]}")
        print(BANNER)
        return
    if not _ensure_environment(force=args.setup, refuse=args.no_setup):
        return

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
    # Visual by default, evidential only on request — the ablation measured the
    # geometry channel removing correct matches and no incorrect ones.
    os.environ["EYES_ENABLE_3D_IDENTIFICATION"] = (
        "1" if (enable_3d and args.three_d_identification) else "0")
    if not enable_3d and cargen[0]:
        cargen = (True, "installed, disabled via --no-3d")
    elif enable_3d:
        cargen = (cargen[0], cargen[1] + (
            "; feeding identification (--3d-identification)"
            if args.three_d_identification else "; visual only"))

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
