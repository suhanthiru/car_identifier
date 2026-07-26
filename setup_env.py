"""Environment bootstrap — stdlib only, so it imports before anything is installed.

`start.py` calls this when it finds the environment incomplete. It is also
runnable on its own:

    python setup_env.py            # install whatever is missing
    python setup_env.py --check    # report only, change nothing

Three things need setting up and only one of them is a plain `pip install`:

1. **torch** — the default PyPI wheel is CPU-only on Windows/Linux. If this
   machine has an NVIDIA GPU we install from PyTorch's CUDA index instead,
   *before* requirements.txt, so that ultralytics finds torch already
   satisfied and doesn't drag the CPU build in behind our back.
2. **requirements.txt** — ordinary.
3. **cargen** — a separate local repo installed editable; its path differs
   per machine, so we look for it rather than hard-coding one.

Nothing here is required to run the synthetic demo on a CPU-only box: every
step degrades to "skipped, here's why" rather than failing the run.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Bumping this is how you move to a newer CUDA wheel; EYES_TORCH_INDEX wins.
DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

# Imported by start.py before the server exists. Deliberately short: these are
# the modules whose absence stops the process from booting at all, not the
# whole dependency set (pip is the authority on that).
CORE_MODULES = ("httpx", "uvicorn", "fastapi", "sqlmodel", "numpy")

# Enables cargen's real 3D prior instead of the procedural stub — this is what
# start.py's _detect_cargen probes for. Deliberately NOT including rembg
# (cargen's background remover): it requires numpy>=2.3 and would silently
# break this repo's pinned numpy==1.26.4. Install it into a separate env if you
# need it; see cargen's own docs/SETUP.md.
CARGEN_EXTRAS = ("trimesh", "pillow")

# Pins that another package's resolver likes to walk over. Checked after every
# install so a conflict surfaces here rather than as a numpy ABI error later.
CRITICAL_PINS = {"numpy": "1.26.4"}


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def missing_core() -> list[str]:
    return [m for m in CORE_MODULES if not _have(m)]


def gpu_present() -> bool:
    """An NVIDIA GPU on the machine — says nothing about torch's build."""
    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True,
                              timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def torch_status() -> tuple[bool, bool, str]:
    """(installed, cuda_usable, detail). Importing torch is slow but it is the
    only honest answer — a CUDA build on a driverless box still reports False."""
    if not _have("torch"):
        return False, False, "not installed"
    try:
        import torch
    except Exception as exc:                          # noqa: BLE001
        return False, False, f"present but unimportable ({type(exc).__name__})"
    if torch.cuda.is_available():
        return True, True, f"{torch.__version__} on {torch.cuda.get_device_name(0)}"
    return True, False, f"{torch.__version__} (CPU-only build)"


def find_cargen() -> Path | None:
    """Locate the cargen repo. Its checkout path is per-machine, so search
    instead of hard-coding: env var, then siblings of this repo, then one
    level deeper (people nest it inside a projects folder)."""
    override = os.environ.get("EYES_CARGEN_PATH")
    if override:
        cand = Path(override)
        return cand if (cand / "cargen" / "__init__.py").is_file() else None
    if _have("cargen"):                                # already installed
        return None
    parent = ROOT.parent
    for pattern in ("*/cargen/__init__.py", "*/*/cargen/__init__.py"):
        for hit in sorted(parent.glob(pattern)):
            repo = hit.parent.parent
            if (repo / "pyproject.toml").is_file():
                return repo
    return None


def _pip(*args: str, dry_run: bool = False) -> bool:
    """Always sys.executable -m pip: addressing the interpreter directly is
    what keeps an unactivated shell from installing into the system Python."""
    cmd = [sys.executable, "-m", "pip", "install", *args]
    print(f"\n  $ {' '.join(cmd)}\n")
    if dry_run:
        return True
    return subprocess.run(cmd).returncode == 0


def report() -> dict[str, object]:
    installed, cuda, detail = torch_status()
    cargen_repo = find_cargen()
    return {
        "missing_core": missing_core(),
        "torch_installed": installed,
        "torch_cuda": cuda,
        "torch_detail": detail,
        "gpu": gpu_present(),
        "cargen_installed": _have("cargen"),
        "cargen_repo": cargen_repo,
        "cargen_real_backend": _have("trimesh"),
    }


def needs_setup() -> bool:
    st = report()
    return bool(st["missing_core"]) or (
        bool(st["gpu"]) and st["torch_installed"] and not st["torch_cuda"])


def print_report(st: dict[str, object] | None = None) -> None:
    st = st or report()
    gpu = "yes" if st["gpu"] else "no NVIDIA GPU detected"
    gone = st["missing_core"]
    core = "missing " + ", ".join(gone) if gone else "present"    # type: ignore[arg-type]
    print(f"  python      : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  core deps   : {core}")
    print(f"  nvidia-smi  : {gpu}")
    print(f"  torch       : {st['torch_detail']}")
    if st["cargen_installed"]:
        # trimesh only means mesh tooling is importable — it is not a
        # reconstructor, and claiming a "real prior backend" on that basis is
        # how the launcher used to overstate the 3D panel. `start.py --check`
        # probes the generative backends themselves; say only what is known
        # here.
        extras = "with trimesh" if st["cargen_real_backend"] else "no trimesh"
        print(f"  cargen      : installed ({extras})")
    elif st["cargen_repo"]:
        print(f"  cargen      : not installed, repo found at {st['cargen_repo']}")
    else:
        print("  cargen      : not installed, no local repo found (3D panel off)")


def bootstrap(*, want_cuda: bool = True, want_cargen: bool = True,
              dry_run: bool = False) -> bool:
    """Install what's missing. Returns False if a required step failed."""
    st = report()
    ok = True

    # 1. torch first, so requirements.txt doesn't resolve a CPU build for us.
    index = os.environ.get("EYES_TORCH_INDEX", DEFAULT_TORCH_INDEX)
    if want_cuda and st["gpu"] and not st["torch_cuda"]:
        if st["torch_installed"]:
            print("\n  torch is a CPU-only build but this machine has a GPU;"
                  "\n  replacing it with the CUDA wheel.")
            extra = ["--force-reinstall"]
        else:
            extra = []
        if not _pip("torch", "torchvision", "--index-url", index, *extra,
                    dry_run=dry_run):
            print(f"\n  ! CUDA wheels failed from {index}."
                  f"\n    The tag moves between releases — check pytorch.org for the"
                  f"\n    current one and re-run with:"
                  f"\n      $env:EYES_TORCH_INDEX = 'https://download.pytorch.org/whl/cuXXX'"
                  f"\n    Continuing; the CPU build still runs everything, slower.")
    elif want_cuda and not st["gpu"]:
        print("\n  no NVIDIA GPU — skipping the CUDA wheel, CPU torch is correct here.")

    # 2. everything else.
    if not _pip("-r", str(ROOT / "requirements.txt"), dry_run=dry_run):
        print("\n  ! requirements.txt failed to install — see the pip output above.")
        ok = False

    # 3. cargen: optional by design, never fails the run.
    if want_cargen and not st["cargen_installed"]:
        repo = st["cargen_repo"]
        if repo:
            if _pip("-e", str(repo), dry_run=dry_run):
                _pip(*CARGEN_EXTRAS, dry_run=dry_run)
            else:
                print("\n  ! cargen install failed; the 3D panel stays off.")
        else:
            print("\n  cargen repo not found nearby — the 3D panel stays off."
                  "\n  If you have it, point at it and re-run:"
                  "\n    $env:EYES_CARGEN_PATH = 'D:\\path\\to\\car_gen_and_modeling'")
    elif want_cargen and st["cargen_installed"] and not st["cargen_real_backend"]:
        _pip(*CARGEN_EXTRAS, dry_run=dry_run)

    if not dry_run:
        ok = _restore_pins() and ok
    return ok


def _restore_pins() -> bool:
    """Put back any pinned version a later install clobbered.

    pip resolves each invocation independently, so a package installed in step
    3 can happily upgrade something step 2 pinned. numpy is the one that bites:
    the upgrade succeeds and the failure shows up much later as an ABI error
    from a C extension built against the other major version.
    """
    ok = True
    for name, want in CRITICAL_PINS.items():
        try:
            from importlib.metadata import version
            have = version(name)
        except Exception:                             # noqa: BLE001
            continue
        if have != want:
            print(f"\n  {name} {have} was installed over the pinned {want}"
                  f" — restoring it.")
            ok = _pip(f"{name}=={want}") and ok
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report only, install nothing")
    ap.add_argument("--dry-run", action="store_true", help="print the pip commands only")
    ap.add_argument("--no-cuda", action="store_true", help="force the CPU torch build")
    ap.add_argument("--no-cargen", action="store_true", help="skip the 3D bridge")
    args = ap.parse_args()

    print("=" * 66)
    print("  EYES EVERYWHERE - environment")
    print("=" * 66)
    print_report()
    print("=" * 66)
    if args.check:
        return 0

    ok = bootstrap(want_cuda=not args.no_cuda, want_cargen=not args.no_cargen,
                   dry_run=args.dry_run)
    if not args.dry_run:
        print("\n" + "=" * 66)
        print("  after setup")
        print("=" * 66)
        # Re-exec the report in a fresh interpreter: this process cached
        # import state from before the install and would lie about torch.
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--check"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
