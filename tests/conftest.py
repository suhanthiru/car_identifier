"""Test-wide defaults.

The fast suite must stay fast and deterministic. cargen picks its backends
from the environment (`CARGEN_PRIOR_BACKEND`, `CARGEN_SEGMENTER`), defaulting
to the real ones — SF3D and rembg — so on a machine where those are actually
installed the 3D tests silently start loading ~2 GB of weights and spending
~90s of GPU per fusion. `pytest -m "not slow"` went from 65s to minutes the
moment the real backend was installed, which punishes exactly the developer
who set the project up completely.

Pin both to stubs here. The tests assert the bridge's own behaviour — gating,
snapshots, provenance, geometry attributes — none of which depends on the
prior's visual quality, and the `slow` marker already exists for anything that
genuinely needs real weights.
"""
import os
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolate_derived_cache():
    """Keep the suite out of the real `data/cache`.

    The browse-index cache path is relative to the CWD, so tests running from
    the repo root wrote fixture indexes next to — and, before the path carried
    the dataset root, directly ON TOP OF — the real ones. The real S01 index is
    31 MB and takes a full rebuild to replace, and that rebuild is paid at the
    moment an operator switches scenario.
    """
    previous = os.environ.get("EYES_CACHE_DIR")
    with tempfile.TemporaryDirectory(prefix="eyes-test-cache-") as tmp:
        os.environ["EYES_CACHE_DIR"] = tmp
        yield
    if previous is None:
        os.environ.pop("EYES_CACHE_DIR", None)
    else:
        os.environ["EYES_CACHE_DIR"] = previous


@pytest.fixture(scope="session", autouse=True)
def _stub_cargen_backends():
    """Force cargen's cheap backends for the whole session.

    Set rather than defaulted: a developer with SF3D installed and no env vars
    would otherwise get the real ones, and the point is that the suite behaves
    identically on both machines.
    """
    previous = {k: os.environ.get(k)
                for k in ("CARGEN_PRIOR_BACKEND", "CARGEN_SEGMENTER")}
    os.environ["CARGEN_PRIOR_BACKEND"] = "stub"
    os.environ["CARGEN_SEGMENTER"] = "stub"
    yield
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
