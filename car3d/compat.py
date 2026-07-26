"""Bridge across cargen's two branches.

cargen's `master` and its `ee-adapter` branch have diverged. `InsufficientDetail`
— the "this photo is too small to reconstruct from" refusal — exists only on
`ee-adapter` (commit 9975a1e); `master` carries newer rendering work instead and
never defines it. Importing it unconditionally, as the server used to, turns a
plain checkout of `master` into an ImportError raised inside a worker thread,
which the fusion path then swallowed: 3D silently never appeared, with nothing
in the log to say why.

So: take cargen's class when it is there, define an equivalent when it is not,
and — because the refusal is a real protection, not a formality — enforce the
size bar here whenever cargen isn't doing it. A distant vehicle upsampled to the
model's 512px input comes back a blob, and folding a blob into an asset that was
previously correct is worse than skipping the frame.
"""
from __future__ import annotations

import numpy as np

try:
    from cargen.prior_generation.interface import InsufficientDetail

    CARGEN_ENFORCES_DETAIL = True
except ImportError:                        # cargen master
    class InsufficientDetail(ValueError):
        """The image does not carry enough of the subject to reconstruct from.

        Mirrors cargen's ee-adapter class, including being a ValueError
        subclass, so callers can keep telling "bad input" apart from "broken
        backend" no matter which branch is installed.
        """

    CARGEN_ENFORCES_DETAIL = False

# cargen's own bar, stated in 9975a1e: a tight 64px crop passes, a 1080p still
# with a 40px car does not. Measured against real CityFlow S01 crops (median
# 108px on the short side) this keeps ~89% of them and drops the tail that
# cannot support geometry anyway.
MIN_SUBJECT_PX = 64


def require_subject_detail(crop_bgr: np.ndarray, min_px: int = MIN_SUBJECT_PX) -> None:
    """Raise InsufficientDetail if the crop is too small to reconstruct from.

    A no-op when cargen enforces this itself — deferring to the backend keeps
    one bar rather than two that can drift apart.

    The crop handed to fusion is already the vehicle's bounding box, so its
    short side *is* the subject's pixel size; no masking needed.
    """
    if CARGEN_ENFORCES_DETAIL:
        return
    if crop_bgr is None or crop_bgr.size == 0:
        raise InsufficientDetail("empty crop")
    short_side = min(crop_bgr.shape[:2])
    if short_side < min_px:
        raise InsufficientDetail(
            f"subject is {short_side}px on its short side, below the {min_px}px "
            f"floor — upsampling this would reconstruct interpolation, not detail")
