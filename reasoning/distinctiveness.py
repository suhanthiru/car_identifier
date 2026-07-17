"""Distinctiveness: how uniquely the confirmed evidence names ONE vehicle.

The thesis in one number. A silver sedan matched on class attributes alone
describes thousands of vehicles — the honest output is a candidate SET, not
an individual. A plate names one. This score, in [0,1], gates whether the
system is allowed to individuate at all: below a floor it refuses and returns
the set, and no alert auto-fires on sub-floor evidence.

It is derived from the same cascade weights (reasoning/weights.py), not a new
parallel ladder — it generalizes the existing plate-grade vs appearance-grade
axis. It reads SYMBOLIC signals only; ReID similarity never feeds it (a good
appearance match between look-alikes is exactly the case we must not let
masquerade as distinctiveness).
"""
from __future__ import annotations

from reasoning.signals import MatchSignals
from reasoning.weights import (
    W_CLASS_ATTRS, W_GEOMETRY, W_INSTANCE_ATTR, W_PLATE_EXACT, W_PLATE_NEAR,
)

# Contributions normalized so a clean plate == 1.0.
_NEAR = W_PLATE_NEAR / W_PLATE_EXACT        # ~0.39
_MARK = W_INSTANCE_ATTR / W_PLATE_EXACT     # ~0.28 per distinguishing mark
_GEOM = W_GEOMETRY / W_PLATE_EXACT          # ~0.11 (coarse, view-invariant)
_CLASS = W_CLASS_ATTRS / W_PLATE_EXACT      # ~0.22 (class-only floor)


def distinctiveness(signals: MatchSignals) -> float:
    """Symbolic-only distinctiveness in [0,1]. ReID excluded by construction."""
    if signals.plate_exact:
        return 1.0
    d = _NEAR if signals.plate_near else 0.0
    d += _MARK * signals.mark_match_count
    if signals.geometry_consistent:
        d += _GEOM
    if not (signals.plate_near or signals.mark_match_count or signals.geometry_consistent) \
            and signals.attrs_consistent:
        d = max(d, _CLASS)   # class attributes alone: the low floor
    return min(1.0, d)
