"""Cascade evidence weights and thresholds — the numeric policy, in one place.

Extracted from cascade.py so distinctiveness.py and counterfactual.py can import
the constants without a cycle back through cascade. These are DESIGN CONSTANTS,
deliberately simple and inspectable — not learned parameters.
"""
from __future__ import annotations

# Evidence weights (contribution to the [0,1] match score).
W_PLATE_EXACT = 0.90
W_PLATE_NEAR = 0.35
W_CLASS_ATTRS = 0.20
W_INSTANCE_ATTR = 0.25
# 3D-geometry consistency (car3d bridge): attribute-tier, small on purpose —
# view-invariant but coarse; it narrows, it does not identify.
W_GEOMETRY = 0.10
# ReID appearance similarity: tiebreaker only, capped.
W_REID_MAX = 0.30
# Render-and-compare verification (car3d/match.py): a SEPARATELY-CALIBRATED
# tiebreaker consulted only on an already-narrowed look-alike shortlist. It
# contributes 0 to this score — it can reorder tied candidates and annotate a
# review, never move a decision across a threshold. Kept here for discoverability.
W_RENDER_MAX = 0.0

# Verdict thresholds.
LIKELY_THRESHOLD = 0.45
CONFIRM_THRESHOLD = 0.85

# Distinctiveness floor (feature B): below this, the system refuses to assert an
# individual and returns a candidate set instead. Sits between class-only
# (~0.22) and plate-near (~0.39) distinctiveness.
DISTINCTIVENESS_FLOOR = 0.30
