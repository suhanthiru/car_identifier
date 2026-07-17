"""View-invariant geometric attributes from a cargen splat cloud.

This is the payoff of holding a 3D model per target: length/width/height
RATIOS survive viewpoint changes that break 2D appearance embeddings (a
front view and a side view of the same car embed far apart in 2D; its
aspect ratios are the same from everywhere). The ratios feed the identity
cascade at the class/instance-attribute stage — deliberately NOT as the
ReID tiebreaker.

Honesty, up front:
- cargen's canonical frame discards metric scale (length ≈ 2.0 by
  convention), so absolute meters are unavailable; only ratios and coarse
  buckets are exposed.
- From a single traffic crop these numbers are rough by construction (one
  view, generative prior guessing the rest). They firm up as more confirmed
  sightings fuse — `confidence` reflects how much of the cloud is OBSERVED
  rather than prior guesswork, and the cascade weights accordingly.
- With cargen's `stub` prior backend the geometry is a procedural sedan and
  means nothing; `signature_from_cloud` reports the backend-quality caveat
  through `observed_fraction` and callers must gate on it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cargen.core.splat import GaussianCloud, Provenance
from cargen.prior_generation.canonical import canonicalize_orientation

# Extent percentiles: a splat cloud has stray outliers; 2..98 is robust.
_P_LO, _P_HI = 2.0, 98.0
# Minimum observed fraction before geometry is trusted for cascade use.
MIN_OBSERVED_FRACTION = 0.15
# Bucket boundaries chosen from published vehicle-class proportions
# (sedans h/l ~0.28-0.33, SUVs ~0.35-0.42, vans/buses higher). Coarse on
# purpose: single-crop reconstructions cannot support finer claims.
_PROFILE_BUCKETS = ((0.34, "low"), (0.44, "mid"), (10.0, "tall"))
_LW_BUCKETS = ((2.25, "compact"), (2.6, "standard"), (10.0, "elongated"))


@dataclass(frozen=True)
class GeometrySignature:
    length: float            # canonical units (length ≈ 2.0), NOT meters
    width: float
    height: float
    lw_ratio: float          # length / width
    hl_ratio: float          # height / length
    body_profile: str        # low | mid | tall
    length_class: str        # compact | standard | elongated
    n_splats: int
    observed_fraction: float # 0 = pure generative guess, 1 = fully confirmed

    @property
    def trustworthy(self) -> bool:
        return self.observed_fraction >= MIN_OBSERVED_FRACTION


def _bucket(value: float, buckets) -> str:
    for limit, name in buckets:
        if value < limit:
            return name
    return buckets[-1][1]


def signature_from_cloud(cloud: GaussianCloud) -> GeometrySignature | None:
    """Canonicalize and measure. None when the cloud is too small to mean
    anything (an empty or near-empty asset has no geometry to speak of)."""
    if cloud.n < 100:
        return None
    canonical, _ = canonicalize_orientation(cloud.positions)
    lo = np.percentile(canonical, _P_LO, axis=0)
    hi = np.percentile(canonical, _P_HI, axis=0)
    length, width, height = (hi - lo).tolist()
    if length <= 0 or width <= 0 or height <= 0:
        return None
    observed = float(np.mean(cloud.provenance == Provenance.OBSERVED))
    lw, hl = length / width, height / length
    return GeometrySignature(
        length=round(length, 4), width=round(width, 4), height=round(height, 4),
        lw_ratio=round(lw, 3), hl_ratio=round(hl, 3),
        body_profile=_bucket(hl, _PROFILE_BUCKETS),
        length_class=_bucket(lw, _LW_BUCKETS),
        n_splats=cloud.n, observed_fraction=round(observed, 3),
    )


def signature_to_attrs(sig: GeometrySignature | None) -> dict[str, str]:
    """Cascade-ready attributes, prefixed so their provenance is visible.

    Only emitted when enough of the cloud is real evidence — a signature
    measured mostly on the generative prior would be laundering a guess
    into an 'attribute'.
    """
    if sig is None or not sig.trustworthy:
        return {}
    return {
        "geom3d:body_profile": sig.body_profile,
        "geom3d:length_class": sig.length_class,
    }


@dataclass(frozen=True)
class GeometryComparison:
    verdict: str      # "consistent" | "inconsistent" | "insufficient"
    detail: str       # plain English for the fact list


def compare_signatures(
    a: GeometrySignature | None, b: GeometrySignature | None,
    ratio_tolerance: float = 0.18,
) -> GeometryComparison:
    """Numeric ratio comparison, tolerant of single-crop noise.

    Never treated as a hard veto by callers: reconstruction error on sparse
    evidence is too common for geometry alone to disqualify a match. It
    supports or cautions; plates and physics still outrank it.
    """
    if a is None or b is None or not (a.trustworthy and b.trustworthy):
        return GeometryComparison(
            "insufficient",
            "3D geometry not compared: too little confirmed structure on one side.")
    rel = max(abs(a.lw_ratio - b.lw_ratio) / b.lw_ratio,
              abs(a.hl_ratio - b.hl_ratio) / b.hl_ratio)
    if rel <= ratio_tolerance:
        return GeometryComparison(
            "consistent",
            f"3D proportions consistent (L/W {a.lw_ratio:.2f} vs {b.lw_ratio:.2f}, "
            f"H/L {a.hl_ratio:.2f} vs {b.hl_ratio:.2f}); view-invariant signal.")
    return GeometryComparison(
        "inconsistent",
        f"3D proportions disagree beyond tolerance (L/W {a.lw_ratio:.2f} vs "
        f"{b.lw_ratio:.2f}, H/L {a.hl_ratio:.2f} vs {b.hl_ratio:.2f}); "
        f"treated as caution, not veto — sparse reconstructions mismeasure.")
