"""Identity cascade: ordered evidence tiers, ReID last.

Why a cascade instead of one similarity score: a single score lets strong
appearance similarity overrule cheap-to-check hard evidence, which is
exactly backwards for look-alike vehicles. Here evidence is consulted in
order of reliability —

    1. plate            (near-unique when read cleanly)
    2. class attributes (make/model/body/color — eliminates, never confirms)
    3. instance attributes (distinguishing marks — narrows within look-alikes)
    4. ReID embedding   (tiebreaker ONLY; can rank candidates, cannot
                         confirm a match by itself)

— and the symbolic plausibility checks can veto at any tier. A veto is
final: physics outranks appearance.

Verdicts:
- confirmed: plate-grade evidence, no vetoes. Auto-associable.
- likely:    attribute + ReID support, no vetoes. Goes to operator review;
             the system never auto-confirms on appearance alone.
- undecided: not enough evidence either way.
- rejected:  a hard veto fired (or evidence actively contradicts).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from perception.embedder import max_similarity
from perception.types import Observation
from reasoning.facts import Fact, info, support
from reasoning.plausibility import run_all_checks
from reasoning.profile import TargetProfile
from reasoning.signals import MatchSignals, compute_signals
from reasoning.weights import (
    CONFIRM_THRESHOLD, LIKELY_THRESHOLD, W_CLASS_ATTRS, W_GEOMETRY,
    W_INSTANCE_ATTR, W_PLATE_EXACT, W_PLATE_NEAR, W_REID_MAX,
)
from sim.road_graph import RoadGraph

VERDICT_CONFIRMED = "confirmed"
VERDICT_LIKELY = "likely"
VERDICT_UNDECIDED = "undecided"
VERDICT_REJECTED = "rejected"
# Feature B: evidence is real but too generic to name one vehicle. Non-
# associable; routes to review with a candidate set. Falls through every
# `verdict in (CONFIRMED, LIKELY)` gate downstream, so it auto-fires nothing.
VERDICT_CANDIDATE = "candidate"


# Default ReID similarity -> [0,1] squash. Replaced by isotonic calibration
# in calibration/; this fallback is a crude linear map and is labeled as such
# in decisions that use it.
def _uncalibrated_reid_prob(sim: float) -> float:
    return max(0.0, min(1.0, (sim - 0.5) / 0.5))


@dataclass(frozen=True)
class MatchDecision:
    target_id: str
    event_id: str
    verdict: str
    score: float
    deciding_tier: str            # "plate" | "attributes" | "reid" | "none"
    facts: tuple[Fact, ...]
    reid_similarity: float
    requires_review: bool
    # True when a plate matched but plausibility vetoed: the classic
    # cloned-plate / clock-skew anomaly. Surfaced loudly in the console.
    anomaly: bool = False
    # Structured signals behind this decision (feature foundation).
    signals: MatchSignals | None = None
    # Feature B: how uniquely the confirmed evidence names a vehicle [0,1].
    distinctiveness: float = 1.0
    refused_to_individuate: bool = False
    candidate_ids: tuple[str, ...] = ()
    # Feature C: nearest single-signal changes that flip the outcome.
    # Typed as Any to avoid a cascade<->counterfactual import cycle.
    counterfactuals: tuple[Any, ...] = ()


@dataclass(frozen=True)
class ScoredDecision:
    """Pure output of the numeric policy — no facts, no side effects."""

    verdict: str
    score: float
    deciding_tier: str
    anomaly: bool
    requires_review: bool


@dataclass(frozen=True)
class CascadeConfig:
    reid_prob_fn: Callable[[float], float] = _uncalibrated_reid_prob
    reid_calibration_label: str = "uncalibrated-linear"


def score_from_signals(signals: MatchSignals, config: CascadeConfig | None = None) -> ScoredDecision:
    """The entire numeric verdict policy, as a pure function of the signals.

    Reproduces the historical scoring exactly. The counterfactual engine
    re-runs THIS function on perturbed signals, so its flip points are
    provably faithful to the live rule rather than reconstructed prose.
    """
    vetoed = signals.any_veto
    score = 0.0
    tier = "none"
    if signals.plate_exact:
        score += W_PLATE_EXACT
        tier = "plate"
    elif signals.plate_near:
        score += W_PLATE_NEAR
        tier = "plate"
    if signals.attrs_consistent:
        score += W_CLASS_ATTRS
        if tier == "none":
            tier = "attributes"
    if signals.mark_match_count:
        score += W_INSTANCE_ATTR * signals.mark_match_count
        if tier in ("none", "attributes"):
            tier = "attributes"
    if signals.geometry_consistent:
        score += W_GEOMETRY
        if tier == "none":
            tier = "attributes"
    if signals.has_gallery and score > 0 and not vetoed:
        # ReID only refines an already-supported candidate — never rescues one.
        score += W_REID_MAX * signals.reid_prob
        if tier == "none":
            tier = "reid"
    score = min(1.0, score)

    if vetoed:
        anomaly = signals.plate_exact  # plate says yes, physics/logic says no
        return ScoredDecision(VERDICT_REJECTED, 0.0, tier, anomaly, requires_review=anomaly)
    if signals.plate_exact and score >= CONFIRM_THRESHOLD:
        return ScoredDecision(VERDICT_CONFIRMED, score, tier, False, False)
    if score >= LIKELY_THRESHOLD:
        # Appearance/attribute evidence alone never auto-confirms.
        return ScoredDecision(VERDICT_LIKELY, score, tier, False, True)
    return ScoredDecision(VERDICT_UNDECIDED, score, tier, False, False)


def evaluate(
    obs: Observation,
    profile: TargetProfile,
    graph: RoadGraph,
    config: CascadeConfig | None = None,
) -> MatchDecision:
    """Score one observation against one target, with full explanation."""
    cfg = config or CascadeConfig()
    facts = list(run_all_checks(obs, profile, graph))
    signals = compute_signals(obs, profile, graph)

    reid_sim = max_similarity(obs.embedding, list(profile.gallery))
    reid_p = cfg.reid_prob_fn(reid_sim) if profile.gallery else 0.0
    signals = signals.with_reid(bool(profile.gallery), reid_sim, reid_p)
    if profile.gallery:
        facts.append(info(
            f"Appearance similarity {reid_sim:.2f} vs target gallery "
            f"({len(profile.gallery)} crops) -> p={reid_p:.2f} "
            f"[{cfg.reid_calibration_label}]. Tiebreaker only.", "reid"))
    else:
        facts.append(info("Target has no appearance gallery yet; ReID unavailable.", "reid"))

    scored = score_from_signals(signals, cfg)
    if scored.anomaly:
        facts.append(support(
            "Anomaly: plate matches but a hard veto fired — possible plate "
            "clone or camera clock skew. Flagged for review.", "plate"))

    return MatchDecision(
        target_id=profile.target_id, event_id=obs.event_id,
        verdict=scored.verdict, score=scored.score, deciding_tier=scored.deciding_tier,
        facts=tuple(facts), reid_similarity=reid_sim,
        requires_review=scored.requires_review, anomaly=scored.anomaly,
        signals=signals,
    )


@dataclass(frozen=True)
class RankedMatch:
    best: MatchDecision
    margin: float                 # score gap to the runner-up
    all_decisions: tuple[MatchDecision, ...] = field(repr=False)


def rank_candidates(
    obs: Observation,
    profiles: Sequence[TargetProfile],
    graph: RoadGraph,
    config: CascadeConfig | None = None,
) -> RankedMatch | None:
    """Evaluate an observation against every flagged target.

    ReID's tiebreaker role lives here: among surviving candidates with
    equal symbolic evidence, the ReID contribution (already folded into
    the score with a capped weight) decides the ranking. A thin margin
    between look-alike candidates is itself a reason for review.
    """
    if not profiles:
        return None
    decisions = sorted(
        (evaluate(obs, p, graph, config) for p in profiles),
        key=lambda d: d.score, reverse=True,
    )
    margin = (decisions[0].score - decisions[1].score) if len(decisions) > 1 else decisions[0].score
    return RankedMatch(best=decisions[0], margin=margin, all_decisions=tuple(decisions))
