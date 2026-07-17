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
from typing import Callable, Sequence

from perception.embedder import max_similarity
from perception.types import Observation
from reasoning.facts import KIND_SUPPORT, Fact, has_veto, info, support
from reasoning.plausibility import run_all_checks
from reasoning.profile import TargetProfile
from sim.road_graph import RoadGraph

VERDICT_CONFIRMED = "confirmed"
VERDICT_LIKELY = "likely"
VERDICT_UNDECIDED = "undecided"
VERDICT_REJECTED = "rejected"

# Evidence weights. Deliberately simple and inspectable — these are design
# constants, not learned parameters.
W_PLATE_EXACT = 0.90
W_PLATE_NEAR = 0.35
W_CLASS_ATTRS = 0.20
W_INSTANCE_ATTR = 0.25
W_REID_MAX = 0.30        # ReID can contribute at most this much
LIKELY_THRESHOLD = 0.45
CONFIRM_THRESHOLD = 0.85

# Default ReID similarity -> [0,1] squash. Replaced by isotonic calibration
# in calibration/ (Phase 8); this fallback is a crude linear map and is
# labeled as such in decisions that use it.
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


@dataclass(frozen=True)
class CascadeConfig:
    reid_prob_fn: Callable[[float], float] = _uncalibrated_reid_prob
    reid_calibration_label: str = "uncalibrated-linear"


def evaluate(
    obs: Observation,
    profile: TargetProfile,
    graph: RoadGraph,
    config: CascadeConfig | None = None,
) -> MatchDecision:
    """Score one observation against one target, with full explanation."""
    cfg = config or CascadeConfig()
    facts = list(run_all_checks(obs, profile, graph))

    plate_support = [f for f in facts if f.check == "plate" and f.kind == KIND_SUPPORT]
    plate_exact = any("exactly matches" in f.text for f in plate_support)
    plate_near = any("OCR-confusable" in f.text for f in plate_support)
    attrs_support = any(f.check == "attributes" and f.kind == KIND_SUPPORT
                        and f.text.startswith("Class attributes") for f in facts)
    mark_matches = sum(1 for f in facts if f.check == "attributes"
                       and f.kind == KIND_SUPPORT and "mark matches" in f.text)

    reid_sim = max_similarity(obs.embedding, list(profile.gallery))
    reid_p = cfg.reid_prob_fn(reid_sim) if profile.gallery else 0.0
    if profile.gallery:
        facts.append(info(
            f"Appearance similarity {reid_sim:.2f} vs target gallery "
            f"({len(profile.gallery)} crops) -> p={reid_p:.2f} "
            f"[{cfg.reid_calibration_label}]. Tiebreaker only.", "reid"))
    else:
        facts.append(info("Target has no appearance gallery yet; ReID unavailable.", "reid"))

    vetoed = has_veto(facts)
    score = 0.0
    tier = "none"
    if plate_exact:
        score += W_PLATE_EXACT
        tier = "plate"
    elif plate_near:
        score += W_PLATE_NEAR
        tier = "plate"
    if attrs_support:
        score += W_CLASS_ATTRS
        if tier == "none":
            tier = "attributes"
    if mark_matches:
        score += W_INSTANCE_ATTR * mark_matches
        if tier in ("none", "attributes"):
            tier = "attributes"
    if profile.gallery and score > 0 and not vetoed:
        # ReID only refines an already-supported candidate — never rescues
        # one with no symbolic evidence.
        score += W_REID_MAX * reid_p
        if tier == "none":
            tier = "reid"
    score = min(1.0, score)

    if vetoed:
        anomaly = plate_exact  # plate says yes, physics/logic says no
        if anomaly:
            facts.append(support(
                "Anomaly: plate matches but a hard veto fired — possible plate "
                "clone or camera clock skew. Flagged for review.", "plate"))
        return MatchDecision(
            target_id=profile.target_id, event_id=obs.event_id,
            verdict=VERDICT_REJECTED, score=0.0, deciding_tier=tier,
            facts=tuple(facts), reid_similarity=reid_sim,
            requires_review=anomaly, anomaly=anomaly,
        )

    if plate_exact and score >= CONFIRM_THRESHOLD:
        verdict, review = VERDICT_CONFIRMED, False
    elif score >= LIKELY_THRESHOLD:
        # Appearance/attribute evidence alone never auto-confirms.
        verdict, review = VERDICT_LIKELY, True
    else:
        verdict, review = VERDICT_UNDECIDED, False
    return MatchDecision(
        target_id=profile.target_id, event_id=obs.event_id,
        verdict=verdict, score=score, deciding_tier=tier,
        facts=tuple(facts), reid_similarity=reid_sim, requires_review=review,
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
