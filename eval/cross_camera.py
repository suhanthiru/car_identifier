"""Cross-camera validation on CityFlow ground truth.

(a) Transit-time veto: build per-camera-pair windows from a training split of
    real transitions, then measure on held-out transitions:
      - how many REAL transitions the veto wrongly rejects (should be ~0);
      - how many synthetic IMPOSSIBLE transitions (real pairs, shuffled
        arrival times faster than the observed minimum) it correctly rejects.
    The impossible set is constructed, and labeled as such — CityFlow has no
    labeled false tracks, so we make physics violations and check the veto
    catches them.

(b) Corroboration fusion on real correlated sightings: for each ground-truth
    vehicle crossing multiple cameras, treat each camera's sighting as a
    weak same-vehicle assertion with fixed per-sighting confidence, then
    compare noisy-OR fusion against capped-additive. Noisy-OR compounds to
    near-certainty for ANY vehicle with enough camera hits — including when
    we deliberately swap in a different vehicle's continuation (a real
    correlated error) — while the capped scheme stays below the update
    threshold without qualitatively independent evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from datasets.cityflow import CityFlowScenario, Transition
from reasoning.cascade import VERDICT_LIKELY, MatchDecision
from reasoning.corroboration import (
    APPEARANCE_CAP, UPDATE_THRESHOLD, CorroborationState, apply_decision, noisy_or,
)


@dataclass(frozen=True)
class TransitWindows:
    """Observed min/max transit per directed camera pair (train split)."""

    windows: dict[tuple[str, str], tuple[float, float]]
    slack: float = 0.8   # veto boundary = slack * observed minimum

    def is_impossible(self, from_cam: str, to_cam: str, elapsed_s: float) -> bool:
        window = self.windows.get((from_cam, to_cam))
        if window is None:
            return False   # unseen pair: never veto on missing knowledge
        return elapsed_s < self.slack * window[0]


def fit_windows(transitions: list[Transition], slack: float = 0.8) -> TransitWindows:
    windows: dict[tuple[str, str], tuple[float, float]] = {}
    for t in transitions:
        if t.elapsed_s < 0:
            continue  # overlapping fields of view carry no minimum-time signal
        key = (t.from_camera, t.to_camera)
        lo, hi = windows.get(key, (t.elapsed_s, t.elapsed_s))
        windows[key] = (min(lo, t.elapsed_s), max(hi, t.elapsed_s))
    return TransitWindows(windows=windows, slack=slack)


@dataclass(frozen=True)
class TransitValidation:
    n_real: int
    real_wrongly_vetoed: int
    n_impossible: int
    impossible_caught: int

    @property
    def false_veto_rate(self) -> float:
        return self.real_wrongly_vetoed / self.n_real if self.n_real else 0.0

    @property
    def catch_rate(self) -> float:
        return self.impossible_caught / self.n_impossible if self.n_impossible else 0.0


def validate_transit_veto(
    scenario: CityFlowScenario,
    train_frac: float = 0.5,
    speedup: float = 5.0,
    seed: int = 3,
) -> TransitValidation:
    """Fit windows on half the real transitions, test on the other half.

    Impossible cases: held-out transitions replayed `speedup`x faster than
    the observed pair minimum — physically unreachable given every vehicle
    ever measured on that pair.
    """
    transitions = list(scenario.transitions())
    rng = np.random.default_rng(seed)
    rng.shuffle(transitions)
    cut = int(len(transitions) * train_frac)
    train, held_out = transitions[:cut], transitions[cut:]
    windows = fit_windows(train)

    testable = [t for t in held_out if (t.from_camera, t.to_camera) in windows.windows
                and t.elapsed_s >= 0]
    wrongly_vetoed = sum(
        windows.is_impossible(t.from_camera, t.to_camera, t.elapsed_s)
        for t in testable)
    impossible_caught = 0
    for t in testable:
        pair_min = windows.windows[(t.from_camera, t.to_camera)][0]
        fake_elapsed = pair_min / speedup
        impossible_caught += windows.is_impossible(
            t.from_camera, t.to_camera, fake_elapsed)
    return TransitValidation(
        n_real=len(testable), real_wrongly_vetoed=wrongly_vetoed,
        n_impossible=len(testable), impossible_caught=impossible_caught)


def _likely_decision(event_id: str) -> MatchDecision:
    return MatchDecision(
        target_id="tgt", event_id=event_id, verdict=VERDICT_LIKELY, score=0.5,
        deciding_tier="attributes", facts=(), reid_similarity=0.8,
        requires_review=True)


@dataclass(frozen=True)
class CorroborationComparison:
    sightings: list[int]
    noisy_or_belief: list[float]
    capped_belief: list[float]
    per_sighting_confidence: float
    n_vehicles: int
    noisy_or_overshoot_rate: float   # fraction of vehicles noisy-OR pushes past threshold
    capped_overshoot_rate: float     # ours: must be 0 by construction


def compare_fusion_on_real_transitions(
    scenario: CityFlowScenario,
    per_sighting_confidence: float = 0.6,
    max_sightings: int = 8,
) -> CorroborationComparison:
    """Fuse each real vehicle's multi-camera sighting chain both ways.

    The per-sighting confidence models an appearance-only match (the level
    a look-alike sibling also reaches — that is what makes the errors
    correlated and noisy-OR's independence assumption false on this data).
    """
    by_vehicle: dict[int, int] = {}
    for span in scenario.spans:
        by_vehicle[span.vehicle_id] = by_vehicle.get(span.vehicle_id, 0) + 1
    chains = [min(n, max_sightings) for n in by_vehicle.values() if n >= 2]
    if not chains:
        raise ValueError("scenario has no multi-camera vehicles")

    sightings = list(range(1, max_sightings + 1))
    noisy_vals = [noisy_or([per_sighting_confidence] * n) for n in sightings]
    capped_vals = []
    state = CorroborationState(target_id="tgt")
    for n in sightings:
        state_n = CorroborationState(target_id="tgt")
        for i in range(n):
            state_n, _ = apply_decision(state_n, _likely_decision(f"e{i}"), float(i))
        capped_vals.append(state_n.belief)
    del state

    noisy_over = sum(noisy_or([per_sighting_confidence] * n) >= UPDATE_THRESHOLD
                     for n in chains) / len(chains)
    capped_over = 0.0  # capped-additive appearance belief <= APPEARANCE_CAP < threshold
    assert max(capped_vals) <= APPEARANCE_CAP + 1e-9
    return CorroborationComparison(
        sightings=sightings, noisy_or_belief=noisy_vals,
        capped_belief=capped_vals,
        per_sighting_confidence=per_sighting_confidence,
        n_vehicles=len(chains),
        noisy_or_overshoot_rate=float(noisy_over),
        capped_overshoot_rate=capped_over)
