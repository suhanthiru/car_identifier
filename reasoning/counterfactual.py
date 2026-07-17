"""Counterfactual explanations: the nearest single change that flips a decision.

Contestability made concrete. For each decision the console already shows the
fact list ("why"); counterfactuals show the "what would have to be different":

    "matched on plate; without it this would be an ambiguous candidate set"
    "accepted; would be REJECTED if the transit were faster than 120s"
    "rejected by body-style contradiction; would pass if body-style agreed"
    "refused to individuate; a plate read would name the vehicle"

Because the symbolic layer is a pure function of the signals, the engine
perturbs ONE signal to its decision boundary and re-runs the exact same
`score_from_signals` — so every flip point is provably faithful to the live
rule, not reconstructed prose. Each counterfactual holds all other signals
fixed; it does not model correlated changes (stated in the docs).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from reasoning.cascade import (
    VERDICT_REJECTED, CascadeConfig, score_from_signals,
)
from reasoning.signals import MatchSignals


@dataclass(frozen=True)
class Counterfactual:
    signal: str            # plate | transit | body_style | distinctiveness
    current_outcome: str
    flipped_outcome: str
    boundary: str          # the numeric/logical boundary, e.g. "fastest = 120s"
    text: str              # plain English, user-facing


def _verdict(signals: MatchSignals, config: CascadeConfig) -> str:
    return score_from_signals(signals, config).verdict


def counterfactuals(
    signals: MatchSignals, config: CascadeConfig | None = None
) -> tuple[Counterfactual, ...]:
    cfg = config or CascadeConfig()
    current = _verdict(signals, cfg)
    out: list[Counterfactual] = []

    # Plate: the single strongest lever.
    if signals.plate_exact:
        flipped = _verdict(replace(signals, plate_exact=False), cfg)
        if flipped == current:
            text = (f"The plate match is not what decides this — without it the "
                    f"outcome is still '{flipped}' (another signal dominates).")
        else:
            text = (f"Matched on the plate. Without the plate read, this would be "
                    f"'{flipped}' on the remaining, class-level evidence.")
        out.append(Counterfactual("plate", current, flipped, "plate match removed", text))
    elif signals.plate_available and not signals.plate_contradiction:
        flipped = _verdict(replace(signals, plate_exact=True), cfg)
        if flipped != current:
            out.append(Counterfactual(
                "plate", current, flipped, "clean plate match added",
                f"A clean plate read would make this '{flipped}'."))

    # Transit: the physics boundary is a single scalar.
    if signals.transit_applicable and signals.transit_fastest_s is not None:
        fastest = signals.transit_fastest_s
        dt = signals.transit_dt_s
        if signals.transit_veto:
            flipped = _verdict(replace(signals, transit_veto=False), cfg)
            out.append(Counterfactual(
                "transit", current, flipped, f"fastest possible = {fastest:.0f}s",
                f"Rejected as physically impossible ({dt:.0f}s for a hop that "
                f"needs at least {fastest:.0f}s). It would pass if the vehicle had "
                f"had ≥ {fastest:.0f}s to travel."))
        else:
            flipped = _verdict(replace(signals, transit_veto=True), cfg)
            out.append(Counterfactual(
                "transit", current, flipped, f"fastest possible = {fastest:.0f}s",
                f"Accepted; it would be REJECTED if the transit were faster than "
                f"the {fastest:.0f}s minimum for this camera hop."))

    # Body-style contradiction.
    if signals.body_veto:
        flipped = _verdict(replace(signals, body_veto=False), cfg)
        out.append(Counterfactual(
            "body_style", current, flipped, "body-style contradiction removed",
            "Rejected by a body-style contradiction; it would pass if the body "
            "style agreed."))

    # Distinctiveness refusal.
    if current == "candidate":
        flipped = _verdict(replace(signals, plate_exact=True), cfg)
        out.append(Counterfactual(
            "distinctiveness", current, flipped, "distinctiveness below floor",
            "Refused to name an individual — the evidence is class-level. A plate "
            "read (or another distinguishing mark) would let it individuate."))

    return tuple(out)


def render_counterfactuals(cfs: tuple[Counterfactual, ...]) -> str:
    """One line per counterfactual, for the console explanation panel."""
    return "\n".join(f"[?] {c.text}" for c in cfs)
