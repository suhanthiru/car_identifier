"""Regenerate RESULTS.md from whatever datasets are actually on disk.

    python -m eval.run [--quick]

Rules, enforced here and non-negotiable:
- Absent dataset -> a PENDING section quoting DATASETS.md. Nothing is ever
  fabricated or extrapolated for data that isn't present.
- Every number's provenance (dataset, split sizes, model arch, calibration
  version) is written next to the number.
- Synthetic-fixture sections are labeled SYNTHETIC in the heading.

--quick subsamples VeRi for a fast smoke run and stamps the tables as
subsampled; headline numbers come from the full default run.
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import numpy as np

RESULTS_PATH = Path("RESULTS.md")
EMBED_ARCH = "osnet_x0_25 (torchreid, ImageNet-pretrained; VeRi-776-trained "\
    "weights are a manual download — see README)"


def _pending(name: str, why: str) -> str:
    return (f"## {name}\n\n**PENDING — dataset not present.** {why} "
            f"See [DATASETS.md](DATASETS.md) for the request/download steps. "
            f"This section is generated only from real data; nothing is "
            f"simulated in its place.\n")


# --------------------------------------------------------------- veri block

def veri_section(quick: bool) -> str:
    from datasets.veri776 import Veri776

    if not Veri776.exists():
        return _pending(
            "VeRi-776: retrieval, calibration, ablation",
            "VeRi-776 requires the authors' research-use request form.")
    from calibration.isotonic import build_report, save
    from eval.ablation import run_ablation
    from eval.embed_dataset import embed_images
    from eval.hard_negatives import hardest_pairs, mine_pairs
    from eval.plots import (
        plot_cmc, plot_pair_gallery, plot_pr_sweep, plot_reliability,
    )
    from eval.reliability import compute_reliability
    from eval.retrieval import evaluate_retrieval
    import cv2

    ds = Veri776.load()
    query, gallery = list(ds.query), list(ds.gallery)
    if quick:
        query, gallery = query[:200], gallery[:1500]
    print(f"VeRi-776: {len(query)} query / {len(gallery)} gallery images")
    q_emb = embed_images([i.path for i in query], f"veri-query-{len(query)}")
    g_emb = embed_images([i.path for i in gallery], f"veri-gallery-{len(gallery)}")

    res = evaluate_retrieval(
        q_emb, [i.vehicle_id for i in query], [i.camera_id for i in query],
        g_emb, [i.vehicle_id for i in gallery], [i.camera_id for i in gallery])
    plot_cmc(res.cmc, f"VeRi-776 CMC ({EMBED_ARCH.split()[0]})", "veri_cmc.png")

    pairs = mine_pairs(gallery, g_emb)
    report = build_report(pairs)
    save(report, "calibration/artifacts/veri776.json")
    rel = compute_reliability(pairs, report.model)
    plot_reliability(rel.bins, rel.ece, "veri_reliability.png")
    plot_pr_sweep(report.sweep, report.chosen_threshold, "veri_sweep.png")

    hard = hardest_pairs(pairs, top=6)
    gallery_rows = []
    for p in hard:
        a = cv2.imread(str(gallery[p.a_index].path))
        b = cv2.imread(str(gallery[p.b_index].path))
        if a is not None and b is not None:
            gallery_rows.append((a, b, f"DIFFERENT vehicles, sim "
                                       f"{p.similarity:.3f} ({p.bucket})"))
    if gallery_rows:
        plot_pair_gallery(gallery_rows, "veri_confusables.png",
                          "Hardest real look-alike pairs (all different vehicles)")

    metrics, cases = run_ablation(
        query, q_emb, gallery, g_emb, threshold=report.chosen_threshold)
    raw = next(m for m in metrics if m.name == "raw")
    cas = next(m for m in metrics if m.name == "cascade")

    subsample_note = " *(subsampled `--quick` run — not headline numbers)*" if quick else ""
    n_hard = sum(p.hard_negative for p in pairs)
    lines = [
        f"## VeRi-776: retrieval, calibration, ablation{subsample_note}",
        "",
        f"Embeddings: {EMBED_ARCH}. {len(query)} query / {len(gallery)} gallery "
        f"images, standard same-camera exclusion protocol.",
        "",
        "### Retrieval",
        "",
        "| Rank-1 | Rank-5 | Rank-10 | mAP | queries |",
        "|---|---|---|---|---|",
        f"| {res.rank1:.1%} | {res.rank(5):.1%} | {res.rank(10):.1%} "
        f"| {res.mean_ap:.1%} | {res.n_queries_scored} |",
        "",
        "![CMC](eval/figures/veri_cmc.png)",
        "",
        "### Calibration on mined real hard negatives",
        "",
        f"{len(pairs)} pairs ({n_hard} hard negatives = same-color same-body "
        f"different-vehicle, mined by bucket). Calibration version "
        f"`isotonic-{report.model.version}`; ECE {rel.ece:.3f}. Alert threshold "
        f"{report.chosen_threshold:.3f} chosen for target precision "
        f"{report.target_precision:.2f}; hard-negative FPR at that threshold: "
        f"{report.hard_negative_fpr_at_threshold:.1%}.",
        "",
        "![reliability](eval/figures/veri_reliability.png)",
        "![sweep](eval/figures/veri_sweep.png)",
        "![confusables](eval/figures/veri_confusables.png)",
        "",
        "### THE ABLATION: raw ReID alerting vs cascade + vetoes",
        "",
        "Attribute channel uses the dataset's own labels (a perfect attribute "
        "classifier), so the cascade delta is an **upper bound** on what a real "
        "attribute head buys. Review-rate is the cost of refusing to guess.",
        "",
        "| policy | precision | recall | F1 | alerts | false positives | reviews |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in (raw, cas):
        r = m.row()
        lines.append(f"| {r['policy']} | {r['precision']:.1%} | {r['recall']:.1%} "
                     f"| {r['f1']:.1%} | {r['alerts']} | {r['false_positives']} "
                     f"| {r['review_rate']} |")
    dp = cas.precision - raw.precision
    fp_cut = (1 - cas.false_positives / raw.false_positives) if raw.false_positives else 0.0
    lines += [
        "",
        f"**Delta: {dp:+.1%} precision; {fp_cut:.0%} of raw false positives "
        f"eliminated by the attribute veto + look-alike ambiguity refusal.**",
        "",
        "### Failure cases (honest, not curated away)",
        "",
    ]
    wrong = [c for c in cases if c.action == "alert" and not c.correct][:5]
    refused = [c for c in cases if c.action == "review"][:5]
    for c in wrong:
        lines.append(f"- **Wrong alert** (sim {c.similarity:.2f}): query "
                     f"`{query[c.query_index].path.name}` matched gallery "
                     f"`{gallery[c.top_index].path.name}` — {c.reason}")
    for c in refused:
        lines.append(f"- **Refused (review)**: query "
                     f"`{query[c.query_index].path.name}` — {c.reason}")
    if not wrong:
        lines.append("- No wrong alerts at the chosen threshold on this run.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------- vehicleid block

def vehicleid_section(quick: bool) -> str:
    from datasets.vehicleid import VehicleID

    if not VehicleID.exists():
        return _pending("VehicleID: retrieval",
                        "PKU VehicleID requires an emailed research request.")
    from eval.embed_dataset import embed_images
    from eval.plots import plot_cmc
    from eval.retrieval import evaluate_retrieval

    ds = VehicleID()
    split = ds.test_split(800)
    query, gallery = list(split.query), list(split.gallery)
    if quick:
        query = query[:300]
    q_emb = embed_images([i.path for i in query], f"vid-query-{len(query)}")
    g_emb = embed_images([i.path for i in gallery], f"vid-gallery-{len(gallery)}")
    res = evaluate_retrieval(
        q_emb, [i.vehicle_id for i in query], ["q"] * len(query),
        g_emb, [i.vehicle_id for i in gallery], ["g"] * len(gallery))
    plot_cmc(res.cmc, "VehicleID (test-800) CMC", "vehicleid_cmc.png")
    return "\n".join([
        "## VehicleID: retrieval",
        "",
        f"Test-800 protocol (1 gallery image per identity). {EMBED_ARCH}.",
        "",
        "| Rank-1 | Rank-5 | mAP | queries |",
        "|---|---|---|---|",
        f"| {res.rank1:.1%} | {res.rank(5):.1%} | {res.mean_ap:.1%} "
        f"| {res.n_queries_scored} |",
        "",
        "![CMC](eval/figures/vehicleid_cmc.png)",
        "",
    ])


# ----------------------------------------------------------- cityflow block

def cityflow_section() -> str:
    from datasets.cityflow import CityFlow

    if not CityFlow.exists():
        return _pending(
            "CityFlow: cross-camera transit veto + corroboration",
            "AI City Challenge MTMC data requires a signed data-use agreement.")
    from eval.cross_camera import (
        compare_fusion_on_real_transitions, validate_transit_veto,
    )
    from eval.plots import plot_corroboration_curves, plot_transit_hist

    ds = CityFlow()
    names = ds.scenario_names()
    lines = ["## CityFlow: cross-camera validation", ""]
    for name in names[:3]:
        scen = ds.load_scenario(name)
        val = validate_transit_veto(scen)
        cmp = compare_fusion_on_real_transitions(scen)
        elapsed = [t.elapsed_s for t in scen.transitions() if t.elapsed_s >= 0]
        if elapsed:
            plot_transit_hist(elapsed, min(elapsed) * 0.8,
                              f"cityflow_{name}_transits.png")
        plot_corroboration_curves(cmp.sightings, cmp.noisy_or_belief,
                                  cmp.capped_belief, f"cityflow_{name}_fusion.png")
        lines += [
            f"### Scenario {name} ({len(scen.cameras)} cameras, "
            f"{len(scen.spans)} ground-truth tracks)",
            "",
            "| check | result |",
            "|---|---|",
            f"| real transitions wrongly vetoed | {val.real_wrongly_vetoed}"
            f"/{val.n_real} ({val.false_veto_rate:.1%}) |",
            f"| constructed impossible transitions caught | {val.impossible_caught}"
            f"/{val.n_impossible} ({val.catch_rate:.1%}) |",
            f"| noisy-OR pushes past update threshold | "
            f"{cmp.noisy_or_overshoot_rate:.0%} of {cmp.n_vehicles} real "
            f"multi-camera vehicles |",
            f"| capped-additive pushes past threshold | "
            f"{cmp.capped_overshoot_rate:.0%} (cap is below it by design) |",
            "",
            f"![transits](eval/figures/cityflow_{name}_transits.png)",
            f"![fusion](eval/figures/cityflow_{name}_fusion.png)",
            "",
            "Impossible transitions are *constructed* (real pairs replayed "
            "faster than any observed vehicle) because CityFlow has no labeled "
            "false tracks; the table says exactly what was tested.",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------- synthetic block

def synthetic_section() -> str:
    from reasoning.corroboration import (
        APPEARANCE_CAP, UPDATE_THRESHOLD, CorroborationState, apply_decision, noisy_or,
    )
    from eval.cross_camera import _likely_decision

    state = CorroborationState(target_id="t")
    beliefs = []
    for i in range(8):
        state, _ = apply_decision(state, _likely_decision(f"e{i}"), float(i))
        beliefs.append(round(state.belief, 3))
    return "\n".join([
        "## SYNTHETIC adversarial fixture: the independence trap "
        "(not real data)",
        "",
        "Deterministic unit-level demonstration on the synthetic world's "
        "correlated look-alikes (the same invariant is validated on real "
        "CityFlow transitions above when that dataset is present):",
        "",
        f"- eight correlated appearance-only sightings, capped-additive belief: "
        f"`{beliefs}` — never exceeds the appearance cap "
        f"{APPEARANCE_CAP} < update threshold {UPDATE_THRESHOLD};",
        f"- the same eight sightings under noisy-OR (p=0.6 each): "
        f"`{round(noisy_or([0.6] * 8), 5)}` — false near-certainty.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="subsample for a fast smoke run (not headline numbers)")
    args = parser.parse_args()

    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        "# RESULTS",
        "",
        f"*Regenerated by `python -m eval.run` on {stamp}. Every section is "
        f"computed from data actually present on disk; missing datasets "
        f"produce PENDING sections, never substituted numbers. Nothing here "
        f"claims production accuracy — see the Limits section of the README.*",
        "",
        veri_section(args.quick),
        vehicleid_section(args.quick),
        cityflow_section(),
        synthetic_section(),
    ]
    RESULTS_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
