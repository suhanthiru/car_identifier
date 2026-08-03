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

# The VeRi block used to be hard-wired to OSNet, which meant the headline
# ablation could only ever be measured against an ImageNet-pretrained backbone
# that scores near chance cross-camera (fair AUC 0.52). That understates the
# question the ablation asks: does the symbolic layer still help when the
# appearance model is already good? Selectable so both answers are reachable.
_EMBEDDERS = {
    "osnet": ("osnet_x0_25 (torchreid, ImageNet-pretrained; VeRi-776-trained "
              "weights are a manual download — see README)", "osnet_x0_25"),
    "fastreid": ("FastReID SBS(R50-ibn), VeRi-776-finetuned "
                 "(models/veri_sbs_R50-ibn.pth)", "fastreid_sbs_r50ibn"),
}
EMBED_KEY = "osnet"
EMBED_ARCH = _EMBEDDERS[EMBED_KEY][0]


def _select_embedder(key: str) -> None:
    """Point the VeRi block at one of the two backbones.

    Also switches the embedding cache tag, so switching backbones can never
    silently reuse the other one's cached vectors — the failure that would
    look like "FastReID scores exactly what OSNet scored".
    """
    global EMBED_KEY, EMBED_ARCH
    EMBED_KEY = key
    EMBED_ARCH = _EMBEDDERS[key][0]


def _embedder_for_run():
    """(embedder instance or None, cache arch tag)."""
    _, arch_tag = _EMBEDDERS[EMBED_KEY]
    if EMBED_KEY == "fastreid":
        from perception.fastreid_backbone import FastReidEmbedder

        return FastReidEmbedder(), arch_tag
    return None, arch_tag


def _pending(name: str, why: str, headline: str = "dataset not present",
             see_datasets: bool = True) -> str:
    """A PENDING block that names the actual blocker.

    `headline` is parameterised because not every gap is a missing dataset:
    the per-embedder section needs summary artifacts produced by scripts that
    are too expensive to run inline, and reporting that as "dataset not
    present" sends the reader to DATASETS.md to re-download data they already
    have.
    """
    tail = ("See [DATASETS.md](DATASETS.md) for the request/download steps. "
            if see_datasets else "")
    return (f"## {name}\n\n**PENDING — {headline}.** {why} {tail}"
            f"This section is generated only from real data; nothing is "
            f"simulated in its place.\n")


# --------------------------------------------------------------- veri block

def _threshold_note(report) -> str:
    """Say whether the target precision was actually reached.

    "chosen for target precision 0.95" reads as a guarantee. When no live
    threshold reaches the target, `choose_threshold` deliberately falls back
    to the best-F1 point — a materially different claim, and on a weak
    embedder the honest headline: a similarity threshold cannot separate
    these look-alikes at any operating point.
    """
    at = min((p for p in report.sweep
              if abs(p.threshold - report.chosen_threshold) < 1e-9),
             key=lambda p: p.threshold, default=None)
    reached = at is not None and at.precision >= report.target_precision
    if reached:
        return (f"Alert threshold {report.chosen_threshold:.3f} meets the "
                f"{report.target_precision:.2f} target precision.")
    got = f"{at.precision:.1%}" if at is not None else "unknown"
    return (f"Alert threshold {report.chosen_threshold:.3f} — **no threshold "
            f"reached the {report.target_precision:.2f} target precision** "
            f"(best achievable here: {got}), so this is the best-F1 fallback. "
            f"That is the finding, not a configuration detail: on these mined "
            f"look-alikes this embedder's similarity does not separate identity "
            f"at any operating point.")


def available_embedders(requested: str) -> list[str]:
    """Which backbones this machine can actually run, `requested` first.

    Reporting one embedder at a time was a mistake: `--embedder fastreid`
    overwrote the OSNet ablation, and the comparison between a near-chance
    backbone and a vehicle-finetuned one IS the interesting result — whether
    the symbolic layer still earns its keep once appearance matching is good.
    Both are emitted whenever both are installed.
    """
    from pathlib import Path as _P

    keys = [requested] + [k for k in sorted(_EMBEDDERS) if k != requested]
    out = []
    for key in keys:
        if key == "fastreid" and not _P("models/veri_sbs_R50-ibn.pth").is_file():
            continue
        out.append(key)
    return out


def _ablation_sweep_block(query, q_emb, gallery, g_emb, chosen, pct) -> list[str]:
    """The ablation across operating points, not just the chosen one.

    A single threshold can hide the whole effect. On FastReID the chosen
    threshold fires 36 alerts out of 1678 (2.1% recall) and both policies score
    100% precision with zero false positives — reported alone that reads as
    "the cascade adds nothing", when what it actually means is "at this
    operating point there was nothing left to add".

    The cause is a distribution mismatch worth naming: the threshold is fitted
    on `mine_pairs`' adversarially-selected top-k look-alikes and then applied
    to natural query->gallery top-1 matches. Adversarial pairs are far harder,
    so the threshold that survives them is far too conservative for the real
    task. Sweeping shows where the cascade actually earns its keep.
    """
    from eval.ablation import run_ablation

    rows = []
    for t in (0.50, 0.60, 0.70, 0.80, 0.90):
        metrics, _ = run_ablation(query, q_emb, gallery, g_emb, threshold=t)
        raw = next(m for m in metrics if m.name == "raw")
        cas = next(m for m in metrics if m.name == "cascade")
        cut = (1 - cas.false_positives / raw.false_positives
               if raw.false_positives else float("nan"))
        mark = " *(chosen)*" if abs(t - chosen) < 0.026 else ""
        rows.append(
            f"| {t:.2f}{mark} | {pct(raw.precision)} | {pct(raw.recall)} | "
            f"{raw.false_positives} | {pct(cas.precision)} | {pct(cas.recall)} | "
            f"{cas.false_positives} | {cas.reviews} | {pct(cut)} |")
    return [
        "",
        "#### The same ablation across operating points",
        "",
        "One threshold can hide the entire effect, so here is the sweep. The "
        "chosen threshold comes from a calibration fitted on *adversarially "
        "mined* look-alike pairs and is then applied to *natural* top-1 "
        "matches; adversarial pairs are much harder, so that threshold is far "
        "more conservative than the task needs, and at the extreme it alerts "
        "so rarely that neither policy can be wrong. Read the high-recall rows "
        "for what the symbolic layer actually buys.",
        "",
        "| threshold | raw P | raw R | raw FP | cascade P | cascade R | "
        "cascade FP | reviews | FP eliminated |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
    ]


def veri_section(quick: bool, embedders: list[str] | None = None) -> str:
    from datasets.veri776 import Veri776

    if not Veri776.exists():
        return _pending(
            "VeRi-776: retrieval, calibration, ablation",
            "VeRi-776 requires the authors' research-use request form.")
    blocks = []
    for i, key in enumerate(embedders or [EMBED_KEY]):
        _select_embedder(key)
        blocks.append(_veri_block(quick, primary=(i == 0)))
    return "\n".join(blocks)


def _veri_block(quick: bool, primary: bool = True) -> str:
    from datasets.veri776 import Veri776
    from calibration.isotonic import build_report, save, split_by_identity
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
    print(f"VeRi-776: {len(query)} query / {len(gallery)} gallery images "
          f"[{EMBED_KEY}]")
    embedder, arch_tag = _embedder_for_run()
    q_emb = embed_images([i.path for i in query], f"veri-query-{len(query)}",
                         embedder=embedder, arch=arch_tag)
    g_emb = embed_images([i.path for i in gallery], f"veri-gallery-{len(gallery)}",
                         embedder=embedder, arch=arch_tag)

    # Per-embedder filenames: two blocks in one document must not overwrite
    # each other's figures, and a stale veri_cmc.png silently showing the other
    # backbone's curve is exactly the kind of quiet wrongness this repo cares
    # about. The primary run also writes the unsuffixed artifact the cascade
    # loads at runtime.
    sfx = f"_{EMBED_KEY}"
    res = evaluate_retrieval(
        q_emb, [i.vehicle_id for i in query], [i.camera_id for i in query],
        g_emb, [i.vehicle_id for i in gallery], [i.camera_id for i in gallery])
    plot_cmc(res.cmc, f"VeRi-776 CMC ({EMBED_ARCH.split()[0]})", f"veri_cmc{sfx}.png")

    pairs = mine_pairs(gallery, g_emb)
    # Identity-disjoint fit/evaluation split.
    #
    # Everything here used to be computed on one undivided set: the isotonic
    # regression was fitted on `pairs`, then the PR sweep, the chosen
    # threshold, the hard-negative FPR and the ECE were all measured on those
    # same pairs. Isotonic regression is a flexible monotone fit that minimises
    # error against exactly those labels, so an in-sample ECE near zero is what
    # the method produces by construction — not evidence that the mapping
    # generalises. Splitting on VEHICLE rather than on pairs matters because
    # two crops of one car share paint, camera and plate, so a pair-level split
    # would still leak the identity being scored.
    fit_pairs, eval_pairs = split_by_identity(
        pairs, lambda p: (gallery[p.a_index].vehicle_id,
                          gallery[p.b_index].vehicle_id))
    report = build_report(fit_pairs, eval_pairs=eval_pairs, note=(
        "Calibrated on VeRi-776 gallery pairs (real vehicle crops): measures "
        "this embedder's confusability on that dataset — it does not "
        "transfer to other deployments (see eval/generalization notes). "
        "Fitted and scored on identity-disjoint halves; the reported "
        "threshold, sweep, hard-negative FPR and ECE are held-out."))
    save(report, f"calibration/artifacts/veri776{sfx}.json")
    if primary:
        save(report, "calibration/artifacts/veri776.json")
    rel = compute_reliability(eval_pairs, report.model)
    plot_reliability(rel.bins, rel.ece, f"veri_reliability{sfx}.png")
    plot_pr_sweep(report.sweep, report.chosen_threshold, f"veri_sweep{sfx}.png")

    hard = hardest_pairs(pairs, top=6)
    gallery_rows = []
    for p in hard:
        a = cv2.imread(str(gallery[p.a_index].path))
        b = cv2.imread(str(gallery[p.b_index].path))
        if a is not None and b is not None:
            gallery_rows.append((a, b, f"DIFFERENT vehicles, sim "
                                       f"{p.similarity:.3f} ({p.bucket})"))
    if gallery_rows:
        plot_pair_gallery(gallery_rows, f"veri_confusables{sfx}.png",
                          "Hardest real look-alike pairs (all different vehicles)")

    metrics, cases = run_ablation(
        query, q_emb, gallery, g_emb, threshold=report.chosen_threshold)
    raw = next(m for m in metrics if m.name == "raw")
    cas = next(m for m in metrics if m.name == "cascade")

    subsample_note = " *(subsampled `--quick` run — not headline numbers)*" if quick else ""
    n_hard = sum(p.hard_negative for p in pairs)
    lines = [
        f"## VeRi-776 [{EMBED_KEY}]: retrieval, calibration, ablation"
        f"{subsample_note}",
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
        f"![CMC](eval/figures/veri_cmc{sfx}.png)",
        "",
        "### Calibration on mined real hard negatives",
        "",
        f"{len(pairs)} pairs ({n_hard} hard negatives = same-color same-body "
        f"different-vehicle, mined by bucket). Calibration version "
        f"`isotonic-{report.model.version}`; ECE {rel.ece:.3f}. "
        f"{_threshold_note(report)} Hard-negative FPR at that threshold: "
        f"{report.hard_negative_fpr_at_threshold:.1%}.",
        "",
        f"![reliability](eval/figures/veri_reliability{sfx}.png)",
        f"![sweep](eval/figures/veri_sweep{sfx}.png)",
        f"![confusables](eval/figures/veri_confusables{sfx}.png)",
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
    def _pct(v: float) -> str:
        # An undefined rate must not render as a number a reader can quote.
        return "n/a" if v != v else f"{v:.1%}"

    for m in (raw, cas):
        r = m.row()
        lines.append(f"| {r['policy']} | {_pct(m.precision)} | {_pct(m.recall)} "
                     f"| {_pct(m.f1)} | {r['alerts']} | {r['false_positives']} "
                     f"| {r['review_rate']} |")
    fp_cut = (1 - cas.false_positives / raw.false_positives) if raw.false_positives else 0.0
    lines.append("")
    if not raw.alerts and not cas.alerts:
        # Say it outright rather than let a table of n/a imply a tie.
        lines.append(
            "**No alerts fired under either policy, so there is no delta to "
            "report.** Every top-1 similarity fell below the chosen threshold — "
            "the comparison did not run. This is a statement about the embedder, "
            "not about the cascade: an ImageNet-pretrained OSNet-x0_25 does not "
            "separate VeRi-776 identities well enough to reach an alerting "
            "threshold at all (see the retrieval table above). Re-run with "
            "VeRi-776-trained weights or the FastReID backbone to measure the "
            "cascade's contribution.")
    else:
        dp = cas.precision - raw.precision
        note = ""
        if raw.alerts and not raw.false_positives:
            # A zero-delta headline is misleading when the baseline had nothing
            # to get wrong; point at the sweep rather than let it stand alone.
            note = (" At this threshold the raw policy made no false positives "
                    "at all, so there was nothing for the cascade to remove — "
                    "see the sweep below for operating points where there is.")
        lines.append(
            f"**Delta: {_pct(dp) if dp == dp else 'n/a'} precision; {fp_cut:.0%} "
            f"of raw false positives eliminated by the attribute veto + "
            f"look-alike ambiguity refusal.**{note}")
    lines += _ablation_sweep_block(query, q_emb, gallery, g_emb,
                                   report.chosen_threshold, _pct)
    lines += [
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


def _threed_measured(s: dict) -> str:
    """Report a real 3D ablation run from scripts/ablate_3d_cityflow.py.

    Unlike the VeRi ablation's colour channel — dataset labels, i.e. a perfect
    classifier, so its delta is an upper bound — this channel is measured from
    actual reconstructions of actual crops. The number here is what the
    pipeline produces, not a ceiling. A null result is therefore a real result
    and is reported as one.
    """
    rows = s["rows"]
    order = ["2D only (colour)", "+ 3D geometry"]
    body = []
    for name in order:
        r = rows.get(name)
        if not r:
            continue
        pct = lambda v: "n/a" if v != v else f"{v:.1%}"       # noqa: E731
        body.append(
            f"| {name} | {pct(r['precision'])} | {pct(r['recall'])} | "
            f"{r['alerts']} | {r['false_positives']} | {r['review_rate']} | "
            f"{r.get('attribute_vetoes', 0)} |")

    a, b = rows.get(order[0], {}), rows.get(order[1], {})
    d_fp = (a.get("false_positives", 0) - b.get("false_positives", 0))
    d_p = (b.get("precision", 0) - a.get("precision", 0))
    tp_a = a.get("alerts", 0) - a.get("false_positives", 0)
    tp_b = b.get("alerts", 0) - b.get("false_positives", 0)
    extra_vetoes = b.get("attribute_vetoes", 0) - a.get("attribute_vetoes", 0)
    yield_pct = 100 * s["crops_with_geometry"] / max(s["crops"], 1)

    if not d_p and not d_fp:
        verdict = ("**No measurable difference.** The geometry channel changed "
                   "neither precision nor the false-positive count here.")
    elif d_p < 0 or (d_fp <= 0 and tp_b < tp_a):
        # State harm as harm. A signed delta alone invites reading a negative
        # result as a rounding artifact.
        verdict = (
            f"**The geometry channel made this worse: {d_p:+.1%} precision, "
            f"{d_fp:+d} false positives removed.** It fired {extra_vetoes} "
            f"additional attribute vetoes and those vetoes cost "
            f"{tp_a - tp_b} correct matches while removing no incorrect ones — "
            f"it is vetoing true positives. Single-crop proportion buckets "
            f"(`body_profile`, `length_class`) are evidently noisy enough on "
            f"real CCTV crops that they contradict on same-vehicle pairs about "
            f"as often as on different-vehicle ones, which is precisely the "
            f"failure mode that makes an attribute channel harmful rather than "
            f"merely useless. On this evidence the 3D channel should not be "
            f"enabled for identification decisions.")
    else:
        verdict = (f"**Delta: {d_p:+.1%} precision, {d_fp:+d} false positives "
                   f"removed**, for {tp_a - tp_b:+d} correct matches lost.")

    return "\n".join([
        "## 3D-geometry ablation (car3d bridge) — MEASURED",
        "",
        f"Run by `scripts/ablate_3d_cityflow.py` on {s['scenario']} with the "
        f"`{s['embedder']}` embedder: {s['crops']} real ground-truth crops over "
        f"{s['vehicles_kept']} vehicles, threshold {s['threshold']:.3f}. Both "
        "rows score IDENTICAL rankings; the only difference is whether "
        "`car3d/geometry.py`'s proportion attributes reach the cascade's "
        "contradiction check.",
        "",
        f"**The binding constraint is reconstruction yield: usable geometry on "
        f"{s['crops_with_geometry']}/{s['crops']} crops ({yield_pct:.0f}%).** "
        "The rest fall below the 64px subject bar or produce a signature whose "
        "observed fraction is too low to trust — `signature_to_attrs` refuses "
        "to emit attributes measured mostly on the generative prior, which "
        "would launder a guess into evidence. On real CityFlow footage most "
        "vehicles are simply too small in frame to carry geometry, and that "
        "ceiling bounds anything this channel can contribute.",
        "",
        "| policy | precision | recall | alerts | false positives | reviews | attribute vetoes |",
        "|---|---|---|---|---|---|---|",
        *body,
        "",
        verdict,
        "",
        "Unlike the VeRi colour/body-type ablation — which uses dataset labels, "
        "i.e. a perfect attribute classifier, making its delta an upper bound — "
        "this channel is measured end to end from real reconstructions. There "
        "is no idealisation left in it, so a null result here is a finding "
        "about the feature rather than an artifact of the setup.",
        "",
        "**What was done about it:** the geometry channel no longer feeds "
        "identification by default. Reconstruction, export, provenance and the "
        "operator dossier are unchanged — what is gated is the promotion of "
        "geometry to evidence (profile attributes and the render-based "
        "shortlist verifier). Re-enable per deployment, with numbers, via "
        "`--3d-identification` / `EYES_ENABLE_3D_IDENTIFICATION=1`. A feature "
        "that measures as harmful should be off, not documented as a caveat "
        "and left running.",
        "",
    ])


def threed_section() -> str:
    """The 3D-geometry ablation is doubly gated and says so."""
    try:
        from cargen import backends
        backends.build_prior_generator()
        backend_ok = True
    except Exception:
        backend_ok = False
    from datasets.veri776 import Veri776

    measured = _load_json(Path("data/ablation3d_s01_fastreid.json")) or \
        _load_json(Path("data/ablation3d_s01.json"))
    if measured:
        return _threed_measured(measured)

    if Veri776.exists() and backend_ok:
        # Both gates are open — but the row they were gating does not exist:
        # veri_section calls run_ablation without the `extra_attrs` hook, so
        # nothing merges 3D geometry into the contradiction check. Returning ""
        # here used to delete the whole section at precisely the moment its
        # prerequisites were satisfied, turning "PENDING, here is the plan"
        # into silence. Say where it actually stands instead.
        return "\n".join([
            "## 3D-geometry ablation (car3d bridge)",
            "",
            "**NOT RUN — prerequisites met, experiment not wired up.** VeRi-776 "
            "is present and a real cargen image-to-3D backend (SF3D) builds and "
            "reconstructs; what is missing is the wiring, not the capability: "
            "`veri_section` calls `run_ablation` without the `extra_attrs` hook, "
            "so no 3D-derived attribute reaches the cascade's contradiction "
            "check.",
            "",
            "The blocker is cost, and it should be stated rather than implied. "
            "A single SF3D reconstruction measured on an RTX 3080 Ti takes "
            "**~98 s** (120k splats). VeRi-776's evaluation split is 1678 query "
            "+ 11579 gallery images, so reconstructing all of them is ~350 GPU-"
            "hours — not a thing this harness can do inline. A defensible "
            "version subsamples cross-view pairs, and choosing that subsample "
            "is an experimental-design decision, not a configuration default; "
            "until someone makes it deliberately, this section reports nothing "
            "rather than a number whose sampling nobody chose.",
            "",
            "Planned, code in place (`eval/ablation.py` `extra_attrs` hook + "
            "`car3d/geometry.py`): (a) cascade precision with vs without "
            "3D-derived proportion attributes on cross-VIEW query/gallery "
            "pairs, where 2D ReID degrades most; (b) geometry error vs number "
            "of fused sightings, to substantiate — or refute — the claim that "
            "the model firms up with corroboration.",
            "",
        ])
    missing = []
    if not Veri776.exists():
        missing.append("VeRi-776 (manual request, see DATASETS.md)")
    if not backend_ok:
        missing.append("a real cargen image-to-3D backend (SF3D/TRELLIS — the "
                       "stub prior emits a procedural sedan whose geometry is "
                       "meaningless for identification)")
    return "\n".join([
        "## 3D-geometry ablation (car3d bridge)",
        "",
        "**PENDING — requires " + " and ".join(missing) + ".**",
        "",
        "Planned, code in place (`eval/ablation.py` `extra_attrs` hook + "
        "`car3d/geometry.py`): (a) cascade precision with vs without "
        "3D-derived proportion attributes on cross-VIEW query/gallery pairs, "
        "where 2D ReID degrades most; (b) geometry error vs number of fused "
        "sightings, to substantiate — or refute — the claim that the model "
        "firms up with corroboration. Both will be reported even if the gain "
        "is small or absent.",
        "",
    ])


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


# ------------------------------------------- embedder / look-alike block

_EMBEDDER_LABELS = {
    "osnet": "OSNet-x0_25 (ImageNet-pretrained, never vehicle-finetuned)",
    "fastreid": "FastReID SBS(R50-ibn), VeRi-776-finetuned",
}


def _load_json(path: Path):
    import json

    return json.loads(path.read_text()) if path.is_file() else None


def _separability_commentary(sep_by_key: dict[str, dict]) -> str:
    """Describe the separability table using only rows that are in it.

    This paragraph used to be a fixed string quoting both embedders' numbers.
    With only the default backbone measured — the usual case, since FastReID
    needs a 198 MB checkpoint — RESULTS.md asserted a finetuned figure of
    "~0.64" that no run had produced. Prose narrating uncomputed numbers is
    the exact failure this document exists to avoid, so it is generated.
    """
    osnet = sep_by_key.get("osnet")
    fast = sep_by_key.get("fastreid")
    parts = []
    ceilings = [f["same_camera_auc"] for f in sep_by_key.values()]
    if ceilings:
        multi = len(ceilings) > 1
        which = "Both embedders score" if multi else "The embedder scores"
        span = (f"{min(ceilings):.2f}–{max(ceilings):.2f}" if multi
                else f"{ceilings[0]:.2f}")
        parts.append(
            f"(1) {which} {span} on the same-camera ceiling, so crops, "
            f"preprocessing and model loading are all sound — whatever is "
            f"failing is not the plumbing.")
    if osnet:
        parts.append(
            f"(2) The ImageNet-pretrained default sits at {osnet['fair_auc']:.2f} "
            f"cross-camera, i.e. near chance: it encodes viewpoint-conditioned "
            f"appearance, not vehicle identity.")
    if fast:
        parts.append(
            f"The vehicle-finetuned checkpoint lifts that to "
            f"{fast['fair_auc']:.2f}, a real gain but far from solved.")
    else:
        parts.append(
            "The vehicle-finetuned comparison is absent from this run: no "
            "FastReID summary was generated (see README for the checkpoint), "
            "so no claim is made about what finetuning would buy.")
    advs = [f["adversarial_auc"] for f in sep_by_key.values()]
    if advs:
        scope = "for BOTH" if len(advs) > 1 else "for the measured embedder"
        span = (f"{min(advs):.2f}–{max(advs):.2f}" if len(advs) > 1
                else f"{advs[0]:.2f}")
        parts.append(
            f"(3) The adversarial column stays at {span} "
            f"{scope} — on the most-confusable tail, appearance alone remains "
            f"worse than a coin flip. That residue is the genuine look-alike "
            f"problem, and it is an argument for refusing to individuate on "
            f"appearance rather than for buying a better model.")
    return "Reading the table. " + " ".join(parts)


def embedder_section() -> str:
    """Cross-camera separability, retrieval and the two-sided operational
    metric, per embedder.

    Reads the JSON summaries written by scripts/{inspect_cityflow_calibration,
    eval_cityflow_retrieval,analyze_cityflow_recognizability}.py. Those runs
    are expensive (minutes of CPU embedding per embedder) so they are not
    re-executed here; absent summaries produce a PENDING block exactly like a
    missing dataset, and no number is ever carried over from a previous run.
    """
    rows_sep, rows_retr, rows_ops = [], [], []
    sep_by_key: dict[str, dict] = {}
    for key, label in _EMBEDDER_LABELS.items():
        tag = "s01" if key == "osnet" else f"s01_{key}"
        sep = _load_json(Path(f"data/separability_{tag}.json"))
        retr = _load_json(Path(f"data/retrieval_{tag}.json"))
        ops = _load_json(Path(f"data/recognizability_{tag}.json"))

        if sep:
            f = sep["fair_protocol"]
            sep_by_key[key] = f
            rows_sep.append(
                f"| {label} | {f['fair_auc']:.3f} | {f['adversarial_auc']:.3f} | "
                f"{f['same_camera_auc']:.3f} | {f['difficulty_gap']:.3f} |")
        if retr:
            rows_retr.append(
                f"| {label} | {retr['rank1'] * 100:.1f}% | "
                f"{retr['rank5'] * 100:.1f}% | {retr['rank10'] * 100:.1f}% | "
                f"{retr['mAP'] * 100:.1f}% | {retr['queries']} |")
        if ops and isinstance(ops, dict) and "recall" in ops:
            rec, imp = ops["recall"], ops["impostors"]
            rate = rec["rate"]
            fpr = imp["false_positive_rate"]
            # Genuine passages the transit veto actively threw out. Counted
            # from the traces because it is a cost of the space-time signal,
            # not a neutral "no decision", and belongs beside its benefit.
            rejected = sum(1 for v in ops.get("per_vehicle", [])
                           for t in v.get("trace", [])
                           if t.get("verdict") == "rejected")
            evaluated = rec["evaluated_passages"] or 1
            sc = imp["same_color"]
            rows_ops.append(
                f"| {label} | {rec['proposed']}/{rec['evaluated_passages']} "
                f"({0.0 if rate is None else rate * 100:.1f}%) | "
                f"{imp['impostor_proposals']}/{imp['impostor_evaluations']} "
                f"({0.0 if fpr is None else fpr * 100:.1f}%) | "
                f"{sc['proposals']}/{sc['evaluations']} "
                f"({0.0 if sc['rate'] is None else sc['rate'] * 100:.1f}%) | "
                f"{rejected}/{rec['evaluated_passages']} "
                f"({100 * rejected / evaluated:.0f}%) |")

    if not (rows_sep or rows_retr or rows_ops):
        return _pending(
            "CityFlow: embedder separability, retrieval, recall/FPR",
            "These runs cost minutes of embedding per embedder, so they are "
            "not executed inline. Run `scripts/inspect_cityflow_calibration.py`, "
            "`scripts/eval_cityflow_retrieval.py` and "
            "`scripts/analyze_cityflow_recognizability.py` (per embedder) to "
            "produce the summaries this section reads.",
            headline="per-embedder summaries not generated yet",
            see_datasets=False)

    out = [
        "## CityFlow S01: can the embedder do cross-camera at all?",
        "",
        "Real ground-truth crops from S01 (795 crops, 95 vehicles, 5 cameras). "
        "This block exists because the live console never proposed a single "
        "cross-camera match on real data, and the first diagnosis of that "
        "(\"the embedder ranks different cars above same cars\") turned out to "
        "rest on a biased sample — see the two negative-selection protocols below.",
        "",
    ]
    if rows_sep:
        out += [
            "### Separability: fair vs adversarial negative selection",
            "",
            "**fair** = cross-camera positives vs negatives sampled *uniformly* "
            "within an appearance bucket. **adversarial** = the same positives "
            "vs *top-k most similar* mined negatives (`eval/hard_negatives.py`, "
            "which is the right sample for fitting a calibration curve and the "
            "wrong one for judging an embedder). **same-camera** = positives "
            "from the same camera — near-duplicate frames, so this is a sanity "
            "ceiling, not a re-identification.",
            "",
            "| embedder | fair AUC | adversarial AUC | same-camera ceiling | gap |",
            "|---|---|---|---|---|",
            *rows_sep,
            "",
            _separability_commentary(sep_by_key),
            "",
            "An earlier reading of this data claimed the embedder ranked "
            "different cars *above* same cars (AUC 0.102). That figure came "
            "from the adversarial column and was reported as if it were the "
            "fair one; the honest version is the table above.",
            "",
        ]
    if rows_retr:
        out += [
            "### Retrieval (standard protocol, same-camera-same-id excluded)",
            "",
            "Scored by the same `eval/retrieval.py:evaluate_retrieval` used for "
            "the VeRi-776 table above, so the two are directly comparable.",
            "",
            "| embedder | Rank-1 | Rank-5 | Rank-10 | mAP | queries |",
            "|---|---|---|---|---|---|",
            *rows_retr,
            "",
        ]
    if rows_ops:
        out += [
            "### Two-sided operational metric",
            "",
            "Recall and false-positive rate must be read together: a system that "
            "proposes nothing scores a perfect FPR, and one that proposes "
            "everything scores perfect recall. Impostors are sampled 2:1 toward "
            "the *same* estimated colour, because that is the hard case.",
            "",
            "| embedder | recall (genuine passages) | FPR (impostor passages) | same-colour impostors | genuine passages vetoed |",
            "|---|---|---|---|---|",
            *rows_ops,
            "",
            "**Measured with space-time evidence enabled** "
            "(`W_TRANSIT_CONSISTENT`). Before it, both embedders sat at the "
            "same degenerate corner — 0/284 recall and 0/1032 FPR — despite a "
            "~3x retrieval gap between them, which is what identified the "
            "cascade's arithmetic rather than the embedder as the binding "
            "constraint: colour (`W_CLASS_ATTRS` 0.20) plus appearance "
            "(`W_REID_MAX` 0.30) could not clear `LIKELY_THRESHOLD` (0.45) on "
            "an honestly-calibrated signal, however good the embedding got.",
            "",
            "Space-time breaks that because it is independent of pixels: it "
            "narrows 95 candidate vehicles to a median of 4 "
            "(P(same | in window) ≈ 0.28 against a ≈0.011 prior). **Every "
            "proposal is a `candidate` verdict** — distinctiveness stays at "
            "0.222, below the 0.30 floor, so the system offers a narrowed set "
            "for review and never asserts an individual. Four co-plausible "
            "vehicles is a set, and it says so.",
            "",
            "Three defects were then found and fixed. The table below is a "
            "**recorded history, not a recomputed result**: these figures come "
            "from an earlier FastReID run and are reproduced verbatim so the "
            "sequence of fixes stays auditable. Unlike every other table in "
            "this document, re-running `python -m eval.run` does not "
            "regenerate it — re-measure with the FastReID checkpoint to "
            "confirm the numbers still hold.",
            "",
            "| stage | recall | FPR | same-colour FPR |",
            "|---|---|---|---|",
            "| space-time only | 17.6% | 1.6% | 2.1% |",
            "| + transit-veto fix | 57.7% | 2.1% | 2.9% |",
            "| + base-rate calibration | 52.1% | 0.5% | 0.7% |",
            "",
            "The **transit-veto fix** was the large one. `to_road_graph` was "
            "discarding every transition with a negative exit→enter gap as "
            "noise, but those are real vehicles reaching the next camera "
            "before leaving the previous one's field of view — 74% of all "
            "ground-truth transitions in S01. `min_s` was therefore built "
            "from the non-overlapping minority, came out far too high, and "
            "vetoed as impossible the very hops it had excluded (99% of "
            "observed vetoes were passages overlapping in wall time). Keeping "
            "them tripled recall for +0.8pt of same-colour FPR.",
            "",
            "The **calibration fix** traded recall for precision, and is kept "
            "on those terms: fitting P(same) on `mine_pairs`' top-k sample "
            "made look-alikes the bulk of the data, inverted the "
            "similarity/identity relationship, and collapsed the isotonic fit "
            "to a constant (~0.5 across the whole operating range) — an "
            "appearance signal contributing the same value to every "
            "candidate. Refitting on base-rate negatives "
            "(`eval/separability.py:natural_population_pairs`) restores a "
            "monotone curve, and the true:false proposal ratio improves from "
            "164:22 to 148:5. Hard mining keeps its reporting role above; it "
            "simply must not define P(same).",
            "",
            "Note what this table does NOT show: the two embedders land within "
            "noise of each other on every operational column, despite the ~3x "
            "retrieval gap in the section above. Both calibration curves map "
            "their typical cross-camera similarity to p≈0.5, and `W_REID_MAX` "
            "caps the result at 0.30, so each contributes ≈0.15 and the "
            "outcome is decided by colour and space-time — neither of which "
            "depends on the embedder. A better embedding is real (see "
            "retrieval) and currently unspendable.",
            "",
        ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="subsample for a fast smoke run (not headline numbers)")
    parser.add_argument("--embedder", choices=sorted(_EMBEDDERS), default="osnet",
                        help="backbone for the VeRi block (default: osnet). "
                             "fastreid needs models/veri_sbs_R50-ibn.pth")
    args = parser.parse_args()
    _select_embedder(args.embedder)

    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        "# RESULTS",
        "",
        f"*Regenerated by `python -m eval.run --embedder {args.embedder}` on "
        f"{stamp}. Every section is computed from data actually present on "
        f"disk; missing datasets produce PENDING sections, never substituted "
        f"numbers. Nothing here claims production accuracy — see the Limits "
        f"section of the README.*",
        "",
        veri_section(args.quick, available_embedders(args.embedder)),
        vehicleid_section(args.quick),
        cityflow_section(),
        embedder_section(),
        threed_section(),
        synthetic_section(),
    ]
    RESULTS_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
