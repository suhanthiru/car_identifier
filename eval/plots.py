"""Matplotlib figures for RESULTS.md. Headless (Agg), files only."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("eval/figures")


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_cmc(cmc: np.ndarray, title: str, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ranks = np.arange(1, len(cmc) + 1)
    ax.plot(ranks, cmc * 100, lw=2)
    ax.set_xlabel("rank")
    ax.set_ylabel("matching rate (%)")
    ax.set_title(title)
    ax.set_xlim(1, len(cmc))
    ax.grid(alpha=0.3)
    return _save(fig, name)


def plot_reliability(bins, ece: float, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
    xs = [b.mean_predicted for b in bins]
    ys = [b.empirical_accuracy for b in bins]
    sizes = [max(20, min(200, b.count)) for b in bins]
    ax.scatter(xs, ys, s=sizes, alpha=0.8, label="bins (size = pair count)")
    ax.set_xlabel("predicted P(same vehicle)")
    ax.set_ylabel("empirical fraction same")
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, name)


def plot_pr_sweep(sweep, chosen: float, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ts = [p.threshold for p in sweep]
    ax.plot(ts, [p.precision for p in sweep], label="precision", lw=2)
    ax.plot(ts, [p.recall for p in sweep], label="recall", lw=2)
    ax.plot(ts, [p.f1 for p in sweep], label="F1", lw=1, ls=":")
    ax.axvline(chosen, color="gray", ls="--", lw=1, label=f"chosen t={chosen:.3f}")
    ax.set_xlabel("similarity threshold")
    ax.set_title("Alert-threshold sweep on mined real pairs")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, name)


def plot_pair_gallery(pair_images: list[tuple[np.ndarray, np.ndarray, str]],
                      name: str, title: str) -> Path:
    """Side-by-side crops per row: (left, right, caption). BGR input."""
    rows = len(pair_images)
    fig, axes = plt.subplots(rows, 2, figsize=(4.6, 1.9 * rows))
    if rows == 1:
        axes = np.array([axes])
    for r, (a, b, caption) in enumerate(pair_images):
        for c, img in enumerate((a, b)):
            axes[r, c].imshow(img[:, :, ::-1])
            axes[r, c].axis("off")
        axes[r, 0].set_title(caption, fontsize=7, loc="left")
    fig.suptitle(title, fontsize=10)
    return _save(fig, name)


def plot_transit_hist(elapsed: list[float], vetoed_below: float, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.hist(elapsed, bins=40, alpha=0.85)
    ax.axvline(vetoed_below, color="red", ls="--", lw=1.5,
               label=f"veto boundary ({vetoed_below:.0f}s)")
    ax.set_xlabel("real cross-camera transition time (s)")
    ax.set_ylabel("count")
    ax.set_title("Ground-truth transitions vs transit-veto boundary")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, name)


def plot_corroboration_curves(sightings: list[int], noisy_or_vals: list[float],
                              capped_vals: list[float], name: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(sightings, noisy_or_vals, "o-", label="noisy-OR (wrong: assumes independence)")
    ax.plot(sightings, capped_vals, "s-", label="capped-additive (ours)")
    ax.set_xlabel("number of correlated look-alike sightings")
    ax.set_ylabel("fused belief")
    ax.set_ylim(0, 1.02)
    ax.set_title("Correlated-evidence fusion: overshoot vs cap")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, name)
