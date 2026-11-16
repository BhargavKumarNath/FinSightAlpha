"""Figures from saved results only (dark, site-matched style)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/phase3")
FIG = OUT / "figures"
BG, FG = "#0a0a0a", "#d8d8d0"
INK, ACC, GOLD, ROSE, MUTE = "#e8e6d4", "#7fd4ff", "#ffc83d", "#ff7b7b", "#8a8a8a"
plt.rcParams.update({
    "font.size": 10, "font.family": "DejaVu Sans Mono", "figure.dpi": 140,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": "#3a3a3a", "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": "#9a9a9a", "ytick.color": "#9a9a9a", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#161616", "legend.labelcolor": FG,
})
MODES = [("dense", "dense only", MUTE), ("sparse", "BM25 only", ROSE),
         ("hybrid", "hybrid (RRF)", ACC), ("hybrid_rerank", "hybrid + rerank", GOLD)]


def retrieval_fig() -> None:
    s = json.loads((OUT / "retrieval_summary.json").read_text())["metrics"]
    ks = (1, 3, 5, 8)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, grp, title in ((axes[0], "single", "single-hop (n=35)"), (axes[1], "multi", "two-passage (n=12)")):
        for mode, label, col in MODES:
            m = np.array([s[f"test/{grp}/{mode}/all_hops@{k}"] for k in ks])
            ax.errorbar(ks, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], color=col,
                        marker="o", ms=4, lw=1.6, capsize=2, label=label, zorder=3)
        ax.set_title(title, color=FG, fontsize=10, loc="left")
        ax.set_xlabel("top-k chunks given to the reader")
        ax.set_xticks(ks)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("questions with all gold passages")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("Retrieval, held-out test questions, 95% intervals", color=FG, fontsize=10, x=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "p3_01_retrieval_recall.png")
    plt.close(fig)


def answers_fig() -> None:
    r = json.loads((OUT / "results.json").read_text())["reader"]["hybrid_rerank"]
    labels, vals = [], []
    for key, lab in (("single", "single-hop"), ("multi_same", "two passages,\nsame filing"),
                     ("multi_cross", "two companies")):
        m, lo, hi, n = r["by_type"][key]
        labels.append(f"{lab}\n(n={n})")
        vals.append((m, lo, hi))
    m, lo, hi = r["unanswerable_declined"]
    labels.append(f"unanswerable:\ndeclined (n={r['n_unanswerable']})")
    vals.append((m, lo, hi))
    v = np.array(vals)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    cols = [ACC, GOLD, GOLD, "#8ee6a6"]
    ax.bar(range(len(v)), v[:, 0], color=cols, width=0.58, zorder=3)
    ax.errorbar(range(len(v)), v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none",
                ecolor=INK, capsize=3, lw=1.2, zorder=4)
    for i, x in enumerate(v[:, 0]):
        ax.text(i, 0.03, f"{x:.0%}", ha="center", color=BG, fontsize=10, fontweight="bold", zorder=5)
    ax.set_xticks(range(len(v)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("correct, or declined")
    ax.set_title("Answer quality, hybrid + rerank retrieval, 65 held-out questions", color=FG, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "p3_02_answer_quality.png")
    plt.close(fig)


if __name__ == "__main__":
    retrieval_fig()
    answers_fig()
    print("ok")
