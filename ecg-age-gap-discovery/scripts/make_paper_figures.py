#!/usr/bin/env python3
"""Generate the two main-text figures at publication size.

Numbers are the finalised results recorded in docs/reproducibility.md. They are
written literally here rather than re-derived, so a figure cannot silently drift
from the tables in the paper: changing a number requires changing it in both
places deliberately.

Sized for a NeurIPS workshop page and legible at 100% print scale.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

GREY, BLUE, RED, PURPLE = "#BBBBBB", "#4C72B0", "#C44E52", "#7B6B9E"


def figure_1_dissolution() -> None:
    """Variance decomposition across the three analyses, and the effect dying."""
    # Short tick labels; the configurations are spelled out in the caption.
    # Long multi-line ticks collided with the legend.
    labels = ["A", "B", "C"]
    demographics = np.array([29.3, 20.7, 20.4])
    intervals = np.array([3.9, 3.5, 14.6])
    unexplained = np.array([66.8, 75.8, 65.0])
    # Two superclasses, because the finding does not simply shrink across the
    # three analyses - it moves from one label to another and only then dies.
    mi_delta = np.array([0.0155, 0.0062, 0.0010])
    mi_significant = [True, False, False]
    sttc_delta = np.array([0.0040, 0.0165, 0.0006])
    sttc_significant = [False, True, False]

    figure, axes = plt.subplots(1, 2, figsize=(6.6, 2.6),
                                gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(3)

    axes[0].bar(x, demographics, color=GREY, label="demographics (age, sex)")
    axes[0].bar(x, intervals, bottom=demographics, color=BLUE,
                label="known ECG measurements")
    axes[0].bar(x, unexplained, bottom=demographics + intervals, color=RED,
                label="unexplained")
    for i, value in enumerate(intervals):
        axes[0].text(i, demographics[i] + value / 2 + 1.5, f"{value:.1f}%",
                     ha="center", va="center", color="white", fontsize=7.5,
                     fontweight="bold")
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("analysis configuration")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("share of age-gap variance (%)")
    axes[0].set_title("Where the age gap goes", loc="left")
    handles, legend_labels = axes[0].get_legend_handles_labels()

    # Significance is carried by saturation, not hue, so the two superclasses
    # stay distinguishable in greyscale print.
    width = 0.34
    for offset, delta, significant, colour, name in (
        (-width / 2, mi_delta, mi_significant, RED, "infarction"),
        (+width / 2, sttc_delta, sttc_significant, PURPLE, "ST/T change"),
    ):
        axes[1].bar(x + offset, delta, width=width, label=name,
                    color=[colour if s else "white" for s in significant],
                    edgecolor=colour, linewidth=1.0)
        for i, value in enumerate(delta):
            if significant[i]:
                axes[1].text(i + offset, value + 0.0006, "*", ha="center",
                             fontsize=9, color=colour, fontweight="bold")
    axes[1].axhline(0.020, color="0.3", linestyle="--", linewidth=0.9)
    axes[1].text(2.45, 0.0206, "pre-registered\nrelevance threshold", fontsize=6.2,
                 color="0.3", ha="right", va="bottom")
    axes[1].set_xticks(x, ["A", "B", "C"])
    axes[1].set_ylim(0, 0.027)
    axes[1].set_ylabel(r"$\Delta$AUC over known measurements")
    axes[1].set_title("The effect moves, then dies", loc="left")
    axes[1].legend(loc="upper left", frameon=False, fontsize=6.2,
                   handlelength=1.1, borderpad=0.1, labelspacing=0.25)
    axes[1].text(1, -0.17, "filled = significant after correction",
                 transform=axes[1].get_xaxis_transform(), ha="center",
                 fontsize=5.8, color="0.45")

    figure.legend(handles, legend_labels, loc="lower center", ncol=3,
                  frameon=False, bbox_to_anchor=(0.5, -0.06))
    figure.tight_layout()
    figure.savefig(OUT / "fig1_dissolution.pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {OUT / 'fig1_dissolution.pdf'}")


def figure_2_occlusion() -> None:
    """Causal occlusion per segment in both cohorts, and every model's P interval."""
    segments = ["P wave", "QRS", "T wave"]
    ptbxl = np.array([0.582, 7.348, 0.617])
    chapman = np.array([1.043, 5.010, 1.006])
    ptbxl_err = np.array([[0.114, 1.633, 0.113], [0.151, 1.123, 0.143]])
    chapman_err = np.array([[0.392, 1.331, 0.281], [0.285, 1.326, 0.312]])

    figure, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    x = np.arange(3)
    width = 0.36

    axes[0].bar(x - width / 2, ptbxl, width, yerr=ptbxl_err, capsize=2.5,
                color=BLUE, label="PTB-XL (Germany)", error_kw={"linewidth": 0.8})
    axes[0].bar(x + width / 2, chapman, width, yerr=chapman_err, capsize=2.5,
                color=RED, label="Chapman (China)", error_kw={"linewidth": 0.8})
    axes[0].axhline(0, color="0.3", linewidth=0.8)
    axes[0].set_xticks(x, segments)
    axes[0].set_ylabel(r"$\Delta$ MAE vs matched control (years)")
    axes[0].set_title("Occlusion: what the model needs", loc="left")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].annotate("inflated by\ndistribution shift", xy=(1.28, 5.1),
                     xytext=(1.62, 6.9), fontsize=6, color="0.45", ha="left",
                     arrowprops=dict(arrowstyle="-", color="0.6", linewidth=0.6))
    axes[0].set_ylim(0, 10.2)

    names = ["PTB-XL s0", "PTB-XL s1", "PTB-XL s2",
             "Chapman s0", "Chapman s1", "Chapman s2", "transfer"]
    values = np.array([0.468, 0.733, 0.546, 0.651, 1.149, 1.328, 0.510])
    lows = np.array([0.189, 0.470, 0.291, 0.383, 0.818, 1.038, 0.040])
    highs = np.array([0.747, 0.996, 0.801, 0.918, 1.480, 1.618, 0.980])
    colours = [BLUE] * 3 + [RED] * 3 + [PURPLE]

    for i, (value, low, high, colour) in enumerate(zip(values, lows, highs, colours)):
        axes[1].plot([low, high], [i, i], color=colour, linewidth=1.8)
        axes[1].plot(value, i, "o", color=colour, markersize=4)
    axes[1].axvline(0, color="0.3", linestyle="--", linewidth=0.9)
    axes[1].set_yticks(range(len(names)), names)
    axes[1].invert_yaxis()
    axes[1].set_xlim(-0.15, 1.75)
    axes[1].set_xlabel(r"P-wave occlusion, $\Delta$ MAE (years)")
    axes[1].set_title("Every model clears zero", loc="left")

    figure.tight_layout()
    figure.savefig(OUT / "fig2_occlusion.pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {OUT / 'fig2_occlusion.pdf'}")


if __name__ == "__main__":
    figure_1_dissolution()
    figure_2_occlusion()
