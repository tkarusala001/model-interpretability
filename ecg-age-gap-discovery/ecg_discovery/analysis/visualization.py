"""Plots intended to be handed to a cardiologist, not admired by an ML audience.

The design constraint throughout is that a domain expert must be able to
*disagree* with what they see. That means attribution is always shown against
the actual waveform with the detected fiducial points marked, so a reader can
check both the model's evidence and the delineation it was aggregated over - a
heatmap floating free of the trace is unfalsifiable.

Colour is used sparingly and consistently: one hue per cardiac segment
throughout the project, and attribution magnitude shown by opacity rather than
by a rainbow colormap, since opacity has an unambiguous ordering while a rainbow
does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # figures are written to disk, never displayed interactively

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from ecg_discovery.interpretability.fiducial_attribution import (
    SEGMENT_NAMES,
    FiducialAttribution,
)
from ecg_discovery.signal_processing.wave_delineation import BeatDelineation

__all__ = [
    "SEGMENT_COLOURS",
    "plot_attribution_overlay",
    "plot_segment_summary",
    "plot_age_gap_scatter",
    "plot_residual_decomposition",
    "plot_diagnostic_link",
]

#: One colour per segment, used consistently in every figure.
SEGMENT_COLOURS = {
    "P": "#4C72B0",
    "QRS": "#C44E52",
    "T": "#55A868",
    "other": "#BBBBBB",
}


def _segment_spans(beats: Sequence[BeatDelineation]) -> list[tuple[int, int, str]]:
    """Flatten delineated beats into ``(start, end, segment)`` spans."""
    spans: list[tuple[int, int, str]] = []
    for beat in beats:
        if beat.p_wave is not None:
            spans.append((beat.p_wave.onset, beat.p_wave.offset, "P"))
        spans.append((beat.qrs.onset, beat.qrs.offset, "QRS"))
        if beat.t_wave is not None:
            spans.append((beat.t_wave.onset, beat.t_wave.offset, "T"))
    return spans


def plot_attribution_overlay(
    signal: np.ndarray,
    attributions: np.ndarray,
    beats: Sequence[BeatDelineation],
    lead_names: Sequence[str],
    sampling_rate_hz: float,
    leads: Sequence[str] = ("II", "V2", "V5"),
    time_range_s: tuple[float, float] | None = None,
    title: str | None = None,
    path: str | Path | None = None,
):
    """Draw attribution on the real ECG trace, annotated with fiducial points.

    Each selected lead is drawn as its actual waveform, with the line's opacity
    set by the magnitude of attribution at that sample - so the eye follows the
    model's evidence along the trace it belongs to. Detected P, QRS and T
    segments are shaded behind, which is what lets a reader see whether the
    aggregation was applied to a sensible delineation.

    Parameters
    ----------
    signal:
        ``(n_leads, n_samples)`` waveform in millivolts.
    attributions:
        ``(n_leads, n_samples)`` per-sample attribution for the same recording.
    beats:
        Delineated beats, for the shaded segments and fiducial marks.
    leads:
        Which leads to draw. Three is usually right for a printed figure.
    time_range_s:
        Seconds to display. Defaults to a window around the second beat, which
        shows morphology legibly; the full ten seconds compresses each complex
        into an unreadable spike.

    Returns
    -------
    matplotlib.figure.Figure
    """
    lead_names = list(lead_names)
    missing = [lead for lead in leads if lead not in lead_names]
    if missing:
        raise KeyError(f"unknown lead(s) {missing}; available: {tuple(lead_names)}")

    n_samples = signal.shape[1]
    if time_range_s is None:
        if len(beats) >= 2:
            centre = beats[1].r_peak
            half = int(1.2 * sampling_rate_hz)
            start = max(centre - half, 0)
            end = min(centre + half, n_samples)
        else:
            start, end = 0, min(int(3 * sampling_rate_hz), n_samples)
    else:
        start = max(int(time_range_s[0] * sampling_rate_hz), 0)
        end = min(int(time_range_s[1] * sampling_rate_hz), n_samples)

    time_ms = np.arange(start, end) / sampling_rate_hz * 1000.0
    # A shared attribution scale across leads, so opacity is comparable between
    # panels rather than each lead being normalised to its own maximum.
    scale = np.percentile(np.abs(attributions[:, start:end]), 99)
    scale = float(scale) if scale > 0 else 1.0

    figure, axes = plt.subplots(
        len(leads), 1, figsize=(11, 2.1 * len(leads)), sharex=True
    )
    axes = np.atleast_1d(axes)

    for axis, lead in zip(axes, leads):
        index = lead_names.index(lead)
        trace = signal[index, start:end]
        weight = np.clip(np.abs(attributions[index, start:end]) / scale, 0.0, 1.0)

        for onset, offset, segment in _segment_spans(beats):
            if offset < start or onset > end:
                continue
            axis.axvspan(
                max(onset, start) / sampling_rate_hz * 1000.0,
                min(offset, end) / sampling_rate_hz * 1000.0,
                color=SEGMENT_COLOURS[segment], alpha=0.13, linewidth=0,
            )

        # Faint full trace, so the waveform stays readable where attribution is
        # near zero and the coloured overlay disappears.
        axis.plot(time_ms, trace, color="0.75", linewidth=0.8, zorder=2)

        points = np.array([time_ms, trace]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colours = np.zeros((len(segments), 4))
        colours[:, 0] = 0.05                       # near-black line
        colours[:, 3] = weight[:-1]                # opacity carries magnitude
        axis.add_collection(
            LineCollection(segments, colors=colours, linewidths=1.9, zorder=3)
        )

        for beat in beats:
            if start <= beat.r_peak < end:
                axis.plot(
                    beat.r_peak / sampling_rate_hz * 1000.0,
                    signal[index, beat.r_peak],
                    marker="v", markersize=5,
                    color=SEGMENT_COLOURS["QRS"], zorder=4,
                )

        axis.set_ylabel(f"{lead}\n(mV)", fontsize=9)
        axis.margins(x=0)
        axis.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("time (ms)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SEGMENT_COLOURS[name], alpha=0.35)
        for name in ("P", "QRS", "T")
    ]
    axes[0].legend(
        handles, ["P wave", "QRS complex", "T wave"],
        loc="upper right", fontsize=8, frameon=False, ncol=3,
    )
    if title:
        figure.suptitle(title, fontsize=10)
    figure.text(
        0.01, 0.005,
        "line opacity = |attribution|; shading = detected segments",
        fontsize=7, color="0.4",
    )
    figure.tight_layout()

    if path is not None:
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
    return figure


def plot_segment_summary(
    result: FiducialAttribution,
    title: str | None = None,
    path: str | Path | None = None,
):
    """Share and density side by side, because neither alone is honest.

    The two panels exist to make the width confound visible rather than to be
    read separately: share is dominated by how wide a segment is, density
    corrects for it. Segment widths are printed on the share panel so a reader
    can see the confound directly instead of taking it on trust.
    """
    segments = [name for name in SEGMENT_NAMES if name != "other"]
    shares = [result.segment_share(name) for name in segments]
    densities = [result.segment_density(name) for name in segments]
    widths = {
        name: int(result.segment_samples[result.segment_names.index(name)])
        for name in segments
    }

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    colours = [SEGMENT_COLOURS[name] for name in segments]

    axes[0].bar(segments, shares, color=colours)
    axes[0].set_title("Share of total attribution", fontsize=10)
    axes[0].set_ylabel("fraction")
    for index, (name, value) in enumerate(zip(segments, shares)):
        axes[0].text(
            index, value, f"\n{widths[name]} samples",
            ha="center", va="bottom", fontsize=7, color="0.35",
        )

    axes[1].bar(segments, densities, color=colours)
    axes[1].set_title("Attribution density (per sample)", fontsize=10)
    axes[1].set_ylabel("mean |attribution|")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)

    figure.text(
        0.5, -0.04,
        "Left is confounded by segment width; right corrects for it. "
        "Read them together.",
        ha="center", fontsize=8, color="0.4",
    )
    if title:
        figure.suptitle(title, fontsize=10)
    figure.tight_layout()

    if path is not None:
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
    return figure


def plot_residual_decomposition(
    decomposition,
    title: str | None = None,
    path: str | Path | None = None,
):
    """Where the age gap's variance goes, and which known feature accounts for it.

    Left: the variance split for every explainer, stacked so the three parts add
    to the whole - demographics, known intervals beyond demographics, and what
    is left. All explainers are shown side by side, because reporting only the
    most favourable one is the failure mode this project is built to avoid.

    Right: the out-of-fold variance each known feature explains on its own.
    Univariate rather than a coefficient plot, because correlated intervals
    share credit arbitrarily in a joint fit and the resulting bar chart would be
    an artefact of collinearity rather than a fact about the data.

    Parameters
    ----------
    decomposition:
        A ``DecompositionResult`` from
        :mod:`ecg_discovery.validation.residual_decomposition`.
    """
    explainers = list(decomposition.explainers.values())
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    names = [explainer.model_name.replace("_", "\n") for explainer in explainers]
    demographics = [explainer.r2_baseline for explainer in explainers]
    intervals = [explainer.r2_incremental for explainer in explainers]
    unexplained = [explainer.unexplained_fraction for explainer in explainers]

    axes[0].bar(names, demographics, color="#BBBBBB", label="demographics (age, sex)")
    axes[0].bar(names, intervals, bottom=demographics, color="#4C72B0",
                label="known ECG intervals")
    axes[0].bar(
        names, unexplained,
        bottom=[d + i for d, i in zip(demographics, intervals)],
        color="#C44E52", label="unexplained",
    )
    for index, (d, i, u) in enumerate(zip(demographics, intervals, unexplained)):
        if i > 0.04:
            axes[0].text(index, d + i / 2, f"{i:.0%}", ha="center", va="center",
                         color="white", fontsize=9, fontweight="bold")
        if u > 0.04:
            axes[0].text(index, d + i + u / 2, f"{u:.0%}", ha="center", va="center",
                         color="white", fontsize=9, fontweight="bold")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("share of age-gap variance")
    axes[0].set_title("Age-gap variance decomposition", fontsize=10)
    # Below the axes: the bars fill the full 0-1 range, so any in-plot legend
    # would sit on top of the data.
    axes[0].legend(
        fontsize=8, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, -0.10), ncol=3,
    )

    best = decomposition.most_explanatory
    ordered = sorted(best.univariate_r2.items(), key=lambda item: item[1])
    axes[1].barh(
        [name.replace("_", " ") for name, _ in ordered],
        [value for _, value in ordered],
        color="#4C72B0",
    )
    axes[1].axvline(0, color="0.4", linewidth=0.8)
    axes[1].set_xlabel("out-of-fold R² above demographics")
    axes[1].set_title(f"Each known interval alone ({best.model_name})", fontsize=10)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)

    figure.text(
        0.5, -0.16,
        "The unexplained share is a CEILING on any discovery claim, not evidence "
        "for one:\nmeasurement noise and a small known-feature set both inflate it.",
        ha="center", fontsize=8, color="0.4",
    )
    if title:
        figure.suptitle(title, fontsize=11)
    figure.tight_layout()

    if path is not None:
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
    return figure


def plot_diagnostic_link(
    report,
    title: str | None = None,
    path: str | Path | None = None,
):
    """Baseline against augmented classifier performance, with uncertainty.

    Left: held-out AUC per diagnostic superclass, with and without the
    unexplained residual added. Right: the paired difference with its interval,
    which is the quantity the conclusion actually rests on - two overlapping
    absolute AUCs can still differ reliably when compared fold by fold.

    A zero line is drawn on the difference panel and intervals crossing it are
    greyed, so a null result reads as a null result at a glance rather than
    needing the caption to explain it away.

    Parameters
    ----------
    report:
        A ``DiagnosticLinkReport`` from
        :mod:`ecg_discovery.discovery_experiments.unexplained_residual_diagnostic_link`.
    """
    links = list(report.links.values())
    if not links:
        raise ValueError("the report contains no evaluated superclasses to plot")

    names = [link.superclass for link in links]
    positions = np.arange(len(links))
    figure, axes = plt.subplots(
        1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.25, 1]}
    )

    width = 0.38
    axes[0].bar(positions - width / 2, [link.auc_baseline for link in links],
                width, color="#BBBBBB", label="known intervals")
    axes[0].bar(positions + width / 2, [link.auc_augmented for link in links],
                width, color="#4C72B0", label="+ unexplained residual")
    axes[0].axhline(0.5, color="0.4", linestyle="--", linewidth=0.8)
    axes[0].text(len(links) - 0.5, 0.505, "chance", fontsize=7, color="0.4", ha="right")
    axes[0].set_xticks(positions, names)
    axes[0].set_ylim(0.4, 1.0)
    axes[0].set_ylabel("held-out AUC")
    axes[0].set_title("Diagnostic prediction", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")

    for index, link in enumerate(links):
        low, high = link.delta_auc_ci
        colour = "#C44E52" if link.improves else "0.6"
        axes[1].plot([low, high], [index, index], color=colour, linewidth=2)
        axes[1].plot(link.delta_auc, index, "o", color=colour, markersize=6)
    axes[1].axvline(0, color="0.3", linestyle="--", linewidth=1)
    axes[1].set_yticks(positions, names)
    axes[1].set_xlabel("change in AUC from adding the residual")
    axes[1].set_title(f"Paired difference ({report.correction})", fontsize=10)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)

    verdict = (
        "At least one superclass improves - a CANDIDATE for follow-up, not a "
        "validated marker."
        if report.any_improvement
        else "No superclass improves: the unexplained residual carries no "
             "independently verifiable signal here."
    )
    figure.text(0.5, -0.08, verdict, ha="center", fontsize=8, color="0.35")
    if title:
        figure.suptitle(title, fontsize=11)
    figure.tight_layout()

    if path is not None:
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
    return figure


def plot_age_gap_scatter(
    true_age: np.ndarray,
    predicted_age: np.ndarray,
    highlight: np.ndarray | None = None,
    title: str | None = None,
    path: str | Path | None = None,
):
    """Predicted against true age, with the age-gap outliers marked.

    Includes the identity line and a fitted trend, because ECG-age models are
    routinely regressive - they over-predict the young and under-predict the
    old - and a plot without the identity line hides that entirely.
    """
    figure, axis = plt.subplots(figsize=(5.2, 5))
    axis.scatter(true_age, predicted_age, s=12, alpha=0.4, color="#4C72B0",
                 linewidths=0, label="test recordings")

    if highlight is not None and len(highlight):
        axis.scatter(
            true_age[highlight], predicted_age[highlight],
            s=42, facecolors="none", edgecolors="#C44E52", linewidths=1.4,
            label="age-gap outliers", zorder=3,
        )

    limits = [
        min(true_age.min(), predicted_age.min()) - 3,
        max(true_age.max(), predicted_age.max()) + 3,
    ]
    axis.plot(limits, limits, color="0.3", linestyle="--", linewidth=1,
              label="perfect prediction")
    if len(true_age) > 2:
        slope, intercept = np.polyfit(true_age, predicted_age, 1)
        axis.plot(
            limits, [slope * x + intercept for x in limits],
            color="#C44E52", linewidth=1.2,
            label=f"fit (slope {slope:.2f})",
        )

    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("chronological age (years)")
    axis.set_ylabel("predicted age (years)")
    axis.set_aspect("equal")
    axis.legend(fontsize=8, frameon=False, loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)
    if title:
        axis.set_title(title, fontsize=10)
    figure.tight_layout()

    if path is not None:
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
    return figure
