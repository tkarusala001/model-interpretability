"""Tests for the figure-producing code.

Plots cannot be checked for being *good*, but they can be checked for the
failures that would quietly produce a misleading figure: silently dropping a
lead, mislabelling an axis, or crashing on a recording whose P or T wave was
never found. The last case matters because attribution outliers - exactly the
recordings this project plots - are disproportionately the ones where
delineation struggles.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from ecg_discovery.analysis.visualization import (
    SEGMENT_COLOURS,
    plot_age_gap_scatter,
    plot_attribution_overlay,
    plot_segment_summary,
)
from ecg_discovery.config import SignalProcessingConfig, SyntheticConfig
from ecg_discovery.data.synthetic_ecg import LEAD_NAMES, generate_recording
from ecg_discovery.interpretability.fiducial_attribution import (
    aggregate_by_fiducial_segment,
)
from ecg_discovery.signal_processing.qrs_detection import detect_r_peaks
from ecg_discovery.signal_processing.wave_delineation import (
    BeatDelineation,
    WaveBoundaries,
    delineate_beats,
)

SP = SignalProcessingConfig()


@pytest.fixture(scope="module")
def recording_and_beats():
    recording = generate_recording(SyntheticConfig(sampling_rate_hz=100), 0, 0)
    detection = detect_r_peaks(recording.signal, 100.0, SP, recording.lead_names)
    beats = delineate_beats(
        recording.signal, 100.0, detection.r_peaks, SP, recording.lead_names
    )
    rng = np.random.default_rng(0)
    attributions = rng.standard_normal(recording.signal.shape) * 0.01
    for beat in beats:
        attributions[:, beat.qrs.onset : beat.qrs.offset] *= 20
    return recording, beats, attributions


# --------------------------------------------------------------------------- #
# Attribution overlay
# --------------------------------------------------------------------------- #
def test_overlay_draws_one_panel_per_requested_lead(recording_and_beats, tmp_path):
    recording, beats, attributions = recording_and_beats
    path = tmp_path / "overlay.png"
    figure = plot_attribution_overlay(
        recording.signal, attributions, beats, recording.lead_names, 100.0,
        leads=("II", "V2", "V5"), path=path,
    )
    assert len(figure.axes) == 3
    assert path.is_file() and path.stat().st_size > 5000


def test_overlay_labels_each_panel_with_its_lead(recording_and_beats):
    recording, beats, attributions = recording_and_beats
    figure = plot_attribution_overlay(
        recording.signal, attributions, beats, recording.lead_names, 100.0,
        leads=("I", "aVR"),
    )
    labels = [axis.get_ylabel() for axis in figure.axes]
    assert any("I" in label for label in labels)
    assert any("aVR" in label for label in labels)


def test_overlay_rejects_an_unknown_lead(recording_and_beats):
    recording, beats, attributions = recording_and_beats
    with pytest.raises(KeyError, match="unknown lead"):
        plot_attribution_overlay(
            recording.signal, attributions, beats, recording.lead_names, 100.0,
            leads=("V9",),
        )


def test_overlay_handles_beats_with_no_p_or_t_wave(recording_and_beats):
    """Outliers are exactly where delineation struggles, so this must not crash."""
    recording, _, attributions = recording_and_beats
    bare = [
        BeatDelineation(
            r_peak=200, qrs=WaveBoundaries(190, 200, 215),
            p_wave=None, t_wave=None, baseline_mv=0.0,
        )
    ]
    figure = plot_attribution_overlay(
        recording.signal, attributions, bare, recording.lead_names, 100.0
    )
    assert len(figure.axes) == 3


def test_overlay_handles_no_beats_at_all(recording_and_beats):
    recording, _, attributions = recording_and_beats
    figure = plot_attribution_overlay(
        recording.signal, attributions, [], recording.lead_names, 100.0
    )
    assert len(figure.axes) == 3


def test_overlay_handles_all_zero_attribution(recording_and_beats):
    """A zero-attribution recording must not divide by zero when scaling opacity."""
    recording, beats, _ = recording_and_beats
    figure = plot_attribution_overlay(
        recording.signal, np.zeros_like(recording.signal), beats,
        recording.lead_names, 100.0,
    )
    assert len(figure.axes) == 3


def test_overlay_respects_an_explicit_time_range(recording_and_beats):
    recording, beats, attributions = recording_and_beats
    figure = plot_attribution_overlay(
        recording.signal, attributions, beats, recording.lead_names, 100.0,
        leads=("II",), time_range_s=(1.0, 2.0),
    )
    lower, upper = figure.axes[0].get_xlim()
    assert 950 <= lower <= 1050 and 1950 <= upper <= 2050


# --------------------------------------------------------------------------- #
# Segment summary
# --------------------------------------------------------------------------- #
def test_segment_summary_shows_share_and_density_together(recording_and_beats, tmp_path):
    """Both panels must be present: share alone is confounded by segment width."""
    recording, beats, attributions = recording_and_beats
    result = aggregate_by_fiducial_segment(attributions, beats, LEAD_NAMES)
    path = tmp_path / "segments.png"
    figure = plot_segment_summary(result, path=path)

    assert len(figure.axes) == 2
    titles = [axis.get_title() for axis in figure.axes]
    assert any("Share" in title for title in titles)
    assert any("density" in title.lower() for title in titles)
    assert path.is_file()


def test_segment_summary_annotates_widths(recording_and_beats):
    """The width confound is shown, not merely corrected for silently."""
    recording, beats, attributions = recording_and_beats
    result = aggregate_by_fiducial_segment(attributions, beats, LEAD_NAMES)
    figure = plot_segment_summary(result)
    annotations = [text.get_text() for text in figure.axes[0].texts]
    assert any("samples" in text for text in annotations)


# --------------------------------------------------------------------------- #
# Age-gap scatter
# --------------------------------------------------------------------------- #
def test_age_gap_scatter_includes_the_identity_line(tmp_path):
    """Without it, a regressive model looks fine; with it, the flaw is visible."""
    rng = np.random.default_rng(0)
    true_age = rng.uniform(20, 89, 200)
    predicted = 0.6 * true_age + 20 + rng.normal(0, 5, 200)   # deliberately regressive

    path = tmp_path / "scatter.png"
    figure = plot_age_gap_scatter(
        true_age, predicted, highlight=np.array([1, 2, 3]), path=path
    )
    axis = figure.axes[0]
    labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert any("perfect prediction" in label for label in labels)
    assert any("slope" in label for label in labels)
    assert axis.get_xlim() == axis.get_ylim()          # equal axes, or slope misleads
    assert path.is_file()


def test_age_gap_scatter_works_without_highlights():
    true_age = np.linspace(20, 80, 50)
    figure = plot_age_gap_scatter(true_age, true_age + 2.0)
    assert len(figure.axes) == 1


def test_segment_colours_are_defined_for_every_segment():
    from ecg_discovery.interpretability.fiducial_attribution import SEGMENT_NAMES

    assert set(SEGMENT_COLOURS) == set(SEGMENT_NAMES)


# --------------------------------------------------------------------------- #
# Phase 8 and 9 result figures
# --------------------------------------------------------------------------- #
def _decomposition(seed=0):
    from ecg_discovery.config import ValidationFrameworkConfig
    from ecg_discovery.validation.residual_decomposition import decompose_age_gap
    import pandas as pd

    rng = np.random.default_rng(seed)
    n = 300
    qrs = rng.normal(95, 12, n)
    return decompose_age_gap(
        0.4 * (qrs - 95) + rng.normal(0, 2, n),
        pd.DataFrame({"heart_rate_bpm": rng.normal(70, 10, n), "qrs_duration_ms": qrs}),
        ValidationFrameworkConfig(
            known_features=("heart_rate_bpm", "qrs_duration_ms"),
            cv_folds=4, bootstrap_iterations=200, seed=0,
        ),
        ages=rng.uniform(30, 85, n), sexes=rng.integers(0, 2, n),
    )


def _link_report(link_strength=2.5):
    from ecg_discovery.config import ValidationFrameworkConfig
    from ecg_discovery.discovery_experiments.unexplained_residual_diagnostic_link import (
        evaluate_diagnostic_link,
    )
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 600
    qrs = rng.normal(95, 12, n)
    residual = rng.normal(0, 4, n)
    logit = -0.4 + 0.05 * (qrs - 95) + link_strength * residual / 4.0
    abnormal = rng.random(n) < 1 / (1 + np.exp(-logit))
    names = ("NORM", "MI", "STTC")
    labels = np.zeros((n, 3), dtype=int)
    labels[~abnormal, 0] = 1
    for index in np.flatnonzero(abnormal):
        labels[index, 1 + rng.integers(0, 2)] = 1
    return evaluate_diagnostic_link(
        residual,
        pd.DataFrame({"heart_rate_bpm": rng.normal(70, 10, n), "qrs_duration_ms": qrs}),
        labels, names,
        ValidationFrameworkConfig(
            known_features=("heart_rate_bpm", "qrs_duration_ms"),
            cv_folds=5, cv_repeats=2, seed=0,
        ),
        ages=rng.uniform(30, 85, n), sexes=rng.integers(0, 2, n),
    )


def test_decomposition_plot_shows_every_explainer(tmp_path):
    """All explainers appear, so no reader sees only the favourable one."""
    from ecg_discovery.analysis.visualization import plot_residual_decomposition

    decomposition = _decomposition()
    path = tmp_path / "decomposition.png"
    figure = plot_residual_decomposition(decomposition, path=path)

    labels = [label.get_text() for label in figure.axes[0].get_xticklabels()]
    assert len(labels) == len(decomposition.explainers)
    legend = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert any("unexplained" in entry for entry in legend)
    assert path.is_file()


def test_decomposition_plot_states_the_ceiling_caveat():
    """The figure must not let 'unexplained' read as 'discovered'."""
    from ecg_discovery.analysis.visualization import plot_residual_decomposition

    figure = plot_residual_decomposition(_decomposition())
    caption = " ".join(text.get_text() for text in figure.texts)
    assert "CEILING" in caption


def test_diagnostic_link_plot_marks_the_zero_line(tmp_path):
    """A difference plot without a zero line hides the null result."""
    from ecg_discovery.analysis.visualization import plot_diagnostic_link

    report = _link_report()
    path = tmp_path / "link.png"
    figure = plot_diagnostic_link(report, path=path)

    assert len(figure.axes) == 2
    zero_lines = [
        line for line in figure.axes[1].get_lines()
        if len(set(np.round(line.get_xdata(), 9))) == 1
        and abs(line.get_xdata()[0]) < 1e-9
    ]
    assert zero_lines, "the difference panel needs a zero reference line"
    assert path.is_file()


def test_diagnostic_link_plot_verdict_matches_the_report():
    """The caption must say what the numbers say, both ways."""
    from ecg_discovery.analysis.visualization import plot_diagnostic_link

    positive = plot_diagnostic_link(_link_report(link_strength=2.5))
    assert "CANDIDATE" in " ".join(text.get_text() for text in positive.texts)

    null = plot_diagnostic_link(_link_report(link_strength=0.0))
    assert "No superclass improves" in " ".join(text.get_text() for text in null.texts)


def test_diagnostic_link_plot_rejects_an_empty_report():
    from ecg_discovery.analysis.visualization import plot_diagnostic_link
    from ecg_discovery.discovery_experiments.unexplained_residual_diagnostic_link import (
        DiagnosticLinkReport,
    )

    empty = DiagnosticLinkReport({}, {}, 0, "none", 0.02)
    with pytest.raises(ValueError, match="no evaluated superclasses"):
        plot_diagnostic_link(empty)
