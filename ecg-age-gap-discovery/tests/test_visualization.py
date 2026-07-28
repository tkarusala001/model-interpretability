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
