"""Tests for classical interval measurement - the "already known" feature set.

These features are the yardstick Phase 8 measures discovery against, and the
error here is asymmetric in its consequences: features that are noisier than
they should be explain *less* of the model's age gap, which inflates the
unexplained residual and makes a discovery claim look stronger than it is.
Sloppy measurement manufactures false discoveries. So the tests below check
not just that measurements are close to the truth, but that they *correlate*
with it - because correlation, not absolute accuracy, is what determines how
much variance a feature can explain.

``test_measuring_at_100hz_would_bias_towards_false_discovery`` is the one that
justifies a headline design decision of the project.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from ecg_discovery.config import SignalProcessingConfig, SyntheticConfig
from ecg_discovery.data.synthetic_ecg import generate_cohort, generate_recording
from ecg_discovery.signal_processing.interval_features import (
    INTERVAL_FEATURE_NAMES,
    IntervalFeatures,
    compute_interval_features,
    interval_features_table,
)
from ecg_discovery.signal_processing.qrs_detection import detect_r_peaks
from ecg_discovery.signal_processing.wave_delineation import delineate_beats

SP = SignalProcessingConfig()
AT_500 = SyntheticConfig(sampling_rate_hz=500)
AT_100 = SyntheticConfig(sampling_rate_hz=100)

MEASURED_TO_TRUE = {
    "heart_rate_bpm": "true_heart_rate_bpm",
    "qrs_duration_ms": "true_qrs_duration_ms",
    "pr_interval_ms": "true_pr_interval_ms",
    "qt_interval_ms": "true_qt_interval_ms",
    "qtc_bazett_ms": "true_qtc_bazett_ms",
}


def _measured_and_true(config: SyntheticConfig, n_recordings: int = 120):
    cohort = generate_cohort(config, n_recordings=n_recordings)
    table = interval_features_table(
        cohort.signals, config.sampling_rate_hz, SP, cohort[0].lead_names
    )
    return table, cohort.metadata()


def _correlation(measured: np.ndarray, truth: np.ndarray) -> float:
    ok = np.isfinite(measured) & np.isfinite(truth)
    return float(np.corrcoef(measured[ok], truth[ok])[0, 1])


# --------------------------------------------------------------------------- #
# Accuracy against constructed ground truth
# --------------------------------------------------------------------------- #
def test_every_interval_tracks_ground_truth_at_500hz():
    """Each measured interval must correlate strongly with its true value.

    Correlation is the property that matters for the validation framework: a
    feature that tracks the truth explains the variance the truth would explain,
    whatever constant offset the measurement carries.
    """
    table, truth = _measured_and_true(AT_500)
    for measured_name, true_name in MEASURED_TO_TRUE.items():
        r = _correlation(table[measured_name].to_numpy(), truth[true_name].to_numpy())
        assert r > 0.97, f"{measured_name} correlates only r={r:.3f} with ground truth"


def test_interval_biases_are_within_documented_bounds():
    """Pin the systematic offsets so the documented tolerances stay honest.

    Constant biases are acceptable here - they shift every recording equally and
    so do not change how much variance a feature explains - but they must be
    stated rather than discovered later. QT is the largest, because it inherits
    both a slightly late QRS onset and a slightly early tangent-method T offset.
    """
    table, truth = _measured_and_true(AT_500)
    bounds_ms = {
        "heart_rate_bpm": 1.0,      # bpm, not ms
        "qrs_duration_ms": 10.0,
        "pr_interval_ms": 10.0,
        "qt_interval_ms": 35.0,
        "qtc_bazett_ms": 35.0,
    }
    for measured_name, true_name in MEASURED_TO_TRUE.items():
        error = table[measured_name].to_numpy() - truth[true_name].to_numpy()
        error = error[np.isfinite(error)]
        assert abs(error.mean()) < bounds_ms[measured_name], (
            f"{measured_name} bias {error.mean():.1f} exceeds the documented bound"
        )


def test_heart_rate_is_measured_near_exactly():
    """Rate depends only on R peaks, which are located to within one sample."""
    table, truth = _measured_and_true(AT_500)
    error = table["heart_rate_bpm"].to_numpy() - truth["true_heart_rate_bpm"].to_numpy()
    assert np.abs(error).mean() < 1.0
    assert _correlation(
        table["heart_rate_bpm"].to_numpy(), truth["true_heart_rate_bpm"].to_numpy()
    ) > 0.999


def test_measuring_at_100hz_would_bias_towards_false_discovery():
    """Justifies measuring intervals at 500 Hz while the model trains at 100 Hz.

    One sample at 100 Hz is 10 ms, which is coarse relative to the interval
    differences the validation framework must resolve. The effect is not subtle:
    QRS duration, the feature carrying the synthetic known-age channel, tracks
    the truth far less well at 100 Hz.

    That matters because explained variance goes roughly as r-squared. A feature
    correlating at 0.99 instead of 0.69 explains about twice the variance, so
    measuring at 100 Hz would leave a large part of a *genuinely known* effect
    sitting in the "unexplained" residual - the direction that flatters a
    discovery claim. This test exists so that reasoning is checked rather than
    asserted.
    """
    table_500, truth_500 = _measured_and_true(AT_500)
    table_100, truth_100 = _measured_and_true(AT_100)

    r_500 = _correlation(
        table_500["qrs_duration_ms"].to_numpy(), truth_500["true_qrs_duration_ms"].to_numpy()
    )
    r_100 = _correlation(
        table_100["qrs_duration_ms"].to_numpy(), truth_100["true_qrs_duration_ms"].to_numpy()
    )
    assert r_500 > 0.98
    assert r_100 < 0.90
    assert r_500 ** 2 > 1.5 * r_100 ** 2, (
        f"expected 500 Hz to explain substantially more QRS variance than 100 Hz; "
        f"got r={r_500:.3f} vs r={r_100:.3f}"
    )


# --------------------------------------------------------------------------- #
# Aggregation robustness
# --------------------------------------------------------------------------- #
def test_median_aggregation_resists_a_single_corrupted_beat():
    """One badly delineated beat must not move the recording's features.

    This is why the median is the default. The failure being guarded against -
    a single beat delineated wrongly - is common, and its effect on a mean is
    unbounded.
    """
    recording = generate_recording(AT_500, 0, 0)
    detection = detect_r_peaks(recording.signal, 500.0, SP, recording.lead_names)
    beats = delineate_beats(
        recording.signal, 500.0, detection.r_peaks, SP, recording.lead_names
    )
    assert len(beats) >= 6

    clean = compute_interval_features(recording.signal, 500.0, SP, recording.lead_names, beats)

    # Corrupt one beat's QRS offset into an absurd 400 ms complex.
    from ecg_discovery.signal_processing.wave_delineation import WaveBoundaries

    damaged = list(beats)
    bad = damaged[2]
    damaged[2] = dataclasses.replace(
        bad, qrs=WaveBoundaries(bad.qrs.onset, bad.qrs.peak, bad.qrs.onset + 200)
    )

    median_config = dataclasses.replace(SP, aggregation="median")
    mean_config = dataclasses.replace(SP, aggregation="mean")
    with_median = compute_interval_features(
        recording.signal, 500.0, median_config, recording.lead_names, damaged
    )
    with_mean = compute_interval_features(
        recording.signal, 500.0, mean_config, recording.lead_names, damaged
    )

    median_shift = abs(with_median.qrs_duration_ms - clean.qrs_duration_ms)
    mean_shift = abs(with_mean.qrs_duration_ms - clean.qrs_duration_ms)
    assert median_shift < 3.0
    assert mean_shift > 3 * median_shift


# --------------------------------------------------------------------------- #
# Missing data is reported, never fabricated
# --------------------------------------------------------------------------- #
def test_too_few_beats_gives_all_nan_features():
    features = compute_interval_features(np.zeros((12, 5000)), 500.0, SP)
    assert features.n_beats == 0
    assert not features.is_measurable
    assert all(math.isnan(getattr(features, name)) for name in INTERVAL_FEATURE_NAMES)


def test_absent_p_waves_give_nan_pr_not_a_fabricated_value():
    """A missing P wave must not silently become a plausible-looking PR."""
    flat_p = dataclasses.replace(
        AT_500, p_amplitude_mv=0.0, noise_mv_sd=0.0,
        baseline_wander_mv=0.0, powerline_mv=0.0,
    )
    recording = generate_recording(flat_p, 0, 0)
    features = compute_interval_features(recording.signal, 500.0, SP, recording.lead_names)
    assert features.p_detection_rate < 0.05
    assert math.isnan(features.pr_interval_ms)
    # The rest of the recording is still measurable.
    assert np.isfinite(features.heart_rate_bpm)
    assert np.isfinite(features.qrs_duration_ms)


def test_p_detection_rate_is_high_for_normal_recordings():
    recording = generate_recording(AT_500, 0, 0)
    features = compute_interval_features(recording.signal, 500.0, SP, recording.lead_names)
    assert features.p_detection_rate > 0.9
    assert features.t_detection_rate > 0.9
    assert features.is_measurable


def test_min_beats_threshold_is_respected():
    strict = dataclasses.replace(SP, min_beats_for_features=100)
    recording = generate_recording(AT_500, 0, 0)
    features = compute_interval_features(recording.signal, 500.0, strict, recording.lead_names)
    assert not features.is_measurable


# --------------------------------------------------------------------------- #
# Derived quantities
# --------------------------------------------------------------------------- #
def test_rate_corrections_match_their_formulas():
    """QTc must be exactly QT divided by RR to the appropriate power."""
    recording = generate_recording(AT_500, 1, 1)
    features = compute_interval_features(recording.signal, 500.0, SP, recording.lead_names)
    rr_seconds = features.rr_interval_ms / 1000.0
    assert features.qtc_bazett_ms == pytest.approx(
        features.qt_interval_ms / math.sqrt(rr_seconds)
    )
    assert features.qtc_fridericia_ms == pytest.approx(
        features.qt_interval_ms / rr_seconds ** (1 / 3)
    )


def test_heart_rate_and_rr_interval_are_consistent():
    recording = generate_recording(AT_500, 2, 2)
    features = compute_interval_features(recording.signal, 500.0, SP, recording.lead_names)
    assert features.heart_rate_bpm == pytest.approx(60_000.0 / features.rr_interval_ms)


def test_rate_corrections_differ_at_abnormal_heart_rates():
    """Bazett and Fridericia agree at 60 bpm by construction and diverge away from it."""
    fast = dataclasses.replace(
        AT_500, heart_rate_bpm_mean=110.0, heart_rate_bpm_sd=0.0,
        hr_change_bpm_per_decade=0.0,
    )
    recording = generate_recording(fast, 0, 0)
    features = compute_interval_features(recording.signal, 500.0, SP, recording.lead_names)
    assert features.qtc_bazett_ms > features.qtc_fridericia_ms


# --------------------------------------------------------------------------- #
# Feature selection and batch interface
# --------------------------------------------------------------------------- #
def test_feature_vector_selects_in_the_requested_order():
    recording = generate_recording(AT_500, 0, 0)
    features = compute_interval_features(recording.signal, 500.0, SP, recording.lead_names)
    names = ("qrs_duration_ms", "heart_rate_bpm")
    vector = features.feature_vector(names)
    assert vector.shape == (2,)
    assert vector[0] == features.qrs_duration_ms
    assert vector[1] == features.heart_rate_bpm


def test_feature_vector_rejects_unknown_names():
    with pytest.raises(KeyError, match="unknown interval feature"):
        IntervalFeatures.empty().feature_vector(["st_elevation_mv"])


def test_interval_features_table_shape_and_ids():
    cohort = generate_cohort(AT_500, n_recordings=5)
    ids = [r.record_id for r in cohort]
    table = interval_features_table(cohort.signals, 500.0, SP, cohort[0].lead_names, ids)
    assert len(table) == 5
    assert list(table["record_id"]) == ids
    for name in INTERVAL_FEATURE_NAMES:
        assert name in table.columns


def test_interval_features_table_keeps_failed_rows_aligned():
    """A failed recording becomes a NaN row, not a dropped one.

    Dropping it would silently misalign the feature table against the model
    predictions it is later joined to.
    """
    cohort = generate_cohort(AT_500, n_recordings=4)
    signals = cohort.signals.copy()
    signals[2] = 0.0                         # a flat, unmeasurable recording
    table = interval_features_table(signals, 500.0, SP, cohort[0].lead_names)
    assert len(table) == 4
    assert math.isnan(table.loc[2, "heart_rate_bpm"])
    assert np.isfinite(table.loc[3, "heart_rate_bpm"])


def test_interval_features_table_rejects_wrong_shapes():
    cohort = generate_cohort(AT_500, n_recordings=3)
    with pytest.raises(ValueError, match="n_recordings"):
        interval_features_table(cohort[0].signal, 500.0, SP)
    with pytest.raises(ValueError, match="record_ids"):
        interval_features_table(cohort.signals, 500.0, SP, record_ids=["only_one"])


# --------------------------------------------------------------------------- #
# Amplitude, axis and morphology features
# --------------------------------------------------------------------------- #
def test_amplitude_features_are_measured_on_synthetic_beats():
    recording = generate_recording(AT_500, 0, 0)
    features = compute_interval_features(recording.signal, 500.0, SP, recording.lead_names)
    # The generator builds lead II with a 1.2 mV R wave and a 0.3 mV T wave.
    assert 0.5 < features.r_amplitude_mv < 2.5
    assert 0.05 < features.t_amplitude_mv < 0.8
    assert 0.02 < features.p_amplitude_mv < 0.5
    assert np.isfinite(features.sokolow_lyon_mv)


def test_amplitude_features_scale_with_the_signal():
    """Doubling the waveform must double the measured amplitudes."""
    recording = generate_recording(AT_500, 0, 0)
    base = compute_interval_features(recording.signal, 500.0, SP, recording.lead_names)
    doubled = compute_interval_features(
        recording.signal * 2.0, 500.0, SP, recording.lead_names
    )
    assert doubled.r_amplitude_mv == pytest.approx(2 * base.r_amplitude_mv, rel=0.15)
    assert doubled.sokolow_lyon_mv == pytest.approx(2 * base.sokolow_lyon_mv, rel=0.15)
    # Axis is a direction, so scaling the whole signal must not move it.
    assert doubled.qrs_axis_deg == pytest.approx(base.qrs_axis_deg, abs=2.0)


def test_qrs_axis_responds_to_lead_geometry():
    """Axis must be computed from leads I and aVF, not invented.

    Flipping lead aVF reflects the frontal-plane vector about the horizontal
    axis, so the computed axis must change sign.
    """
    recording = generate_recording(AT_500, 0, 0)
    flipped = recording.signal.copy()
    flipped[list(recording.lead_names).index("aVF")] *= -1

    original = compute_interval_features(recording.signal, 500.0, SP, recording.lead_names)
    reflected = compute_interval_features(flipped, 500.0, SP, recording.lead_names)
    assert np.sign(reflected.qrs_axis_deg) != np.sign(original.qrs_axis_deg)
    assert abs(reflected.qrs_axis_deg) == pytest.approx(
        abs(original.qrs_axis_deg), abs=5.0
    )


def test_st_deviation_tracks_an_injected_shift():
    """A deliberate ST-segment offset must show up in the measurement."""
    from ecg_discovery.config import SignalProcessingConfig
    from ecg_discovery.signal_processing.qrs_detection import detect_r_peaks
    from ecg_discovery.signal_processing.wave_delineation import delineate_beats

    recording = generate_recording(
        dataclasses.replace(AT_500, noise_mv_sd=0.0, baseline_wander_mv=0.0), 0, 0
    )
    detection = detect_r_peaks(recording.signal, 500.0, SP, recording.lead_names)
    beats = delineate_beats(
        recording.signal, 500.0, detection.r_peaks, SP, recording.lead_names
    )
    baseline_features = compute_interval_features(
        recording.signal, 500.0, SP, recording.lead_names, beats
    )

    elevated = recording.signal.copy()
    for beat in beats:
        start = beat.qrs.offset
        stop = min(beat.qrs.offset + int(0.10 * 500), elevated.shape[1])
        elevated[:, start:stop] += 0.25          # 0.25 mV ST elevation
    raised = compute_interval_features(
        elevated, 500.0, SP, recording.lead_names, beats
    )
    assert raised.st_deviation_mv > baseline_features.st_deviation_mv + 0.1


def test_r_progression_is_a_precordial_lead_number():
    recording = generate_recording(AT_500, 0, 0)
    features = compute_interval_features(recording.signal, 500.0, SP, recording.lead_names)
    if np.isfinite(features.r_progression_lead):
        assert 1 <= features.r_progression_lead <= 6


def test_amplitude_features_are_nan_without_beats():
    features = compute_interval_features(np.zeros((12, 5000)), 500.0, SP)
    for name in ("r_amplitude_mv", "qrs_axis_deg", "sokolow_lyon_mv"):
        assert math.isnan(getattr(features, name))


def test_single_lead_input_gives_nan_for_multi_lead_features():
    """Axis and Sokolow-Lyon need specific leads; absent, they must not be faked."""
    recording = generate_recording(AT_500, 0, 0)
    features = compute_interval_features(recording.lead("II"), 500.0, SP)
    assert math.isnan(features.qrs_axis_deg)
    assert math.isnan(features.sokolow_lyon_mv)
    assert np.isfinite(features.heart_rate_bpm)      # timing still works
