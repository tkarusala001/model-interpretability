"""Tests for from-scratch R-peak detection.

Every interval feature in this project is measured relative to an R peak, and
those interval features are what the Phase 8 validation framework calls "already
known". A detector that mislocates peaks would corrupt the known-feature set,
which in turn would make the model's age gap look *less* explainable than it is
and bias the analysis toward a false discovery claim. So detection is scored
against constructed ground truth before anything is allowed to depend on it.

The headline requirement, checked by
``test_localisation_is_within_one_sample_of_ground_truth``: detected peaks land
within one sampling interval of the constructed peak, at both 100 Hz and 500 Hz.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ecg_discovery.config import SignalProcessingConfig, SyntheticConfig
from ecg_discovery.data.synthetic_ecg import generate_cohort, generate_recording
from ecg_discovery.signal_processing.qrs_detection import (
    QRSDetection,
    bandpass_filter,
    detect_r_peaks,
    score_detection,
)

SP = SignalProcessingConfig()

CLEAN_500 = SyntheticConfig(
    sampling_rate_hz=500, noise_mv_sd=0.0, baseline_wander_mv=0.0, powerline_mv=0.0
)
NOISY_500 = SyntheticConfig(sampling_rate_hz=500)
NOISY_100 = SyntheticConfig(sampling_rate_hz=100)


def _detect(recording, config: SignalProcessingConfig = SP) -> QRSDetection:
    return detect_r_peaks(
        recording.signal, recording.sampling_rate_hz, config, recording.lead_names
    )


def _score(recording, config: SignalProcessingConfig = SP):
    detection = _detect(recording, config)
    return score_detection(
        detection.r_peaks,
        recording.r_peak_samples,
        recording.sampling_rate_hz,
        config.match_tolerance_ms,
    )


# --------------------------------------------------------------------------- #
# Filtering primitives
# --------------------------------------------------------------------------- #
def test_bandpass_removes_baseline_wander():
    """Slow drift below the passband must be strongly attenuated."""
    fs, duration = 500.0, 10.0
    t = np.arange(int(fs * duration)) / fs
    wander = 2.0 * np.sin(2 * np.pi * 0.15 * t)      # 0.15 Hz drift
    qrs_band = 0.5 * np.sin(2 * np.pi * 10.0 * t)    # 10 Hz, inside the passband

    noisy = wander + qrs_band
    filtered = bandpass_filter(noisy, fs, 5.0, 15.0, 2)
    core = slice(int(fs), -int(fs))     # ignore filter edge transients

    # The drift dominates the input and must be gone from the output, while the
    # in-band component survives essentially untouched.
    assert np.std(noisy[core]) > 3 * np.std(qrs_band[core])
    assert np.std(filtered[core]) == pytest.approx(np.std(qrs_band[core]), rel=0.05)
    assert np.corrcoef(filtered[core], qrs_band[core])[0, 1] > 0.99


def test_bandpass_is_zero_phase():
    """A symmetric pulse must stay put: no group delay, or peaks would shift."""
    fs = 500.0
    x = np.zeros(3000)
    centre = 1500
    width = 25
    x[centre - width : centre + width] = np.hanning(2 * width)

    filtered = bandpass_filter(x, fs, 5.0, 15.0, 2)
    assert abs(int(np.argmax(filtered)) - centre) <= 1


def test_bandpass_rejects_impossible_cutoffs():
    x = np.random.default_rng(0).standard_normal(2000)
    with pytest.raises(ValueError, match="low_hz"):
        bandpass_filter(x, 100.0, 60.0, 80.0, 2)
    with pytest.raises(ValueError, match="too short"):
        bandpass_filter(x[:5], 100.0, 0.5, 40.0, 2)


def test_bandpass_handles_cutoff_at_nyquist():
    """The 0.5-40 Hz clinical band must stay usable at 100 Hz (Nyquist 50)."""
    x = np.random.default_rng(0).standard_normal(2000)
    assert np.isfinite(bandpass_filter(x, 100.0, 0.5, 40.0, 2)).all()
    # Even asking for a cutoff above Nyquist degrades gracefully to a highpass.
    assert np.isfinite(bandpass_filter(x, 100.0, 0.5, 90.0, 2)).all()


# --------------------------------------------------------------------------- #
# The headline requirement
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("config", [CLEAN_500, NOISY_500, NOISY_100], ids=["clean500", "noisy500", "noisy100"])
def test_localisation_is_within_one_sample_of_ground_truth(config):
    """Detected peaks must land within one sampling interval of the true peak.

    One sample is 2 ms at 500 Hz and 10 ms at 100 Hz. This is the tolerance the
    build plan requires before anything downstream may trust the detector.
    """
    sample_ms = 1000.0 / config.sampling_rate_hz
    errors: list[float] = []
    for recording in generate_cohort(config, n_recordings=40):
        score = _score(recording)
        assert score.sensitivity == 1.0
        assert score.ppv == 1.0
        errors.extend(np.abs(score.errors_ms).tolist())

    errors_array = np.array(errors)
    assert errors_array.size > 300
    assert errors_array.max() <= sample_ms + 1e-9
    assert errors_array.mean() < 0.25 * sample_ms


@pytest.mark.parametrize("config", [NOISY_500, NOISY_100], ids=["500", "100"])
def test_detection_over_a_large_cohort(config):
    """No missed beats, and any spurious detection is a window-edge artefact.

    Sensitivity must be exactly 1.0: a missed beat corrupts two RR intervals and
    there is no excuse for one in clean synthetic data.

    False positives are held to a rate bound rather than to zero, because a
    genuine ambiguity exists at the recording boundary. Ground truth admits a
    beat only if its whole QRS complex fits inside the window, which is a hard
    cutoff: a complex whose offset falls a fraction of a millisecond past the
    ten-second mark is excluded, even though ~99.9% of it was recorded and is
    plainly visible. The detector reasonably finds such a complex. That is a
    disagreement about where to draw an arbitrary line, not a detection error.

    So the test checks what actually matters - that every unmatched detection
    sits at a window edge, and none appears mid-recording, where it would
    indicate a real failure such as a T wave being counted as a beat.
    """
    fs = config.sampling_rate_hz
    edge_samples = int(0.12 * fs)      # ~one QRS width from either boundary
    total_reference = total_matched = total_detected = 0
    interior_false_positives = 0

    for recording in generate_cohort(config, n_recordings=100):
        detection = _detect(recording)
        score = _score(recording)
        total_reference += score.n_reference
        total_matched += score.n_matched
        total_detected += score.n_detected

        truth = recording.r_peak_samples
        tolerance = SP.match_tolerance_ms / 1000.0 * fs
        for peak in detection.r_peaks:
            if truth.size and np.min(np.abs(truth - peak)) <= tolerance:
                continue
            near_edge = peak < edge_samples or peak > recording.n_samples - edge_samples
            if not near_edge:
                interior_false_positives += 1

    assert total_reference > 900
    assert total_matched == total_reference, "missed beats"
    assert interior_false_positives == 0, "spurious detection away from a window edge"
    false_positive_rate = (total_detected - total_matched) / total_reference
    assert false_positive_rate < 0.005, f"false positive rate {false_positive_rate:.4f}"


def test_heart_rate_matches_ground_truth():
    """Detected RR intervals must reproduce the constructed heart rate."""
    for recording in generate_cohort(NOISY_500, n_recordings=25):
        detection = _detect(recording)
        assert detection.heart_rate_bpm() == pytest.approx(
            recording.true_intervals["heart_rate_bpm"], rel=0.02
        )


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("heart_rate", [45.0, 60.0, 90.0, 120.0])
def test_detection_across_the_physiological_heart_rate_range(heart_rate):
    """Bradycardia through tachycardia, where refractory and T-wave rules bite."""
    config = dataclasses.replace(
        NOISY_500, heart_rate_bpm_mean=heart_rate, heart_rate_bpm_sd=0.0,
        hr_change_bpm_per_decade=0.0,
    )
    for recording in generate_cohort(config, n_recordings=12):
        score = _score(recording)
        assert score.sensitivity == 1.0, f"missed beats at {heart_rate} bpm"
        assert score.ppv == 1.0, f"false positives at {heart_rate} bpm"


@pytest.mark.parametrize("noise_sd", [0.05, 0.10, 0.20])
def test_detection_survives_elevated_noise(noise_sd):
    """R waves are ~1.2 mV, so 0.2 mV noise is a badly degraded recording."""
    config = dataclasses.replace(NOISY_500, noise_mv_sd=noise_sd)
    scores = [_score(r) for r in generate_cohort(config, n_recordings=20)]
    assert np.mean([s.sensitivity for s in scores]) > 0.98
    assert np.mean([s.ppv for s in scores]) > 0.98


def test_detection_survives_severe_baseline_wander():
    """Wander far larger than the R wave itself must not defeat detection.

    This is what the 5 Hz high-pass edge of the detection band is for.
    """
    config = dataclasses.replace(NOISY_500, baseline_wander_mv=2.0, baseline_wander_hz=0.4)
    scores = [_score(r) for r in generate_cohort(config, n_recordings=20)]
    assert np.mean([s.sensitivity for s in scores]) > 0.99
    assert np.mean([s.ppv for s in scores]) > 0.99


def test_hyperacute_t_waves_are_not_counted_as_beats():
    """A T wave at half the R-wave height must not double the apparent heart rate.

    This is the specific failure the T-wave slope rule exists to prevent, so it
    is provoked directly rather than trusting that normal recordings happen not
    to trigger it. 0.6 mV against a 1.2 mV R wave is already a hyperacute T
    wave; a detector that counted these would report double the true heart rate.
    """
    config = dataclasses.replace(
        NOISY_500, t_amplitude_mv=0.6, r_amplitude_mv=1.2,
        heart_rate_bpm_mean=95.0, heart_rate_bpm_sd=0.0,
    )
    scores = [_score(r) for r in generate_cohort(config, n_recordings=40)]
    # Measured over 150 recordings this is 0.9992, not 1.0 - a couple of T waves
    # in a few hundred still slip through. Asserted as a bound rather than
    # equality so the test states what the detector actually achieves.
    assert float(np.mean([s.ppv for s in scores])) > 0.99
    assert float(np.mean([s.sensitivity for s in scores])) == 1.0


def test_extremely_tall_t_waves_are_a_documented_limitation():
    """Pin the known failure case so it cannot silently get worse - or vanish.

    At 83% of R amplitude the synthetic T wave is nearly as steep as the QRS
    itself, and amplitude-plus-slope thresholding cannot fully separate two
    waves that differ in neither: positive predictive value plateaus near 0.97
    whatever ``t_wave_slope_fraction`` is set to. The slope rule still helps a
    great deal here, which is what this test asserts, but it does not solve the
    case, and the paper should not claim it does. See docs/limitations.md.
    """
    config = dataclasses.replace(
        NOISY_500, t_amplitude_mv=1.0, r_amplitude_mv=1.2,
        heart_rate_bpm_mean=95.0, heart_rate_bpm_sd=0.0,
    )
    disabled = dataclasses.replace(SP, t_wave_slope_fraction=0.0)
    with_rule = [_score(r) for r in generate_cohort(config, n_recordings=15)]
    without_rule = [_score(r, disabled) for r in generate_cohort(config, n_recordings=15)]

    ppv_with = float(np.mean([s.ppv for s in with_rule]))
    ppv_without = float(np.mean([s.ppv for s in without_rule]))

    assert np.mean([s.sensitivity for s in with_rule]) == 1.0   # no beats lost
    assert ppv_with > ppv_without + 0.2, "the slope rule should still help a lot"
    assert 0.9 < ppv_with < 1.0, (
        f"expected the documented plateau near 0.97, got {ppv_with:.3f}; if this "
        "has improved, update docs/limitations.md rather than loosening the test"
    )


def test_detection_on_a_lead_with_negative_r_wave():
    """In aVR the complex points down; auto polarity must handle it.

    Ground-truth R-peak *timing* is identical across leads by construction, so
    a correct detector finds the same instants whichever lead it reads.
    """
    negative_lead = dataclasses.replace(SP, detection_lead="aVR", polarity="auto")
    for recording in generate_cohort(CLEAN_500, n_recordings=10):
        detection = _detect(recording, negative_lead)
        assert detection.polarity == -1
        score = score_detection(
            detection.r_peaks, recording.r_peak_samples,
            recording.sampling_rate_hz, 50.0,
        )
        assert score.sensitivity == 1.0
        assert score.ppv == 1.0
        # aVR inverts the complex, so its extremum is the inverted R wave and
        # sits at the same instant; allow one extra sample of slack.
        assert score.max_absolute_error_ms <= 2 * (1000.0 / 500.0)


def test_forcing_the_wrong_polarity_degrades_localisation():
    """Guard: the polarity test above must not be passing for a trivial reason."""
    wrong = dataclasses.replace(SP, detection_lead="aVR", polarity="positive")
    recording = generate_recording(CLEAN_500, 0, 0)
    detection = _detect(recording, wrong)
    score = score_detection(
        detection.r_peaks, recording.r_peak_samples, 500.0, 50.0
    )
    assert score.mean_absolute_error_ms > 5.0


# --------------------------------------------------------------------------- #
# Degenerate input
# --------------------------------------------------------------------------- #
def test_flat_signal_returns_no_beats():
    detection = detect_r_peaks(np.zeros(5000), 500.0, SP)
    assert detection.n_beats == 0
    assert np.isnan(detection.heart_rate_bpm())
    assert detection.rr_intervals_ms().size == 0


def test_signal_with_nans_returns_no_beats():
    x = np.zeros(5000)
    x[10] = np.nan
    assert detect_r_peaks(x, 500.0, SP).n_beats == 0


def test_very_short_signal_returns_no_beats():
    assert detect_r_peaks(np.random.default_rng(0).standard_normal(20), 500.0, SP).n_beats == 0


def test_pure_noise_does_not_produce_a_plausible_rhythm():
    """Noise may trigger detections, but must not look like a regular heartbeat."""
    rng = np.random.default_rng(0)
    detection = detect_r_peaks(rng.standard_normal(5000) * 0.05, 500.0, SP)
    if detection.n_beats >= 3:
        rr = detection.rr_intervals_ms()
        assert np.std(rr) / np.mean(rr) > 0.15, "noise produced a suspiciously regular rhythm"


def test_single_lead_input_is_accepted():
    recording = generate_recording(CLEAN_500, 0, 0)
    from_1d = detect_r_peaks(recording.lead("II"), 500.0, SP)
    from_2d = _detect(recording)
    np.testing.assert_array_equal(from_1d.r_peaks, from_2d.r_peaks)


def test_unknown_detection_lead_raises():
    recording = generate_recording(CLEAN_500, 0, 0)
    with pytest.raises(KeyError, match="detection_lead"):
        _detect(recording, dataclasses.replace(SP, detection_lead="V9"))


def test_detection_is_deterministic():
    recording = generate_recording(NOISY_500, 3, 3)
    a, b = _detect(recording), _detect(recording)
    np.testing.assert_array_equal(a.r_peaks, b.r_peaks)


def test_edge_guard_discards_truncated_complexes():
    """A complex cut in half by the window edge must not be reported.

    Without the guard the detector fires on partial complexes at the very end of
    a recording, and each one injects a spuriously short RR interval that biases
    the measured heart rate.
    """
    no_guard = dataclasses.replace(SP, edge_guard_ms=0.0)
    guarded_fp = unguarded_fp = 0
    for recording in generate_cohort(NOISY_500, n_recordings=60):
        for config, counter in ((SP, "guarded"), (no_guard, "unguarded")):
            score = _score(recording, config)
            extra = score.n_detected - score.n_matched
            if counter == "guarded":
                guarded_fp += extra
            else:
                unguarded_fp += extra
    assert unguarded_fp > 0, "expected truncated complexes to be detectable at all"
    assert guarded_fp == 0


# --------------------------------------------------------------------------- #
# The scoring function itself
# --------------------------------------------------------------------------- #
def test_score_detection_on_a_perfect_match():
    peaks = np.array([100, 600, 1100])
    score = score_detection(peaks, peaks, 500.0, 50.0)
    assert score.sensitivity == 1.0 and score.ppv == 1.0 and score.f1 == 1.0
    assert score.mean_absolute_error_ms == 0.0


def test_score_detection_counts_misses_and_false_positives():
    reference = np.array([100, 600, 1100, 1600])
    detected = np.array([100, 605, 1600, 3000])   # 1100 missed, 3000 spurious
    score = score_detection(detected, reference, 500.0, 50.0)
    assert score.n_matched == 3
    assert score.sensitivity == pytest.approx(0.75)
    assert score.ppv == pytest.approx(0.75)
    assert score.max_absolute_error_ms == pytest.approx(10.0)


def test_score_detection_matching_is_one_to_one():
    """A detector firing twice per beat must be penalised, not rewarded."""
    reference = np.array([100, 600])
    detected = np.array([100, 105, 600, 605])
    score = score_detection(detected, reference, 500.0, 50.0)
    assert score.n_matched == 2
    assert score.sensitivity == 1.0
    assert score.ppv == pytest.approx(0.5)


def test_score_detection_handles_empty_inputs():
    reference = np.array([100, 600])
    empty = np.empty(0, dtype=np.int64)
    score = score_detection(empty, reference, 500.0, 50.0)
    assert score.n_matched == 0 and score.sensitivity == 0.0 and score.f1 == 0.0
    assert np.isnan(score.mean_absolute_error_ms)
    assert np.isnan(score_detection(reference, empty, 500.0, 50.0).sensitivity)
