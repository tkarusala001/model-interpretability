"""Tests for P-wave and T-wave delineation.

Delineation is explicitly an approximation (see the module docstring), so these
tests do two things rather than one. They check the parts that must be exactly
right - ordering, the absence of gross failures, correct reporting of missing
waves - and they *pin the measured accuracy of the parts that are approximate*,
so the documented tolerances stay honest and cannot silently drift.

The bias figures asserted below are real, measured properties of the method,
not aspirations. Where a boundary is systematically late or early, the test says
so and says why.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ecg_discovery.config import SignalProcessingConfig, SyntheticConfig
from ecg_discovery.data.synthetic_ecg import generate_cohort, generate_recording
from ecg_discovery.signal_processing.qrs_detection import detect_r_peaks
from ecg_discovery.signal_processing.wave_delineation import (
    WaveBoundaries,
    _tangent_boundary,
    _wave_extent,
    delineate_beats,
    delineation_signals,
)

SP = SignalProcessingConfig()
PRECISE = SyntheticConfig(sampling_rate_hz=500)
CLEAN = dataclasses.replace(
    PRECISE, noise_mv_sd=0.0, baseline_wander_mv=0.0, powerline_mv=0.0
)


def _delineate(recording, config: SignalProcessingConfig = SP):
    detection = detect_r_peaks(
        recording.signal, recording.sampling_rate_hz, config, recording.lead_names
    )
    return delineate_beats(
        recording.signal, recording.sampling_rate_hz, detection.r_peaks,
        config, recording.lead_names,
    )


def _errors_ms(config_synth, config_sp=SP, n_recordings=40) -> dict[str, np.ndarray]:
    """Signed error, in milliseconds, of every fiducial against ground truth."""
    collected: dict[str, list[float]] = {
        key: [] for key in (
            "qrs_onset", "qrs_offset", "qrs_duration",
            "p_onset", "p_peak", "p_offset",
            "t_onset", "t_peak", "t_offset",
        )
    }
    to_ms = 1000.0 / config_synth.sampling_rate_hz
    for recording in generate_cohort(config_synth, n_recordings=n_recordings):
        truth = {beat.r_peak: beat for beat in recording.beats}
        for beat in _delineate(recording, config_sp):
            reference = truth.get(beat.r_peak)
            if reference is None:
                continue
            collected["qrs_onset"].append((beat.qrs.onset - reference.qrs_onset) * to_ms)
            collected["qrs_offset"].append((beat.qrs.offset - reference.qrs_offset) * to_ms)
            collected["qrs_duration"].append(
                (beat.qrs.duration_samples
                 - (reference.qrs_offset - reference.qrs_onset)) * to_ms
            )
            if beat.p_wave is not None:
                collected["p_onset"].append((beat.p_wave.onset - reference.p_onset) * to_ms)
                collected["p_peak"].append((beat.p_wave.peak - reference.p_peak) * to_ms)
                collected["p_offset"].append((beat.p_wave.offset - reference.p_offset) * to_ms)
            if beat.t_wave is not None:
                collected["t_onset"].append((beat.t_wave.onset - reference.t_onset) * to_ms)
                collected["t_peak"].append((beat.t_wave.peak - reference.t_peak) * to_ms)
                collected["t_offset"].append((beat.t_wave.offset - reference.t_offset) * to_ms)
    return {key: np.array(values) for key, values in collected.items()}


# --------------------------------------------------------------------------- #
# Structural correctness - these must be exactly right
# --------------------------------------------------------------------------- #
def test_fiducials_are_ordered_within_every_beat():
    """P before QRS before T, with no wave overlapping its neighbour."""
    for recording in generate_cohort(PRECISE, n_recordings=20):
        for beat in _delineate(recording):
            assert beat.qrs.onset < beat.r_peak < beat.qrs.offset
            if beat.p_wave is not None:
                p = beat.p_wave
                assert p.onset <= p.peak <= p.offset
                assert p.offset <= beat.qrs.onset
            if beat.t_wave is not None:
                t = beat.t_wave
                assert t.onset <= t.peak <= t.offset
                assert t.onset >= beat.qrs.offset


def test_qrs_boundaries_do_not_collapse_inside_the_complex():
    """The specific failure the sustained-quiet-run rule exists to prevent.

    The waveform crosses baseline between the Q, R and S deflections, so slope
    briefly falls to zero *inside* the complex. A boundary search that accepted
    the first quiet sample would stop there and report a QRS a fraction of its
    true width. Measured durations must stay physiologically plausible.
    """
    for recording in generate_cohort(PRECISE, n_recordings=20):
        for beat in _delineate(recording):
            duration_ms = beat.qrs_duration_samples * 1000.0 / 500.0
            assert 50.0 < duration_ms < 200.0, (
                f"QRS duration {duration_ms:.0f} ms is not physiologically "
                "plausible - the boundary search has probably collapsed inside "
                "the complex"
            )


def test_one_delineation_per_detected_beat():
    for recording in generate_cohort(PRECISE, n_recordings=10):
        detection = detect_r_peaks(recording.signal, 500.0, SP, recording.lead_names)
        beats = _delineate(recording)
        assert len(beats) == detection.n_beats
        assert [b.r_peak for b in beats] == detection.r_peaks.tolist()


def test_no_r_peaks_gives_no_delineation():
    recording = generate_recording(PRECISE, 0, 0)
    assert delineate_beats(recording.signal, 500.0, np.empty(0, dtype=np.int64), SP) == []


def test_delineation_is_deterministic():
    recording = generate_recording(PRECISE, 5, 5)
    first, second = _delineate(recording), _delineate(recording)
    assert first == second


# --------------------------------------------------------------------------- #
# Missing waves must be reported, never invented
# --------------------------------------------------------------------------- #
def test_absent_p_wave_is_reported_as_missing():
    """No P wave means None, not a fabricated PR interval.

    Atrial fibrillation genuinely has no organised P wave. Inventing one would
    put a fictitious PR interval into the known-feature set that Phase 8
    measures discovery against.
    """
    flat_p = dataclasses.replace(CLEAN, p_amplitude_mv=0.0)
    beats = [beat for r in generate_cohort(flat_p, n_recordings=6) for beat in _delineate(r)]
    assert beats
    detected = sum(beat.p_wave is not None for beat in beats)
    assert detected / len(beats) < 0.05


def test_absent_t_wave_is_reported_as_missing():
    flat_t = dataclasses.replace(CLEAN, t_amplitude_mv=0.0)
    beats = [beat for r in generate_cohort(flat_t, n_recordings=6) for beat in _delineate(r)]
    assert beats
    detected = sum(beat.t_wave is not None for beat in beats)
    assert detected / len(beats) < 0.05


def test_normal_waves_are_found_essentially_always():
    """Guard: the "absent wave" tests must not pass because nothing is found."""
    beats = [beat for r in generate_cohort(PRECISE, n_recordings=15) for beat in _delineate(r)]
    assert sum(b.p_wave is not None for b in beats) / len(beats) > 0.97
    assert sum(b.t_wave is not None for b in beats) / len(beats) > 0.97


# --------------------------------------------------------------------------- #
# Measured accuracy - pinning the documented tolerances
# --------------------------------------------------------------------------- #
def test_wave_peaks_are_located_accurately():
    """Peaks are the well-determined part: they are extrema, not thresholds."""
    errors = _errors_ms(PRECISE)
    for wave in ("p_peak", "t_peak"):
        assert abs(errors[wave].mean()) < 5.0, f"{wave} bias"
        assert np.abs(errors[wave]).mean() < 8.0, f"{wave} MAE"


def test_qrs_duration_error_is_small_and_documented():
    """QRS duration is the feature carrying the synthetic known-age channel.

    Boundaries land a little inside the complex because the slope envelope is
    thresholded at 10% of its peak, but the two errors partly cancel in the
    duration, leaving a small negative bias.
    """
    errors = _errors_ms(PRECISE)
    assert -10.0 < errors["qrs_duration"].mean() < 0.0
    assert errors["qrs_duration"].std() < 8.0


def test_t_offset_is_biased_early_by_a_documented_amount():
    """The tangent method ends the T wave slightly early - stated, not hidden.

    A tangent drawn at the steepest point of the downslope meets baseline before
    the wave has fully flattened, so T offset is systematically early. The bias
    is consistent, which is what matters for Phase 8: a constant offset shifts
    every recording's QT equally and does not affect how much variance the
    feature explains.
    """
    errors = _errors_ms(PRECISE)
    assert -30.0 < errors["t_offset"].mean() < -5.0
    assert errors["t_offset"].std() < 25.0


def test_t_onset_is_the_least_accurate_fiducial():
    """Pin the known weak point so it cannot quietly get worse.

    The synthetic T wave is a raised cosine, which begins with *zero amplitude
    and zero slope* - an onset that is mathematically undetectable. Both the
    tangent and threshold methods therefore fire some way into the wave; for a
    raised cosine the tangent crosses baseline about 9% in, which at a 150 ms
    T wave is a ~14 ms floor before any noise.

    This is tolerable because t_onset feeds no known interval feature: QT is
    measured to t_*offset*. It only sets the T-segment boundary for Phase 6
    attribution, where a late onset shifts a little early-T-wave attribution
    into the isoelectric bucket. Real T waves start less gradually than this
    synthetic one, so this is a pessimistic estimate.
    """
    errors = _errors_ms(PRECISE)
    assert 5.0 < errors["t_onset"].mean() < 45.0
    # Still far better than nothing: the T segment is ~150 ms wide.
    assert np.abs(errors["t_onset"]).mean() < 50.0


def test_tangent_method_beats_the_threshold_method_for_t_offset():
    """Justify the default: the tangent method is chosen on evidence."""
    threshold_config = dataclasses.replace(SP, t_offset_method="threshold")
    tangent_error = _errors_ms(PRECISE, SP, n_recordings=25)["t_offset"]
    threshold_error = _errors_ms(PRECISE, threshold_config, n_recordings=25)["t_offset"]
    assert np.abs(tangent_error).mean() < np.abs(threshold_error).mean()


# --------------------------------------------------------------------------- #
# Configuration paths
# --------------------------------------------------------------------------- #
def test_vector_magnitude_and_single_lead_qrs_sources_both_work():
    """Both QRS boundary sources must produce plausible complexes.

    Note what this test cannot show: the synthetic cohort's leads are exact
    linear projections of three shared components, so wave timing is identical
    in every lead and inter-lead dispersion is zero. The real advantage of a
    multi-lead source - capturing depolarisation that starts in one lead before
    another - is therefore invisible here by construction. This must be
    re-checked on PTB-XL; see docs/limitations.md.
    """
    single = dataclasses.replace(SP, qrs_boundary_source="lead")
    for recording in generate_cohort(PRECISE, n_recordings=8):
        for config in (SP, single):
            beats = _delineate(recording, config)
            assert beats
            durations = [b.qrs_duration_samples * 2.0 for b in beats]
            assert all(50.0 < d < 200.0 for d in durations)


def test_delineation_accepts_single_lead_input():
    recording = generate_recording(PRECISE, 0, 0)
    detection = detect_r_peaks(recording.signal, 500.0, SP, recording.lead_names)
    beats = delineate_beats(recording.lead("II"), 500.0, detection.r_peaks, SP)
    assert len(beats) == detection.n_beats


def test_unknown_delineation_lead_raises():
    recording = generate_recording(PRECISE, 0, 0)
    with pytest.raises(KeyError, match="delineation_lead"):
        _delineate(recording, dataclasses.replace(SP, delineation_lead="V9"))


def test_t_search_window_scales_with_heart_rate():
    """Repolarisation takes longer at slow rates, so the search must widen."""
    slow = dataclasses.replace(
        CLEAN, heart_rate_bpm_mean=45.0, heart_rate_bpm_sd=0.0, hr_change_bpm_per_decade=0.0
    )
    fast = dataclasses.replace(
        CLEAN, heart_rate_bpm_mean=100.0, heart_rate_bpm_sd=0.0, hr_change_bpm_per_decade=0.0
    )

    def median_qt(config) -> float:
        values = [
            beat.qt_interval_samples * 2.0
            for r in generate_cohort(config, n_recordings=6)
            for beat in _delineate(r)
            if beat.qt_interval_samples is not None
        ]
        return float(np.median(values))

    assert median_qt(slow) > median_qt(fast) + 20.0


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def test_wave_extent_on_a_hand_built_triangle():
    """A symmetric triangle: 20% of peak is reached 20% of the way up each side."""
    signal = np.concatenate([np.zeros(10), np.linspace(0, 1, 11), np.linspace(1, 0, 11)[1:], np.zeros(10)])
    peak = int(np.argmax(signal))
    onset, offset = _wave_extent(signal, peak, 0.0, 0, signal.size - 1, 0.2)
    assert onset == pytest.approx(12, abs=1)     # 20% up the rising limb
    assert offset == pytest.approx(28, abs=1)


def test_tangent_boundary_on_a_hand_built_triangle():
    """On a straight limb the tangent is the limb, so it finds the exact corner."""
    signal = np.concatenate([np.zeros(20), np.linspace(0, 1, 21), np.linspace(1, 0, 21)[1:], np.zeros(20)])
    peak = int(np.argmax(signal))
    offset = _tangent_boundary(signal, peak, signal.size - 1, 0.0, fallback=-1, direction=+1)
    onset = _tangent_boundary(signal, peak, 0, 0.0, fallback=-1, direction=-1)
    assert offset == pytest.approx(60, abs=1)    # end of the falling limb
    assert onset == pytest.approx(20, abs=1)     # start of the rising limb


def test_tangent_boundary_falls_back_on_a_flat_limb():
    """A flat signal has no usable tangent; the fallback must be returned."""
    flat = np.zeros(100)
    assert _tangent_boundary(flat, 50, 99, 0.0, fallback=77, direction=+1) == 77


def test_wave_boundaries_helpers():
    wave = WaveBoundaries(onset=10, peak=15, offset=22)
    assert wave.duration_samples == 12
    assert wave.contains(10) and wave.contains(22) and wave.contains(15)
    assert not wave.contains(9) and not wave.contains(23)


def test_delineation_signals_shapes_and_polarity():
    """The wave signal must keep polarity; the envelope must be non-negative."""
    recording = generate_recording(CLEAN, 0, 0)
    wave_signal, envelope = delineation_signals(
        recording.signal, 500.0, SP, recording.lead_names
    )
    assert wave_signal.shape == envelope.shape == (recording.n_samples,)
    assert (envelope >= 0).all()
    assert wave_signal.min() < 0 < wave_signal.max()
