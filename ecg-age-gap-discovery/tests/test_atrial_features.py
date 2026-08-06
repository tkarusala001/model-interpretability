"""Ground-truth tests for the extended atrial measurements.

Each measure is checked against a signal whose correct answer is known by
construction, and - just as important - against a signal where the correct answer
is *zero*. A notch detector that fires on every recording would inflate the known
-feature set with noise and make our own positive result look explained when it
is not, so the false-positive controls here carry as much weight as the
sensitivity ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecg_discovery.config import SignalProcessingConfig, SyntheticConfig
from ecg_discovery.data.synthetic_ecg import LEAD_NAMES, generate_cohort
from ecg_discovery.signal_processing.atrial_features import (
    ATRIAL_FEATURE_NAMES,
    _dispersion_from_durations,
    _notch_depth,
    _terminal_force,
    atrial_features,
    dispersion_noise_floor,
    per_lead_p_durations,
)
from ecg_discovery.signal_processing.interval_features import (
    INTERVAL_FEATURE_NAMES, compute_interval_features,
)
from ecg_discovery.signal_processing.qrs_detection import detect_r_peaks
from ecg_discovery.signal_processing.wave_delineation import delineate_beats


# --------------------------------------------------------------------------- #
# P terminal force
# --------------------------------------------------------------------------- #
def test_terminal_force_zero_when_p_wave_ends_positive():
    """A purely positive P wave has no terminal negativity, so the index is 0."""
    trace = np.array([0.0, 0.05, 0.10, 0.08, 0.03])
    assert _terminal_force(trace, samples_to_ms=2.0) == 0.0


def test_terminal_force_is_duration_times_depth():
    """Constructed biphasic P wave with a terminal negative run of known size."""
    # Four samples below baseline, deepest -0.05 mV, at 500 Hz (2 ms per sample).
    trace = np.array([0.10, 0.05, -0.01, -0.03, -0.05, -0.02])
    result = _terminal_force(trace, samples_to_ms=2.0)
    # 4 samples * 2 ms = 8 ms, depth 0.05 mV -> -0.40 mV.ms
    assert result == pytest.approx(-(8.0 * 0.05))


def test_terminal_force_more_negative_for_deeper_terminal_wave():
    """The index must order recordings the way cardiology reads it."""
    shallow = np.array([0.10, -0.01, -0.02])
    deep = np.array([0.10, -0.01, -0.08])
    assert _terminal_force(deep, 2.0) < _terminal_force(shallow, 2.0)


# --------------------------------------------------------------------------- #
# P notch depth - the false-positive control matters most
# --------------------------------------------------------------------------- #
def test_notch_depth_zero_on_smooth_single_humped_p_wave():
    """A clean raised-cosine P wave is not notched, and must not be reported so."""
    u = np.linspace(0.0, 1.0, 60)
    smooth = 0.15 * 0.5 * (1.0 - np.cos(2.0 * np.pi * u))
    assert _notch_depth(smooth, smooth_samples=4) == 0.0


def test_notch_depth_zero_on_noisy_but_unnotched_p_wave():
    """Sample-level noise creates many local maxima; none is a real notch."""
    rng = np.random.default_rng(0)
    u = np.linspace(0.0, 1.0, 60)
    smooth = 0.15 * 0.5 * (1.0 - np.cos(2.0 * np.pi * u))
    for seed_offset in range(10):
        noisy = smooth + rng.normal(0.0, 0.004, smooth.size)
        assert _notch_depth(noisy, smooth_samples=4) == 0.0, (
            f"spurious notch detected on noise realisation {seed_offset}"
        )


def test_notch_depth_detects_a_genuine_bifid_p_wave():
    """Two humps separated by a real trough must be reported with its depth."""
    u = np.linspace(0.0, 1.0, 60)
    first = 0.12 * np.exp(-((u - 0.30) ** 2) / (2 * 0.09 ** 2))
    second = 0.12 * np.exp(-((u - 0.70) ** 2) / (2 * 0.09 ** 2))
    bifid = first + second
    depth = _notch_depth(bifid, smooth_samples=3)
    assert depth > 0.0
    # The trough sits at the midpoint; check the reported depth matches it.
    expected = float(bifid[np.argmin(np.abs(u - 0.30))] - bifid[bifid.size // 2])
    assert depth == pytest.approx(expected, abs=0.01)


def test_notch_depth_grows_as_the_humps_separate():
    """Deeper inter-atrial delay means a deeper notch; the measure must track it."""
    u = np.linspace(0.0, 1.0, 80)
    depths = []
    for separation in (0.30, 0.40, 0.50):
        centre = 0.5
        a = 0.12 * np.exp(-((u - (centre - separation / 2)) ** 2) / (2 * 0.07 ** 2))
        b = 0.12 * np.exp(-((u - (centre + separation / 2)) ** 2) / (2 * 0.07 ** 2))
        depths.append(_notch_depth(a + b, smooth_samples=3))
    assert depths[0] < depths[1] < depths[2]


def test_notch_depth_handles_inverted_p_wave():
    """An inverted bifid P wave is still notched; polarity must not hide it."""
    u = np.linspace(0.0, 1.0, 60)
    a = 0.12 * np.exp(-((u - 0.30) ** 2) / (2 * 0.09 ** 2))
    b = 0.12 * np.exp(-((u - 0.70) ** 2) / (2 * 0.09 ** 2))
    assert _notch_depth(-(a + b), smooth_samples=3) > 0.0


# --------------------------------------------------------------------------- #
# P area
# --------------------------------------------------------------------------- #
def _measure(signals, fs, config, include_dispersion=True):
    return compute_interval_features(
        signals, fs, config, LEAD_NAMES, include_dispersion=include_dispersion
    )


def test_p_area_scales_with_amplitude_and_duration():
    """Area is the product of height and width, so both must move it.

    This is checked as an invariance rather than against an absolute value
    because each lead is a projection of the underlying P component with its own
    coefficient, so the absolute area in lead II is not known a priori. The
    scaling relationship is, and it is what would break if the integration window
    or the baseline were wrong.
    """
    fs = 500.0
    signal_config = SignalProcessingConfig()

    def area_for(amplitude_mv: float, duration_ms: float) -> float:
        config = SyntheticConfig(
            n_recordings=1, sampling_rate_hz=fs, duration_seconds=10.0,
            p_amplitude_mv=amplitude_mv, p_duration_ms=duration_ms,
            noise_mv_sd=0.0, baseline_wander_mv=0.0, powerline_mv=0.0, seed=3,
        )
        cohort = generate_cohort(config)
        return _measure(cohort.signals[0], fs, signal_config, False).p_area_ii_mv_ms

    base = area_for(0.15, 100.0)
    taller = area_for(0.30, 100.0)
    wider = area_for(0.15, 140.0)

    assert np.isfinite(base) and base > 0
    assert taller == pytest.approx(2.0 * base, rel=0.12)
    assert wider == pytest.approx(1.4 * base, rel=0.15)


def test_p_area_is_not_recoverable_from_duration_and_amplitude_alone():
    """If area were a deterministic function of the other two it would be useless.

    The value of adding it to the known-feature set rests on it carrying
    information those two do not. Across a cohort with varied morphology the
    residual of area on duration and amplitude must therefore be non-trivial.
    """
    fs = 500.0
    config = SyntheticConfig(
        n_recordings=60, sampling_rate_hz=fs, duration_seconds=10.0, seed=11
    )
    cohort = generate_cohort(config)
    signal_config = SignalProcessingConfig()
    rows = [_measure(s, fs, signal_config, False) for s in cohort.signals]

    area = np.array([r.p_area_ii_mv_ms for r in rows])
    predictors = np.column_stack([
        np.ones(len(rows)),
        [r.p_duration_ms for r in rows],
        [r.p_amplitude_mv for r in rows],
    ])
    ok = np.isfinite(area) & np.isfinite(predictors).all(axis=1)
    assert ok.sum() > 30

    coefficients, *_ = np.linalg.lstsq(predictors[ok], area[ok], rcond=None)
    residual = area[ok] - predictors[ok] @ coefficients
    r_squared = 1.0 - residual.var() / area[ok].var()
    # Strongly related, as expected, but not a perfect function of the two.
    assert r_squared < 0.999


# --------------------------------------------------------------------------- #
# P dispersion - the measurement whose ground truth is zero
# --------------------------------------------------------------------------- #
def test_synthetic_leads_have_zero_true_dispersion_by_construction():
    """The premise the noise-floor argument rests on, verified rather than assumed.

    Every synthetic lead is a fixed linear combination of the same three
    components, so all leads share one P-wave onset and offset in time. If this
    ever stopped being true the noise-floor estimate would silently become
    meaningless, so it is asserted here.
    """
    # All three interference sources off. Each is added per lead with an
    # independent phase, so any of them breaks proportionality - powerline
    # interference at 0.005 mV is already 3% of a 0.16 mV P wave.
    config = SyntheticConfig(
        n_recordings=1, sampling_rate_hz=500.0, duration_seconds=10.0,
        noise_mv_sd=0.0, baseline_wander_mv=0.0, powerline_mv=0.0, seed=5,
    )
    cohort = generate_cohort(config)
    signals = cohort.signals[0].astype(np.float64)

    # Leads are *not* scalar multiples of each other overall - each weights the
    # P, QRS and T components differently. The premise is narrower: inside the P
    # window the other two components are identically zero, so there every lead
    # is a scalar multiple of the one shared P component, and therefore shares
    # its onset and offset exactly.
    signal_config = SignalProcessingConfig()
    detection = detect_r_peaks(signals, 500.0, signal_config, LEAD_NAMES)
    beats = delineate_beats(signals, 500.0, detection.r_peaks, signal_config, LEAD_NAMES)
    with_p = [beat for beat in beats if beat.p_wave is not None]
    assert with_p, "no P wave delineated on a clean synthetic recording"

    beat = with_p[len(with_p) // 2]
    window = slice(beat.p_wave.onset, beat.p_wave.offset + 1)
    reference = signals[1, window]                           # lead II
    assert np.dot(reference, reference) > 0

    for lead in range(signals.shape[0]):
        trace = signals[lead, window]
        if np.allclose(trace, 0.0, atol=1e-9):
            continue
        scale = np.dot(trace, reference) / np.dot(reference, reference)
        residual = float(np.abs(trace - scale * reference).max())
        assert residual < 1e-3 * float(np.abs(reference).max()), (
            f"lead {LEAD_NAMES[lead]} is not proportional to lead II inside the "
            "P window; the zero-true-dispersion premise no longer holds"
        )


def test_dispersion_noise_floor_is_measured_not_assumed():
    """On a cohort whose true dispersion is zero, what we measure is our error."""
    fs = 500.0
    config = SyntheticConfig(n_recordings=12, sampling_rate_hz=fs, duration_seconds=10.0, seed=7)
    cohort = generate_cohort(config)
    signal_config = SignalProcessingConfig()

    per_lead = []
    for signals in cohort.signals:
        detection = detect_r_peaks(signals, fs, signal_config, LEAD_NAMES)
        beats = delineate_beats(signals, fs, detection.r_peaks, signal_config, LEAD_NAMES)
        per_lead.append(per_lead_p_durations(signals, beats, fs, signal_config))

    floor = dispersion_noise_floor(per_lead)
    # The true value is 0 ms. We do not assert the floor is small - the point of
    # the measurement is to find out. We assert only that it is finite, so that
    # the number reported in the paper is a real one.
    assert np.isfinite(floor)
    assert floor >= 0.0


def test_dispersion_requires_a_majority_of_leads():
    """Two surviving leads is not a spread; it must not be reported as one."""
    durations = np.full((1, 12), np.nan)
    durations[0, 0] = 90.0
    durations[0, 1] = 130.0
    assert not np.isfinite(_dispersion_from_durations(durations))

    durations[0, 2:8] = 100.0
    assert np.isfinite(_dispersion_from_durations(durations))


# --------------------------------------------------------------------------- #
# Integration
# --------------------------------------------------------------------------- #
def test_atrial_features_are_exposed_in_the_feature_table():
    """The new measures must reach the known-feature set, not stop at the module."""
    for name in ATRIAL_FEATURE_NAMES:
        assert name in INTERVAL_FEATURE_NAMES

    fs = 500.0
    config = SyntheticConfig(n_recordings=1, sampling_rate_hz=fs, duration_seconds=10.0, seed=1)
    cohort = generate_cohort(config)
    features = _measure(cohort.signals[0], fs, SignalProcessingConfig())
    for name in ATRIAL_FEATURE_NAMES:
        assert hasattr(features, name)


def test_missing_leads_give_nan_not_a_fabricated_value():
    """Without V1 there is no terminal force; inventing one would corrupt the set."""
    fs = 500.0
    config = SyntheticConfig(n_recordings=1, sampling_rate_hz=fs, duration_seconds=10.0, seed=2)
    cohort = generate_cohort(config)
    signals = cohort.signals[0][:2]                          # leads I and II only

    detection = detect_r_peaks(signals, fs, SignalProcessingConfig(), LEAD_NAMES[:2])
    beats = delineate_beats(
        signals, fs, detection.r_peaks, SignalProcessingConfig(), LEAD_NAMES[:2]
    )
    result = atrial_features(
        signals, beats, LEAD_NAMES[:2], fs, SignalProcessingConfig(),
        include_dispersion=False,
    )
    assert np.isnan(result["p_terminal_force_v1_mv_ms"])
    assert np.isfinite(result["p_area_ii_mv_ms"])


def test_no_p_wave_gives_nan_throughout():
    """Atrial fibrillation has no organised P wave; the measures must say so."""
    fs = 500.0
    config = SyntheticConfig(n_recordings=1, sampling_rate_hz=fs, duration_seconds=10.0, seed=4)
    cohort = generate_cohort(config)
    signals = cohort.signals[0]
    detection = detect_r_peaks(signals, fs, SignalProcessingConfig(), LEAD_NAMES)
    beats = delineate_beats(signals, fs, detection.r_peaks, SignalProcessingConfig(), LEAD_NAMES)
    stripped = [
        type(beat)(r_peak=beat.r_peak, qrs=beat.qrs, p_wave=None,
                   t_wave=beat.t_wave, baseline_mv=beat.baseline_mv)
        for beat in beats
    ]
    result = atrial_features(signals, stripped, LEAD_NAMES, fs, SignalProcessingConfig())
    assert all(np.isnan(value) for value in result.values())
