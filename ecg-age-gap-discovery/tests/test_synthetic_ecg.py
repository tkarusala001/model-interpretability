"""Tests for the synthetic ECG generator.

The generator is the answer key for every later phase, so these tests are more
than a smoke check. They verify three separate things:

1. **Reproducibility** - same seed, same data; and a small cohort is a genuine
   prefix of a large one, so tests and full runs see identical recordings.
2. **Construction fidelity** - the fiducial points and intervals the generator
   reports really are the ones present in the waveform.
3. **That the injected age effects are actually there** - a sanity check on our
   own generator. If the age correlation we believe we injected were absent, the
   downstream ground-truth tests would be vacuous: they would "pass" against a
   signal that does not exist.

The most important test in this file is
``test_t_wave_skew_is_invisible_to_timing_intervals``. The Phase 8 validation
framework can only be scored if the synthetic "discovery candidate" is genuinely
undetectable by classical interval measurement, and that test is what
establishes it.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from ecg_discovery.config import SyntheticConfig
from ecg_discovery.data.synthetic_ecg import (
    DIAGNOSTIC_CLASSES,
    LEAD_NAMES,
    _bump,
    _peak_position,
    generate_cohort,
    generate_recording,
    load_cohort,
    save_cohort,
)

# A fast configuration for tests that do not need 500 Hz resolution.
FAST = SyntheticConfig(sampling_rate_hz=100, n_recordings=16)
# Full resolution, for tests that compare measured against constructed timings.
PRECISE = SyntheticConfig(sampling_rate_hz=500, n_recordings=8)
# Noise-free, for tests of exact algebraic properties of the clean signal.
CLEAN = dataclasses.replace(
    PRECISE, noise_mv_sd=0.0, baseline_wander_mv=0.0, powerline_mv=0.0
)


# --------------------------------------------------------------------------- #
# The bump primitive: the basis of the whole waveform model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("skew", [-0.8, -0.3, 0.0, 0.3, 0.8])
def test_bump_has_compact_support_and_unit_peak(skew):
    """Zero at both ends and peak exactly 1, whatever the skew.

    Compact support is what makes wave onsets and offsets exact rather than
    threshold-dependent; unit peak is what makes skew an amplitude-preserving
    change.
    """
    u = np.linspace(0.0, 1.0, 20001)
    values = _bump(u, skew)
    assert values[0] == pytest.approx(0.0, abs=1e-12)
    assert values[-1] == pytest.approx(0.0, abs=1e-12)
    assert values.max() == pytest.approx(1.0, abs=1e-6)
    assert values.min() >= -1e-12


@pytest.mark.parametrize("skew", [-0.8, -0.3, 0.0, 0.3, 0.8])
def test_bump_peak_position_matches_analytic_formula(skew):
    u = np.linspace(0.0, 1.0, 200001)
    empirical = u[int(np.argmax(_bump(u, skew)))]
    assert empirical == pytest.approx(_peak_position(skew), abs=1e-4)


def test_skew_moves_the_peak_monotonically():
    """Positive skew must push the peak later, negative earlier."""
    positions = [_peak_position(s) for s in (-0.6, -0.2, 0.0, 0.2, 0.6)]
    assert positions == sorted(positions)
    assert _peak_position(0.0) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Schema and determinism
# --------------------------------------------------------------------------- #
def test_recording_schema():
    rec = generate_recording(PRECISE, 0, 0)
    assert rec.signal.shape == (12, PRECISE.n_samples)
    assert rec.signal.dtype == np.float32
    assert np.isfinite(rec.signal).all()
    assert rec.lead_names == LEAD_NAMES
    assert rec.diagnostic_labels.shape == (len(DIAGNOSTIC_CLASSES),)
    assert rec.diagnostic_labels.sum() == 1
    assert PRECISE.age_min_years <= rec.age_years <= PRECISE.age_max_years
    assert rec.sex in (0, 1)
    assert rec.n_samples == 5000
    assert rec.duration_seconds == pytest.approx(10.0)


def test_generation_is_deterministic():
    a = generate_cohort(FAST, n_recordings=6)
    b = generate_cohort(FAST, n_recordings=6)
    np.testing.assert_array_equal(a.signals, b.signals)
    np.testing.assert_array_equal(a.ages, b.ages)
    np.testing.assert_array_equal(a.diagnostic_labels, b.diagnostic_labels)


def test_different_seed_gives_different_data():
    other = dataclasses.replace(FAST, seed=FAST.seed + 1)
    a = generate_cohort(FAST, n_recordings=6)
    b = generate_cohort(other, n_recordings=6)
    assert not np.allclose(a.signals, b.signals)


def test_small_cohort_is_a_prefix_of_a_large_one():
    """Recording i must not depend on cohort size, so tests see real recordings."""
    small = generate_cohort(FAST, n_recordings=4)
    large = generate_cohort(FAST, n_recordings=32)
    np.testing.assert_array_equal(small.signals, large.signals[:4])
    np.testing.assert_array_equal(small.patient_ids, large.patient_ids[:4])


def test_unknown_lead_name_raises():
    with pytest.raises(KeyError, match="unknown lead"):
        generate_recording(FAST, 0, 0).lead("V7")


# --------------------------------------------------------------------------- #
# Construction fidelity: are the reported fiducials really in the waveform?
# --------------------------------------------------------------------------- #
def test_fiducials_are_strictly_ordered():
    for rec in generate_cohort(PRECISE, n_recordings=4):
        for beat in rec.beats:
            f = beat
            assert f.p_onset < f.p_peak < f.p_offset <= f.qrs_onset
            assert f.qrs_onset < f.q_peak < f.r_peak < f.s_peak < f.qrs_offset
            assert f.qrs_offset <= f.t_onset < f.t_peak < f.t_offset
            assert 0 <= f.p_onset and f.t_offset < rec.n_samples


def test_constructed_intervals_match_reported_intervals():
    """Fiducial spacing must reproduce the reported PR / QRS / QT to <= 1 sample."""
    tolerance_ms = 1000.0 / PRECISE.sampling_rate_hz  # 2 ms at 500 Hz
    for rec in generate_cohort(PRECISE, n_recordings=6):
        fs = rec.sampling_rate_hz
        beat = rec.beats[1]
        pr = (beat.qrs_onset - beat.p_onset) / fs * 1000.0
        qrs = (beat.qrs_offset - beat.qrs_onset) / fs * 1000.0
        qt = (beat.t_offset - beat.qrs_onset) / fs * 1000.0
        assert pr == pytest.approx(rec.true_intervals["pr_interval_ms"], abs=tolerance_ms)
        assert qrs == pytest.approx(rec.true_intervals["qrs_duration_ms"], abs=tolerance_ms)
        # QT varies beat to beat with RR, so compare against this beat's own span
        # rather than the recording mean, allowing for that variation.
        assert qt == pytest.approx(rec.true_intervals["qt_interval_ms"], abs=25.0)


def test_r_peak_is_the_waveform_maximum_in_lead_ii():
    """The reported R-peak index must really be the local maximum of lead II.

    Lead II is the standard rhythm lead and has a positive QRS weight, so its
    maximum inside the complex is the R wave. This is the property an R-peak
    detector will be scored against in Phase 2, so it has to hold here first.
    """
    for rec in generate_cohort(CLEAN, n_recordings=4):
        lead_ii = rec.lead("II")
        for beat in rec.beats:
            window = slice(beat.qrs_onset, beat.qrs_offset + 1)
            local_max = beat.qrs_onset + int(np.argmax(lead_ii[window]))
            assert abs(local_max - beat.r_peak) <= 1


def test_signal_is_isoelectric_between_beats():
    """Outside P-QRS-T the clean signal must sit at zero (compact support)."""
    rec = generate_recording(CLEAN, 0, 0)
    lead_ii = rec.lead("II")
    beats = rec.beats
    assert len(beats) >= 3
    # The gap between one beat's T offset and the next beat's P onset.
    gap = lead_ii[beats[1].t_offset + 1 : beats[2].p_onset]
    assert gap.size > 5
    assert np.abs(gap).max() < 1e-6


def test_einthoven_and_goldberger_relations_hold_in_clean_signal():
    """The 12 leads must satisfy the algebraic constraints of a real recording.

    III = II - I, aVR = -(I + II)/2, aVL = I - II/2, aVF = II - I/2.
    These hold exactly only without per-lead noise, which is also true of real
    ECGs; the test therefore uses the noise-free configuration.
    """
    for rec in generate_cohort(CLEAN, n_recordings=3):
        lead_i, lead_ii = rec.lead("I"), rec.lead("II")
        np.testing.assert_allclose(rec.lead("III"), lead_ii - lead_i, atol=1e-5)
        np.testing.assert_allclose(rec.lead("aVR"), -(lead_i + lead_ii) / 2, atol=1e-5)
        np.testing.assert_allclose(rec.lead("aVL"), lead_i - lead_ii / 2, atol=1e-5)
        np.testing.assert_allclose(rec.lead("aVF"), lead_ii - lead_i / 2, atol=1e-5)


def test_noise_breaks_einthoven_relation():
    """Guard against the noise-free test passing trivially because noise is off."""
    rec = generate_recording(PRECISE, 0, 0)
    assert not np.allclose(rec.lead("III"), rec.lead("II") - rec.lead("I"), atol=1e-3)


def test_v1_shows_rs_pattern_and_inverted_t():
    """A physiological sanity check that the lead projections read as an ECG."""
    rec = generate_recording(CLEAN, 0, 0)
    v1, beat = rec.lead("V1"), generate_recording(CLEAN, 0, 0).beats[1]
    qrs = v1[beat.qrs_onset : beat.qrs_offset + 1]
    assert abs(qrs.min()) > abs(qrs.max())          # dominant S, not R
    t_wave = v1[beat.t_onset : beat.t_offset + 1]
    assert t_wave.min() < 0 and abs(t_wave.min()) > abs(t_wave.max())  # inverted T


def test_r_peaks_include_edge_beats_but_complete_beats_do_not():
    """R peaks cover every beat; full fiducials only cover uncut cycles."""
    rec = generate_recording(PRECISE, 0, 0)
    complete_r = {beat.r_peak for beat in rec.beats}
    assert complete_r.issubset(set(rec.r_peak_samples.tolist()))
    assert len(rec.r_peak_samples) >= len(rec.beats)


def test_rr_intervals_are_consistent_with_reported_heart_rate():
    for rec in generate_cohort(PRECISE, n_recordings=4):
        rr_ms = np.diff(rec.r_peak_samples) / rec.sampling_rate_hz * 1000.0
        assert 60_000.0 / rr_ms.mean() == pytest.approx(
            rec.true_intervals["heart_rate_bpm"], rel=1e-6
        )


# --------------------------------------------------------------------------- #
# Sampling-rate independence
# --------------------------------------------------------------------------- #
def test_same_recording_at_100hz_and_500hz():
    """Both rates must describe the same underlying recording, as PTB-XL does.

    Morphology parameters are drawn before any rate-dependent quantity, so a
    recording generated at 100 Hz and at 500 Hz differs only in rasterisation.
    This is what lets the model train at 100 Hz while intervals are measured at
    500 Hz.
    """
    low = dataclasses.replace(CLEAN, sampling_rate_hz=100)
    high = dataclasses.replace(CLEAN, sampling_rate_hz=500)
    for index in range(4):
        a = generate_recording(low, index, index)
        b = generate_recording(high, index, index)
        assert a.age_years == pytest.approx(b.age_years)
        assert a.sex == b.sex
        assert a.known_age_offset_years == pytest.approx(b.known_age_offset_years)
        assert len(a.r_peak_samples) == len(b.r_peak_samples)
        for key in ("qrs_duration_ms", "pr_interval_ms", "t_wave_skew"):
            assert a.true_intervals[key] == pytest.approx(b.true_intervals[key])
        # Heart rate is derived from rounded sample indices, so it agrees only
        # to within the coarser grid's resolution (10 ms at 100 Hz).
        assert a.true_intervals["heart_rate_bpm"] == pytest.approx(
            b.true_intervals["heart_rate_bpm"], rel=0.02
        )


# --------------------------------------------------------------------------- #
# Are the injected age effects actually present?
# --------------------------------------------------------------------------- #
def _corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def test_injected_qrs_widening_is_present():
    """Older synthetic patients really do have wider QRS complexes."""
    cohort = generate_cohort(FAST, n_recordings=300)
    meta = cohort.metadata()
    assert _corr(meta["age"].to_numpy(), meta["true_qrs_duration_ms"].to_numpy()) > 0.5
    # And the effect size is the one configured: 5 ms per decade.
    fit = np.polyfit(meta["age"], meta["true_qrs_duration_ms"], 1)
    assert fit[0] * 10.0 == pytest.approx(FAST.qrs_widening_ms_per_decade, rel=0.15)


def test_injected_heart_rate_and_skew_effects_are_present():
    cohort = generate_cohort(FAST, n_recordings=300)
    meta = cohort.metadata()
    assert _corr(meta["age"].to_numpy(), meta["true_heart_rate_bpm"].to_numpy()) < -0.2
    assert _corr(meta["age"].to_numpy(), meta["true_t_wave_skew"].to_numpy()) > 0.8


def test_qrs_width_tracks_the_known_offset_not_the_unexplained_one():
    """The two latent offsets must drive strictly separate morphology channels.

    If they leaked into each other, "explained" and "unexplained" variance would
    be entangled and the Phase 8 tests would be measuring nothing.
    """
    cohort = generate_cohort(FAST, n_recordings=400)
    meta = cohort.metadata()
    resid_qrs = meta["true_qrs_duration_ms"] - np.poly1d(
        np.polyfit(meta["age"], meta["true_qrs_duration_ms"], 1)
    )(meta["age"])
    known = meta["true_known_age_offset"].to_numpy()
    unexplained = meta["true_unexplained_age_offset"].to_numpy()
    assert _corr(resid_qrs.to_numpy(), known) > 0.9
    assert abs(_corr(resid_qrs.to_numpy(), unexplained)) < 0.15


def test_t_wave_skew_is_invisible_to_timing_intervals():
    """The synthetic discovery candidate must be undetectable by interval timing.

    This is the load-bearing test of the whole ground-truth design. T-wave skew
    is generated by warping the *shape* of the T wave while its onset, offset,
    duration and peak amplitude stay fixed, so if the Phase 8 decomposition ever
    "explains" it using PR/QRS/QT/heart rate, that is a bug in the
    decomposition rather than a real finding. Here we confirm the premise: the
    skew carries no timing signature.
    """
    config = dataclasses.replace(
        FAST,
        known_age_offset_sd_years=0.0,        # isolate the unexplained channel
        unexplained_age_offset_sd_years=10.0,
    )
    meta = generate_cohort(config, n_recordings=400).metadata()

    # Remove the shared dependence on chronological age, leaving only the part
    # of each quantity driven by the unexplained offset.
    def residualise(column: str) -> np.ndarray:
        values = meta[column].to_numpy()
        ages = meta["age"].to_numpy()
        return values - np.poly1d(np.polyfit(ages, values, 1))(ages)

    offset = meta["true_unexplained_age_offset"].to_numpy()
    assert _corr(residualise("true_t_wave_skew"), offset) > 0.95   # the skew IS there
    for interval in (
        "true_qrs_duration_ms",
        "true_pr_interval_ms",
        "true_qt_interval_ms",
        "true_heart_rate_bpm",
    ):
        column = meta[interval].to_numpy()
        if np.ptp(column) < 1e-9:      # PR is constant by construction
            continue
        assert abs(_corr(residualise(interval), offset)) < 0.15, (
            f"{interval} leaks information about the unexplained offset; the "
            "synthetic discovery candidate would not be a valid ground truth"
        )


def test_t_wave_skew_preserves_amplitude_and_support_in_the_waveform():
    """Check invisibility directly on the samples, not just on reported values.

    The heart-rate age effect is switched off here on purpose. T-wave duration
    legitimately scales with sqrt(RR) (Bazett-style), so a slower older heart
    has a genuinely longer T wave - a *rate* effect, carried by a known feature.
    The invariant under test is the separate claim that skew on its own changes
    neither duration nor amplitude, so the rate channel is held fixed to isolate
    it.
    """
    base = dataclasses.replace(
        CLEAN, known_age_offset_sd_years=0.0, unexplained_age_offset_sd_years=0.0,
        age_min_years=30.0, age_max_years=30.0001, heart_rate_bpm_sd=0.0,
        hr_change_bpm_per_decade=0.0,
    )
    aged = dataclasses.replace(base, age_min_years=85.0, age_max_years=85.0001)

    young_rec, old_rec = generate_recording(base, 0, 0), generate_recording(aged, 0, 0)
    yb, ob = young_rec.beats[1], old_rec.beats[1]

    # Same T-wave onset, offset and duration despite very different skew.
    assert abs(young_rec.true_intervals["t_wave_skew"]
               - old_rec.true_intervals["t_wave_skew"]) > 0.4
    assert (ob.t_offset - ob.t_onset) == pytest.approx(yb.t_offset - yb.t_onset, abs=1)

    # Same peak amplitude, but the peak occurs at a different point in the wave.
    y_wave = young_rec.lead("II")[yb.t_onset : yb.t_offset + 1]
    o_wave = old_rec.lead("II")[ob.t_onset : ob.t_offset + 1]
    assert o_wave.max() == pytest.approx(y_wave.max(), rel=0.02)
    y_frac = int(np.argmax(y_wave)) / y_wave.size
    o_frac = int(np.argmax(o_wave)) / o_wave.size
    assert o_frac - y_frac > 0.08


# --------------------------------------------------------------------------- #
# Patients, and the split hazard they create
# --------------------------------------------------------------------------- #
def test_some_patients_contribute_multiple_recordings():
    """PTB-XL's repeat patients are what make recording-level splitting a bug."""
    cohort = generate_cohort(FAST, n_recordings=200)
    ids = cohort.patient_ids
    _, counts = np.unique(ids, return_counts=True)
    assert counts.max() > 1
    assert (counts > 1).sum() >= 5


def test_repeat_recordings_share_demographics_and_offsets():
    """Two recordings of one patient must be genuinely the same person."""
    cohort = generate_cohort(FAST, n_recordings=200)
    by_patient: dict[int, list] = {}
    for rec in cohort:
        by_patient.setdefault(rec.patient_id, []).append(rec)
    repeats = [v for v in by_patient.values() if len(v) > 1]
    assert repeats, "expected at least one repeat patient"
    for group in repeats:
        first = group[0]
        for other in group[1:]:
            assert other.age_years == pytest.approx(first.age_years)
            assert other.sex == first.sex
            assert other.known_age_offset_years == pytest.approx(
                first.known_age_offset_years
            )
            # ...but the recordings themselves differ (separate visits).
            assert not np.allclose(other.signal, first.signal)


# --------------------------------------------------------------------------- #
# Diagnostic labels for the Phase 9 machinery check
# --------------------------------------------------------------------------- #
def test_diagnostic_labels_are_one_hot_over_the_superclasses():
    cohort = generate_cohort(FAST, n_recordings=100)
    labels = cohort.diagnostic_labels
    assert labels.shape == (100, len(DIAGNOSTIC_CLASSES))
    np.testing.assert_array_equal(labels.sum(axis=1), np.ones(100, dtype=np.int64))


def test_diagnostic_link_can_be_switched_on_and_off():
    """Phase 9 must be verifiable in both directions, so the link is configurable."""
    linked = dataclasses.replace(
        FAST, diagnostic_link_source="unexplained", diagnostic_link_strength=1.5
    )
    unlinked = dataclasses.replace(FAST, diagnostic_link_source="none")

    def abnormality_correlation(config) -> float:
        meta = generate_cohort(config, n_recordings=600).metadata()
        abnormal = 1 - meta["dx_NORM"].to_numpy()
        return _corr(abnormal.astype(float), meta["true_unexplained_age_offset"].to_numpy())

    assert abnormality_correlation(linked) > 0.2
    assert abs(abnormality_correlation(unlinked)) < 0.1


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_save_and_load_round_trip(tmp_path):
    original = generate_cohort(FAST, n_recordings=5)
    save_cohort(original, tmp_path / "cohort")
    restored = load_cohort(tmp_path / "cohort")

    assert len(restored) == len(original)
    np.testing.assert_array_equal(restored.signals, original.signals)
    np.testing.assert_allclose(restored.ages, original.ages)
    np.testing.assert_array_equal(restored.patient_ids, original.patient_ids)
    np.testing.assert_array_equal(restored.diagnostic_labels, original.diagnostic_labels)
    assert restored.config == original.config
    for a, b in zip(restored.recordings, original.recordings):
        assert a.record_id == b.record_id
        assert a.beats == b.beats
        np.testing.assert_array_equal(a.r_peak_samples, b.r_peak_samples)
        for key, value in b.true_intervals.items():
            assert a.true_intervals[key] == pytest.approx(value)
