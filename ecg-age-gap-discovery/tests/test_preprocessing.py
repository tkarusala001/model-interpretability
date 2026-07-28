"""Tests for preprocessing and patient-level splitting.

The build plan singles out patient-level splitting as a real, easy-to-miss bug,
and it is: splitting by recording rather than by patient produces no error, no
warning, and a *better-looking* result. Several tests here therefore go beyond
checking that the split is correct and demonstrate what goes wrong without it -
including ``test_recording_level_split_would_leak_and_inflate_accuracy``, which
measures the inflation directly.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ecg_discovery.config import DataConfig, SyntheticConfig
from ecg_discovery.data.preprocessing import (
    LeadNormalizer,
    SplitIndices,
    assert_no_patient_leakage,
    bandpass_signals,
    patient_level_split,
    resample_signals,
)
from ecg_discovery.data.synthetic_ecg import generate_cohort

DATA = DataConfig()


# --------------------------------------------------------------------------- #
# Patient-level splitting: the correctness that matters most
# --------------------------------------------------------------------------- #
def test_no_patient_appears_in_two_splits():
    """The headline requirement, on a cohort that genuinely contains repeats."""
    cohort = generate_cohort(SyntheticConfig(sampling_rate_hz=100), n_recordings=400)
    patient_ids = cohort.patient_ids
    # Guard: if the cohort had no repeat patients the test would be vacuous.
    _, counts = np.unique(patient_ids, return_counts=True)
    assert (counts > 1).sum() >= 5

    splits = patient_level_split(patient_ids, DATA)
    train = set(patient_ids[splits.train].tolist())
    val = set(patient_ids[splits.val].tolist())
    test = set(patient_ids[splits.test].tolist())
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()


def test_every_recording_is_used_exactly_once():
    patient_ids = np.repeat(np.arange(60), 2)
    splits = patient_level_split(patient_ids, DATA)
    covered = np.concatenate([splits.train, splits.val, splits.test])
    assert sorted(covered.tolist()) == list(range(patient_ids.size))


def test_all_recordings_of_a_patient_land_together():
    """Construct a patient with several recordings and check they stay as one."""
    patient_ids = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    splits = patient_level_split(patient_ids, DATA, seed=5)
    memberships = [
        name for name, indices in
        (("train", splits.train), ("val", splits.val), ("test", splits.test))
        if 0 in set(patient_ids[indices].tolist())
    ]
    assert len(memberships) == 1
    home = {"train": splits.train, "val": splits.val, "test": splits.test}[memberships[0]]
    assert set(np.flatnonzero(patient_ids == 0).tolist()).issubset(set(home.tolist()))


def test_recording_level_split_would_leak_and_inflate_accuracy():
    """Show the bug this design prevents, and quantify what it would buy.

    A recording-level split puts some patients' repeat recordings in both
    training and test. Those test recordings are then *free marks*: a model that
    has memorised a patient's age from one visit can recite it for another
    without reading the ECG at all. Because a patient's age and latent ECG-age
    offsets are identical across their visits, a memoriser scores exactly zero
    error on every leaked recording.

    The test measures that directly - the fraction of test recordings whose
    patient was also seen in training, and the error an oracle memoriser would
    achieve on them - rather than training a model to exhibit the effect.
    Nothing about the leaking version raises an error, which is precisely why it
    has to be checked for.
    """
    config = dataclasses.replace(
        SyntheticConfig(sampling_rate_hz=100), repeat_patient_frac=0.6
    )
    cohort = generate_cohort(config, n_recordings=240)
    ages = cohort.ages
    patient_ids = cohort.patient_ids

    def leaked_fraction(train_idx: np.ndarray, test_idx: np.ndarray) -> float:
        seen = set(patient_ids[train_idx].tolist())
        return float(np.mean([pid in seen for pid in patient_ids[test_idx]]))

    rng = np.random.default_rng(0)
    shuffled = rng.permutation(len(cohort))
    cut = int(0.75 * len(cohort))
    naive_leak = leaked_fraction(shuffled[:cut], shuffled[cut:])

    splits = patient_level_split(patient_ids, DATA)
    honest_leak = leaked_fraction(splits.train, splits.test)

    assert naive_leak > 0.2, (
        f"only {naive_leak:.1%} of the naive test split was leaked; the cohort "
        "may not contain enough repeat patients for this test to mean anything"
    )
    assert honest_leak == 0.0

    # A memoriser is perfectly correct on exactly the leaked recordings, so the
    # naive split hands it a share of the test set for free.
    seen = set(patient_ids[shuffled[:cut]].tolist())
    test_ids = patient_ids[shuffled[cut:]]
    leaked_mask = np.array([pid in seen for pid in test_ids])
    memoriser_error = np.where(leaked_mask, 0.0, np.abs(ages[shuffled[cut:]] - ages.mean()))
    honest_error = np.abs(ages[splits.test] - ages.mean())
    assert memoriser_error.mean() < 0.85 * honest_error.mean()


def test_split_is_deterministic_and_seed_dependent():
    patient_ids = np.repeat(np.arange(80), 2)
    first = patient_level_split(patient_ids, DATA, seed=1)
    again = patient_level_split(patient_ids, DATA, seed=1)
    different = patient_level_split(patient_ids, DATA, seed=2)

    np.testing.assert_array_equal(first.train, again.train)
    np.testing.assert_array_equal(first.test, again.test)
    assert not np.array_equal(first.test, different.test)


def test_split_proportions_are_approximately_respected():
    """Exact proportions are impossible: only whole patients can be moved."""
    cohort = generate_cohort(SyntheticConfig(sampling_rate_hz=100), n_recordings=600)
    splits = patient_level_split(cohort.patient_ids, DATA)
    total = len(cohort)
    assert abs(splits.train.size / total - DATA.train_frac) < 0.05
    assert abs(splits.val.size / total - DATA.val_frac) < 0.05
    assert abs(splits.test.size / total - DATA.test_frac) < 0.05


def test_split_age_distributions_are_comparable():
    """A split that happened to concentrate old patients in test would mislead."""
    cohort = generate_cohort(SyntheticConfig(sampling_rate_hz=100), n_recordings=600)
    splits = patient_level_split(cohort.patient_ids, DATA)
    ages = cohort.ages
    assert abs(ages[splits.train].mean() - ages[splits.test].mean()) < 5.0
    assert abs(ages[splits.train].std() - ages[splits.test].std()) < 5.0


def test_leakage_check_catches_a_corrupted_split():
    """The self-check must actually fire when a patient is in two splits."""
    patient_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    corrupted = SplitIndices(
        train=np.array([0, 2]), val=np.array([1, 4, 5]), test=np.array([3, 6, 7])
    )
    with pytest.raises(ValueError, match="leakage"):
        assert_no_patient_leakage(patient_ids, corrupted)


def test_leakage_check_catches_an_incomplete_partition():
    patient_ids = np.arange(8)
    incomplete = SplitIndices(
        train=np.array([0, 1]), val=np.array([2]), test=np.array([3])
    )
    with pytest.raises(ValueError, match="partition"):
        assert_no_patient_leakage(patient_ids, incomplete)


def test_degenerate_cohorts_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        patient_level_split(np.array([], dtype=int), DATA)
    with pytest.raises(ValueError, match="at least 3 patients"):
        patient_level_split(np.array([0, 0, 1, 1]), DATA)


def test_every_split_is_non_empty_even_for_a_tiny_cohort():
    """Rounding must not produce an empty validation or test split."""
    for n_patients in (3, 4, 5, 10):
        splits = patient_level_split(np.arange(n_patients), DATA)
        assert min(splits.sizes.values()) >= 1, splits.sizes


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def test_normalizer_standardises_the_training_data():
    signals = np.random.default_rng(0).standard_normal((50, 12, 400)) * 3.0 + 1.5
    normalizer = LeadNormalizer.fit(signals)
    transformed = normalizer.transform(signals)
    assert np.allclose(transformed.mean(axis=(0, 2)), 0.0, atol=1e-4)
    assert np.allclose(transformed.std(axis=(0, 2)), 1.0, atol=1e-4)


def test_normalizer_statistics_come_only_from_the_training_split():
    """Fitting on all data would let test-set scale influence training inputs."""
    rng = np.random.default_rng(1)
    signals = rng.standard_normal((60, 12, 200))
    # Make the held-out half wildly different in scale.
    signals[30:] *= 50.0

    train_only = LeadNormalizer.fit(signals[:30])
    everything = LeadNormalizer.fit(signals)

    np.testing.assert_allclose(train_only.mean, signals[:30].mean(axis=(0, 2)))
    np.testing.assert_allclose(train_only.std, signals[:30].std(axis=(0, 2)))
    assert np.all(everything.std > 5 * train_only.std)


def test_normalizer_survives_a_dead_lead():
    """A flat lead has zero variance; dividing by it would produce NaNs."""
    signals = np.random.default_rng(2).standard_normal((20, 12, 300))
    signals[:, 4, :] = 0.0
    normalizer = LeadNormalizer.fit(signals)
    transformed = normalizer.transform(signals)
    assert np.isfinite(transformed).all()
    assert np.allclose(transformed[:, 4, :], 0.0)


def test_per_recording_and_none_modes():
    signals = np.random.default_rng(3).standard_normal((10, 12, 300)) * 4 + 2

    per_recording = LeadNormalizer.fit(signals, mode="per_recording").transform(signals)
    assert np.allclose(per_recording.mean(axis=-1), 0.0, atol=1e-4)
    assert np.allclose(per_recording.std(axis=-1), 1.0, atol=1e-4)

    untouched = LeadNormalizer.fit(signals, mode="none").transform(signals)
    np.testing.assert_allclose(untouched, signals.astype(np.float32))


def test_normalizer_round_trips_through_a_checkpoint():
    """Inference must apply exactly the transformation training used."""
    signals = np.random.default_rng(4).standard_normal((30, 12, 200)) * 2
    original = LeadNormalizer.fit(signals)
    restored = LeadNormalizer.from_dict(original.to_dict())
    np.testing.assert_allclose(original.transform(signals), restored.transform(signals))


def test_normalizer_rejects_a_lead_count_mismatch():
    normalizer = LeadNormalizer.fit(np.zeros((5, 12, 100)) + 1.0)
    with pytest.raises(ValueError, match="12 leads"):
        normalizer.transform(np.zeros((5, 8, 100)))


def test_normalizer_rejects_wrong_dimensionality():
    with pytest.raises(ValueError, match="n_recordings"):
        LeadNormalizer.fit(np.zeros((12, 100)))


# --------------------------------------------------------------------------- #
# Signal conditioning
# --------------------------------------------------------------------------- #
def test_resampling_changes_length_by_the_expected_ratio():
    signals = np.random.default_rng(5).standard_normal((4, 12, 5000))
    assert resample_signals(signals, 500, 100).shape == (4, 12, 1000)
    assert resample_signals(signals, 500, 500).shape == (4, 12, 5000)


def test_resampling_preserves_r_peak_timing():
    """Downsampling to 100 Hz must not move beats, or every interval shifts.

    ``resample_poly`` applies an anti-aliasing filter, which is the reason this
    holds: naive decimation would fold high-frequency content back into the
    signal and distort the QRS complex.
    """
    from ecg_discovery.config import SignalProcessingConfig
    from ecg_discovery.signal_processing.qrs_detection import detect_r_peaks, score_detection

    config = SyntheticConfig(sampling_rate_hz=500)
    cohort = generate_cohort(config, n_recordings=6)
    downsampled = resample_signals(cohort.signals, 500, 100)

    for recording, signal_100 in zip(cohort, downsampled):
        detection = detect_r_peaks(
            signal_100, 100.0, SignalProcessingConfig(), recording.lead_names
        )
        # Ground truth is in 500 Hz samples; convert to the 100 Hz grid.
        reference = np.round(recording.r_peak_samples / 5.0).astype(int)
        score = score_detection(detection.r_peaks, reference, 100.0, 50.0)
        assert score.sensitivity == 1.0
        assert score.max_absolute_error_ms <= 20.0


def test_bandpass_removes_drift_and_keeps_the_beat():
    cohort = generate_cohort(SyntheticConfig(sampling_rate_hz=100), n_recordings=3)
    drifting = cohort.signals + 5.0     # a large constant offset
    filtered = bandpass_signals(drifting, DATA)
    core = slice(50, -50)
    assert np.abs(filtered[:, :, core].mean()) < 0.1
    # The beat survives: filtered signal still tracks the original closely.
    original = bandpass_signals(cohort.signals, DATA)
    for i in range(len(cohort)):
        assert np.corrcoef(
            filtered[i, 1, core], original[i, 1, core]
        )[0, 1] > 0.99
