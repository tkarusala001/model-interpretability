"""Tests for the training loop.

The training loop's real product is not the model but the table of per-recording
**age gaps** that Phases 6 to 9 analyse. So alongside the usual checks that
training converges, these tests concentrate on the things that would silently
corrupt that table: predictions landing in the wrong rows, the wrong checkpoint
being evaluated, or normalisation drifting between training and inference.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

from ecg_discovery.config import (
    BackboneConfig,
    DataConfig,
    SyntheticConfig,
    TrainingConfig,
)
from ecg_discovery.data.preprocessing import resample_signals
from ecg_discovery.data.synthetic_ecg import generate_cohort
from ecg_discovery.runtime import RunContext
from ecg_discovery.training.train import _regression_metrics, train_age_regressor

# A small, fast configuration. Enough signal to learn from, small enough to run
# repeatedly in a test suite.
FAST_TRAINING = TrainingConfig(
    epochs=12,
    batch_size=32,
    learning_rate=3e-3,
    warmup_epochs=1,
    early_stopping_patience=20,
    scatter_every_n_epochs=5,
    seed=0,
)
SMALL_BACKBONE = BackboneConfig(
    stem_channels=16, stage_channels=(16, 24, 32), stride_per_stage=(2, 2, 2)
)


def _cohort(n_recordings: int = 320):
    config = SyntheticConfig(sampling_rate_hz=100)
    cohort = generate_cohort(config, n_recordings=n_recordings)
    return cohort


def _train(cohort, training_config=FAST_TRAINING, run=None, data_config=None):
    return train_age_regressor(
        signals=cohort.signals,
        ages=cohort.ages,
        sexes=cohort.sexes,
        patient_ids=cohort.patient_ids,
        record_ids=[r.record_id for r in cohort],
        data_config=data_config or DataConfig(),
        backbone_config=SMALL_BACKBONE,
        training_config=training_config,
        run=run,
    )


# --------------------------------------------------------------------------- #
# Does it learn?
# --------------------------------------------------------------------------- #
def test_training_beats_the_constant_prediction_baseline():
    """The bar is beating the mean-age guess, not merely reducing the loss.

    A loss can fall simply by converging on the mean, which would pass a
    "loss went down" test while learning nothing about ECGs.
    """
    cohort = _cohort()
    result = _train(cohort)

    test_prediction = result.predictions["test"]
    baseline_mae = float(
        np.mean(np.abs(test_prediction.true_age - cohort.ages[result.splits.train].mean()))
    )
    assert result.test_metrics["mae"] < 0.8 * baseline_mae
    assert result.test_metrics["r2"] > 0.2
    assert test_prediction.predicted_age.std() > 2.0


def test_training_loss_decreases():
    result = _train(_cohort(240))
    losses = [record["train_loss"] for record in result.history]
    assert losses[-1] < losses[0]


def test_history_records_one_entry_per_epoch():
    result = _train(_cohort(160), dataclasses.replace(FAST_TRAINING, epochs=6))
    assert len(result.history) == 6
    for record in result.history:
        assert {"epoch", "train_loss", "val_loss", "val_mae", "learning_rate"} <= set(record)


# --------------------------------------------------------------------------- #
# The age-gap table must be trustworthy
# --------------------------------------------------------------------------- #
def test_predictions_are_aligned_with_the_original_recordings():
    """Every prediction must sit against the right recording.

    The training loader shuffles, so predictions come back out of order. If they
    were reassembled by assumption rather than by carried index, the age-gap
    table would be quietly scrambled and every downstream analysis would be
    studying noise - with nothing raising an error.
    """
    cohort = _cohort(200)
    result = _train(cohort, dataclasses.replace(FAST_TRAINING, epochs=3))

    all_indices = []
    for split_name, prediction in result.predictions.items():
        # True ages must match the source arrays at the carried indices.
        np.testing.assert_allclose(
            prediction.true_age, cohort.ages[prediction.indices], rtol=1e-5
        )
        # And so must the record identifiers.
        expected_ids = [cohort[i].record_id for i in prediction.indices]
        assert prediction.record_ids == expected_ids
        # Indices must belong to the split they claim.
        np.testing.assert_array_equal(
            np.sort(prediction.indices), np.sort(getattr(result.splits, split_name))
        )
        all_indices.append(prediction.indices)

    covered = np.concatenate(all_indices)
    assert sorted(covered.tolist()) == list(range(len(cohort)))


def test_age_gap_is_predicted_minus_true():
    result = _train(_cohort(160), dataclasses.replace(FAST_TRAINING, epochs=3))
    prediction = result.predictions["test"]
    np.testing.assert_allclose(
        prediction.age_gap, prediction.predicted_age - prediction.true_age
    )


def test_test_split_predictions_come_from_unseen_patients():
    """Guard the whole point of the split, at the level of the output table."""
    cohort = _cohort(240)
    result = _train(cohort, dataclasses.replace(FAST_TRAINING, epochs=2))
    train_patients = set(cohort.patient_ids[result.splits.train].tolist())
    test_patients = set(cohort.patient_ids[result.predictions["test"].indices].tolist())
    assert train_patients & test_patients == set()


# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #
def test_best_checkpoint_is_selected_not_the_last():
    """Evaluation must use the best validation epoch, not wherever training ended."""
    result = _train(_cohort(200), dataclasses.replace(FAST_TRAINING, epochs=10))
    val_maes = [record["val_mae"] for record in result.history]
    assert result.best_val_mae == pytest.approx(min(val_maes))
    assert result.best_epoch == int(np.argmin(val_maes))
    # The returned model really is that checkpoint: re-evaluating validation
    # reproduces the best score.
    assert result.predictions["val"].metrics()["mae"] == pytest.approx(
        result.best_val_mae, rel=1e-4
    )


def test_early_stopping_halts_a_stalled_run():
    stalling = dataclasses.replace(
        FAST_TRAINING, epochs=40, early_stopping_patience=2, learning_rate=1e-8
    )
    result = _train(_cohort(160), stalling)
    assert len(result.history) < 40


def test_training_is_reproducible():
    cohort = _cohort(160)
    config = dataclasses.replace(FAST_TRAINING, epochs=4)
    first = _train(cohort, config)
    second = _train(cohort, config)
    np.testing.assert_allclose(
        first.predictions["test"].predicted_age,
        second.predictions["test"].predicted_age,
        rtol=1e-5, atol=1e-5,
    )


def test_different_seed_gives_a_different_model():
    cohort = _cohort(160)
    first = _train(cohort, dataclasses.replace(FAST_TRAINING, epochs=4, seed=0))
    second = _train(cohort, dataclasses.replace(FAST_TRAINING, epochs=4, seed=7))
    assert not np.allclose(
        first.predictions["test"].predicted_age,
        second.predictions["test"].predicted_age,
    )


# --------------------------------------------------------------------------- #
# Normalisation consistency
# --------------------------------------------------------------------------- #
def test_normalizer_is_fitted_on_training_data_only():
    cohort = _cohort(200)
    result = _train(cohort, dataclasses.replace(FAST_TRAINING, epochs=2))
    expected_mean = cohort.signals[result.splits.train].mean(axis=(0, 2))
    np.testing.assert_allclose(result.normalizer.mean, expected_mean, rtol=1e-4)


def test_checkpoint_stores_the_normalizer():
    """Applying a different normalisation at inference would change every prediction."""
    cohort = _cohort(160)
    with RunContext("test_train", _tmp_runs(), seed=0) as run:
        result = _train(cohort, dataclasses.replace(FAST_TRAINING, epochs=2), run=run)

    payload = torch.load(result.checkpoint_path, weights_only=False)
    np.testing.assert_allclose(
        payload["normalizer"]["mean"], result.normalizer.mean, rtol=1e-6
    )
    assert payload["best_epoch"] == result.best_epoch


# --------------------------------------------------------------------------- #
# Run artifacts
# --------------------------------------------------------------------------- #
_TMP_RUNS: list = []


def _tmp_runs():
    import tempfile

    directory = tempfile.mkdtemp(prefix="ecg_runs_")
    _TMP_RUNS.append(directory)
    return directory


def test_run_directory_contains_everything_needed_to_reproduce():
    cohort = _cohort(160)
    with RunContext("test_train", _tmp_runs(), seed=0) as run:
        result = _train(cohort, dataclasses.replace(FAST_TRAINING, epochs=6), run=run)

    assert (run.dir / "config.json").exists()
    assert (run.dir / "metrics.jsonl").exists()
    for name in ("model.pt", "predictions.json", "splits.json", "final_metrics.json"):
        assert (run.artifacts_dir / name).exists(), name

    metrics = json.loads((run.artifacts_dir / "final_metrics.json").read_text())
    assert metrics["test_mae"] == pytest.approx(result.test_metrics["mae"])
    assert metrics["split_sizes"]["test"] == result.splits.test.size

    saved = json.loads((run.artifacts_dir / "predictions.json").read_text())
    np.testing.assert_allclose(
        saved["test"]["age_gap"], result.predictions["test"].age_gap, rtol=1e-5
    )

    epochs = [json.loads(line)["epoch"] for line in
              (run.dir / "metrics.jsonl").read_text().strip().split("\n")]
    assert epochs == list(range(6))


def test_saved_splits_match_the_split_actually_used():
    cohort = _cohort(160)
    with RunContext("test_train", _tmp_runs(), seed=0) as run:
        result = _train(cohort, dataclasses.replace(FAST_TRAINING, epochs=2), run=run)
    saved = json.loads((run.artifacts_dir / "splits.json").read_text())
    assert saved["test"] == result.splits.test.tolist()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_regression_metrics_on_hand_checked_values():
    true = np.array([10.0, 20.0, 30.0, 40.0])
    predicted = np.array([12.0, 18.0, 33.0, 37.0])
    metrics = _regression_metrics(true, predicted)
    assert metrics["mae"] == pytest.approx(2.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9 + 9) / 4))
    assert metrics["r2"] == pytest.approx(1 - (26 / 4) / np.var(true))


def test_r2_is_zero_for_a_mean_predictor():
    """R-squared of 0 must mean 'no better than guessing the mean age'."""
    true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    metrics = _regression_metrics(true, np.full_like(true, true.mean()))
    assert metrics["r2"] == pytest.approx(0.0)


def test_metrics_handle_an_empty_split():
    metrics = _regression_metrics(np.array([]), np.array([]))
    assert all(np.isnan(value) for value in metrics.values())


# --------------------------------------------------------------------------- #
# Resampled input
# --------------------------------------------------------------------------- #
def test_training_works_on_resampled_500hz_data():
    """The real pipeline generates at 500 Hz and trains on the 100 Hz view."""
    cohort = generate_cohort(SyntheticConfig(sampling_rate_hz=500), n_recordings=160)
    signals = resample_signals(cohort.signals, 500, 100)
    assert signals.shape[-1] == 1000

    result = train_age_regressor(
        signals=signals,
        ages=cohort.ages,
        sexes=cohort.sexes,
        patient_ids=cohort.patient_ids,
        backbone_config=SMALL_BACKBONE,
        training_config=dataclasses.replace(FAST_TRAINING, epochs=4),
    )
    assert np.isfinite(result.test_metrics["mae"])
