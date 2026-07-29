"""Training loop for the from-scratch ECG age regressor.

WHAT THIS PRODUCES, AND WHAT IT IS FOR
--------------------------------------
The immediate output is a model that predicts age from an ECG, reported as a
test-set mean absolute error. That number is a sanity check, not a contribution:
predicting ECG age is an established task and this model is not competing with
published work.

The output that actually matters downstream is the table of **per-recording age
gaps** - predicted minus true age - written to the run directory for every
split. Phases 6 to 9 study those residuals. This is why predictions are always
written with their record identifiers attached and in the original recording
order, rather than in whatever order the data loader happened to produce.

DESIGN NOTES
------------
*Patient-level splitting* is delegated to
:func:`~ecg_discovery.data.preprocessing.patient_level_split`, which verifies
its own output for leakage. See that module for why this is the most dangerous
step in the pipeline.

*Normalisation is fitted on the training split only* and saved with the
checkpoint, so inference applies exactly the transformation training used.

*Model selection uses validation MAE*, and the best checkpoint - not the last -
is what gets evaluated on test. Early stopping ends a run that has stopped
improving. Both matter more than usual here: an overfitted model's age gap is
dominated by memorisation noise, which would make every downstream analysis a
study of nothing.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ecg_discovery.config import BackboneConfig, DataConfig, TrainingConfig
from ecg_discovery.data.dataset import ECGArrayDataset
from ecg_discovery.data.preprocessing import (
    LeadNormalizer,
    SplitIndices,
    assert_no_patient_leakage,
    patient_level_split,
)
from ecg_discovery.models.ecg_age_regressor import ECGAgeRegressor
from ecg_discovery.runtime import RunContext, resolve_device, set_global_seed

__all__ = ["TrainingResult", "SplitPredictions", "train_age_regressor", "evaluate"]


@dataclass(frozen=True)
class SplitPredictions:
    """Per-recording predictions for one split, in original recording order.

    Attributes
    ----------
    indices:
        Positions in the original recording arrays, so predictions can be joined
        to interval features and metadata without assuming an ordering.
    age_gap:
        ``predicted - true``, in years. This is the quantity Phases 6 to 9 study.
    """

    split: str
    indices: np.ndarray
    record_ids: list[str] | None
    true_age: np.ndarray
    predicted_age: np.ndarray

    @property
    def age_gap(self) -> np.ndarray:
        """Predicted minus true age, in years."""
        return self.predicted_age - self.true_age

    def metrics(self) -> dict[str, float]:
        """Mean absolute error, root mean squared error and R-squared."""
        return _regression_metrics(self.true_age, self.predicted_age)

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form written to the run directory."""
        payload: dict[str, Any] = {
            "split": self.split,
            "index": self.indices.tolist(),
            "true_age": self.true_age.tolist(),
            "predicted_age": self.predicted_age.tolist(),
            "age_gap": self.age_gap.tolist(),
        }
        if self.record_ids is not None:
            payload["record_id"] = self.record_ids
        return payload


@dataclass
class TrainingResult:
    """Everything a completed training run produced."""

    model: ECGAgeRegressor
    normalizer: LeadNormalizer
    splits: SplitIndices
    predictions: dict[str, SplitPredictions]
    best_epoch: int
    best_val_mae: float
    history: list[dict[str, float]] = field(default_factory=list)
    checkpoint_path: Path | None = None

    @property
    def test_metrics(self) -> dict[str, float]:
        """Held-out performance of the selected checkpoint."""
        return self.predictions["test"].metrics()


def _regression_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and R-squared for age prediction.

    R-squared is reported against the variance of the true ages, so 0 means "no
    better than always guessing the mean age" - the baseline a model that has
    learned nothing converges to.
    """
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    error = predicted - true
    variance = float(np.var(true))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1.0 - np.mean(error ** 2) / variance) if variance > 0 else float("nan"),
    }


def _build_loss(config: TrainingConfig) -> nn.Module:
    """Loss function selected by config.

    Huber is the default: it behaves like squared error for small residuals but
    grows linearly for large ones, so a handful of recordings with implausible
    age labels - which real clinical datasets contain - cannot dominate the
    gradient.
    """
    if config.loss == "huber":
        return nn.SmoothL1Loss(beta=config.huber_delta)
    if config.loss == "mse":
        return nn.MSELoss()
    return nn.L1Loss()


def _build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    return torch.optim.SGD(
        model.parameters(), lr=config.learning_rate,
        momentum=0.9, weight_decay=config.weight_decay,
    )


def _learning_rate_at(epoch: int, config: TrainingConfig) -> float:
    """Learning-rate schedule: linear warm-up, then cosine decay.

    Warm-up matters for a randomly initialised network with batch normalisation:
    the first few batches produce large, poorly-scaled gradients, and stepping
    at full learning rate through them can leave the model in a bad region it
    never recovers from.
    """
    if config.lr_scheduler == "none":
        return config.learning_rate
    if epoch < config.warmup_epochs:
        return config.learning_rate * (epoch + 1) / max(config.warmup_epochs, 1)
    progress = (epoch - config.warmup_epochs) / max(
        config.epochs - config.warmup_epochs, 1
    )
    return 0.5 * config.learning_rate * (1.0 + np.cos(np.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate(
    model: ECGAgeRegressor,
    loader: DataLoader,
    device: str,
    loss_fn: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run the model over a loader without updating it.

    Returns
    -------
    tuple
        ``(indices, true_ages, predicted_ages, mean_loss)``, with rows in the
        loader's iteration order; callers re-sort by ``indices``.
    """
    model.eval()
    indices: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    total_loss = 0.0
    total_count = 0

    for waveform, age, sex, index in loader:
        waveform = waveform.to(device)
        age = age.to(device)
        sex = sex.to(device)
        predicted = model(waveform, sex)
        if loss_fn is not None:
            total_loss += float(loss_fn(predicted, age).detach()) * age.shape[0]
        total_count += age.shape[0]
        indices.append(np.asarray(index))
        trues.append(age.detach().cpu().numpy())
        predictions.append(predicted.detach().cpu().numpy())

    if total_count == 0:
        empty = np.empty(0)
        return empty, empty, empty, float("nan")

    return (
        np.concatenate(indices),
        np.concatenate(trues),
        np.concatenate(predictions),
        total_loss / total_count if loss_fn is not None else float("nan"),
    )


def _predict_split(
    model: ECGAgeRegressor,
    dataset: ECGArrayDataset,
    split_name: str,
    split_indices: np.ndarray,
    config: TrainingConfig,
    device: str,
) -> SplitPredictions:
    """Predict a whole split and restore the original recording order."""
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers,
    )
    local_index, true_age, predicted_age, _ = evaluate(model, loader, device)
    order = np.argsort(local_index)
    local_index = local_index[order]
    return SplitPredictions(
        split=split_name,
        indices=split_indices[local_index],
        record_ids=(
            None if dataset.record_ids is None
            else [dataset.record_ids[i] for i in local_index]
        ),
        true_age=true_age[order],
        predicted_age=predicted_age[order],
    )


def train_age_regressor(
    signals: np.ndarray,
    ages: np.ndarray,
    sexes: np.ndarray,
    patient_ids: np.ndarray,
    record_ids: Sequence[str] | None = None,
    data_config: DataConfig | None = None,
    backbone_config: BackboneConfig | None = None,
    training_config: TrainingConfig | None = None,
    run: RunContext | None = None,
    progress: bool = False,
    splits: SplitIndices | None = None,
) -> TrainingResult:
    """Train the age regressor and return the model with per-split age gaps.

    Parameters
    ----------
    signals:
        ``(n_recordings, n_leads, n_samples)`` preprocessed waveforms.
    ages, sexes, patient_ids:
        Per-recording labels and provenance. ``patient_ids`` drives the split.
    record_ids:
        Optional identifiers, carried into the prediction tables so Phase 8 can
        join age gaps to interval features.
    run:
        Run directory for metrics and artifacts. Logging is skipped if omitted,
        which is what tests do.
    progress:
        Print a per-epoch line to stdout.

    Returns
    -------
    TrainingResult
        The selected model, its normaliser, the splits used, and per-recording
        predictions for train, validation and test.
    """
    data_config = data_config or DataConfig()
    backbone_config = backbone_config or BackboneConfig()
    training_config = training_config or TrainingConfig()

    set_global_seed(training_config.seed)
    device = resolve_device(training_config.device)

    # -- Split by patient, never by recording --------------------------------
    # A caller may supply a split instead - PTB-XL ships its own stratified
    # folds, and using them keeps results comparable with published work. Any
    # supplied split is still verified for patient leakage rather than trusted.
    if splits is None:
        splits = patient_level_split(patient_ids, data_config)
    else:
        assert_no_patient_leakage(patient_ids, splits)

    # -- Normalise using training statistics only ----------------------------
    normalizer = LeadNormalizer.fit(
        np.asarray(signals)[splits.train], mode=data_config.normalization
    )
    normalised = normalizer.transform(signals)

    full = ECGArrayDataset(normalised, ages, sexes, patient_ids, record_ids)
    subsets = {
        name: full.subset(indices)
        for name, indices in (
            ("train", splits.train), ("val", splits.val), ("test", splits.test)
        )
    }

    generator = torch.Generator()
    generator.manual_seed(training_config.seed)
    train_loader = DataLoader(
        subsets["train"], batch_size=training_config.batch_size, shuffle=True,
        num_workers=training_config.num_workers, generator=generator, drop_last=False,
    )
    val_loader = DataLoader(
        subsets["val"], batch_size=training_config.batch_size, shuffle=False,
        num_workers=training_config.num_workers,
    )

    model = ECGAgeRegressor(backbone_config).to(device)
    loss_fn = _build_loss(training_config)
    optimizer = _build_optimizer(model, training_config)

    best_val_mae = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(training_config.epochs):
        learning_rate = _learning_rate_at(epoch, training_config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_count = 0

        for waveform, age, sex, _ in train_loader:
            waveform = waveform.to(device)
            age = age.to(device)
            sex = sex.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(waveform, sex), age)
            loss.backward()
            if training_config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(), training_config.grad_clip_norm
                )
            optimizer.step()

            running_loss += float(loss.detach()) * age.shape[0]
            running_count += age.shape[0]

        _, val_true, val_predicted, val_loss = evaluate(
            model, val_loader, device, loss_fn
        )
        val_metrics = _regression_metrics(val_true, val_predicted)
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": running_loss / max(running_count, 1),
            "val_loss": val_loss,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "seconds": time.time() - epoch_start,
        }
        history.append(record)
        if run is not None:
            run.log(record)
        if progress:
            print(
                f"epoch {epoch:3d}  train {record['train_loss']:7.3f}  "
                f"val_mae {record['val_mae']:6.2f}  lr {learning_rate:.2e}"
            )

        # Periodic predicted-vs-true dump for inspecting calibration over time.
        if (
            run is not None
            and (epoch + 1) % training_config.scatter_every_n_epochs == 0
        ):
            run.save_json(
                f"val_scatter_epoch_{epoch:03d}.json",
                {"true_age": val_true.tolist(), "predicted_age": val_predicted.tolist()},
            )

        # Model selection on validation MAE; the best checkpoint, not the last,
        # is what gets evaluated on test.
        if val_metrics["mae"] < best_val_mae - 1e-6:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.early_stopping_patience:
                if run is not None:
                    run.log({"epoch": epoch, "event": "early_stop"})
                break

    model.load_state_dict(best_state)

    predictions = {
        name: _predict_split(
            model, subsets[name], name,
            getattr(splits, name), training_config, device,
        )
        for name in ("train", "val", "test")
    }

    checkpoint_path: Path | None = None
    if run is not None:
        checkpoint_path = run.artifact_path("model.pt")
        torch.save(
            {
                "state_dict": best_state,
                "backbone_config": backbone_config.__dict__,
                "normalizer": normalizer.to_dict(),
                "best_epoch": best_epoch,
                "best_val_mae": best_val_mae,
            },
            checkpoint_path,
        )
        run.save_json("splits.json", splits.as_dict())
        run.save_json(
            "predictions.json",
            {name: prediction.as_dict() for name, prediction in predictions.items()},
        )
        run.save_json(
            "final_metrics.json",
            {
                "best_epoch": best_epoch,
                "best_val_mae": best_val_mae,
                "split_sizes": splits.sizes,
                **{
                    f"{name}_{key}": value
                    for name, prediction in predictions.items()
                    for key, value in prediction.metrics().items()
                },
            },
        )

    return TrainingResult(
        model=model,
        normalizer=normalizer,
        splits=splits,
        predictions=predictions,
        best_epoch=best_epoch,
        best_val_mae=best_val_mae,
        history=history,
        checkpoint_path=checkpoint_path,
    )
