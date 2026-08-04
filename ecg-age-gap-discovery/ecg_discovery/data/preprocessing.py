"""Signal preprocessing and patient-level dataset splitting.

TWO JOBS, BOTH WITH A LEAKAGE HAZARD
------------------------------------
This module does the unglamorous work between raw recordings and a training
batch, and both halves of it can leak information in ways that inflate measured
performance without producing any visible error.

**Splitting must respect patients, not recordings.** PTB-XL contains patients
who were recorded more than once. Splitting by recording puts the same heart in
both training and test, so the model can score well by recognising a patient it
has already seen rather than by reading the ECG. Nothing crashes; the reported
accuracy is simply wrong, and in the flattering direction.
:func:`patient_level_split` splits by patient and then *verifies its own
output*, because this is exactly the class of bug that survives code review.

**Normalisation statistics must come from training data only.** Computing the
per-lead mean and standard deviation over the whole dataset lets test-set
statistics influence the scale of training inputs. The effect is subtler than a
split leak but the principle is the same, so :class:`LeadNormalizer` is fitted
on the training split and then applied unchanged to validation and test.

WHY NORMALISE PER LEAD AT ALL
-----------------------------
The twelve leads view the heart from different angles and have genuinely
different amplitude scales - a QRS complex is several times taller in V4 than in
aVL. Standardising each lead separately puts them on comparable footing for the
network without erasing the *relative* differences within a lead that carry the
signal. The alternative, ``per_recording``, standardises each recording on its
own; that removes between-patient amplitude information, which is itself
age-related, so it is offered but not the default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import signal as sp_signal

from ecg_discovery.config import DataConfig

__all__ = [
    "SplitIndices",
    "LeadNormalizer",
    "patient_level_split",
    "assert_no_patient_leakage",
    "resample_signals",
    "bandpass_signals",
]


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SplitIndices:
    """Recording indices belonging to each split.

    Indices refer to positions in the original recording arrays, so the same
    object can index signals, ages, sexes and metadata consistently.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @property
    def sizes(self) -> dict[str, int]:
        """Number of recordings in each split."""
        return {
            "train": int(self.train.size),
            "val": int(self.val.size),
            "test": int(self.test.size),
        }

    def as_dict(self) -> dict[str, list[int]]:
        """Split membership as plain lists, for run-directory records."""
        return {
            "train": self.train.tolist(),
            "val": self.val.tolist(),
            "test": self.test.tolist(),
        }


def assert_no_patient_leakage(
    patient_ids: Sequence[int] | np.ndarray, splits: SplitIndices
) -> None:
    """Raise if any patient appears in more than one split.

    Called automatically by :func:`patient_level_split` on its own output. A
    split function that silently leaks is worse than one that fails, because the
    consequence - inflated accuracy - looks like success.

    Raises
    ------
    ValueError
        If a patient's recordings are spread across splits, or if the splits do
        not exactly partition the recordings.
    """
    patient_ids = np.asarray(patient_ids)
    members = {
        name: set(patient_ids[indices].tolist())
        for name, indices in (
            ("train", splits.train), ("val", splits.val), ("test", splits.test)
        )
    }
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = members[first] & members[second]
        if shared:
            raise ValueError(
                f"patient-level leakage: {len(shared)} patient(s) appear in both "
                f"{first} and {second} (for example {sorted(shared)[:5]}). The same "
                "heart in training and test inflates measured accuracy."
            )

    covered = np.concatenate([splits.train, splits.val, splits.test])
    if covered.size != patient_ids.size or set(covered.tolist()) != set(
        range(patient_ids.size)
    ):
        raise ValueError(
            f"splits do not partition the {patient_ids.size} recordings: got "
            f"{covered.size} indices with {len(set(covered.tolist()))} distinct values"
        )


def patient_level_split(
    patient_ids: Sequence[int] | np.ndarray,
    config: DataConfig | None = None,
    seed: int | None = None,
) -> SplitIndices:
    """Partition recordings into train/val/test without splitting any patient.

    Patients - not recordings - are shuffled and dealt into the three splits, so
    every recording from a given patient lands in the same one.

    Parameters
    ----------
    patient_ids:
        Patient identifier for each recording, in recording order.
    config:
        Supplies the split fractions and default seed.
    seed:
        Overrides ``config.split_seed``.

    Returns
    -------
    SplitIndices
        Recording indices per split, verified free of patient leakage.

    Notes
    -----
    The realised *recording* proportions will not exactly equal the requested
    fractions, because patients contribute differing numbers of recordings and
    only whole patients can be moved. The discrepancy is small (a fraction of a
    percent on PTB-XL) and is the unavoidable price of a correct split; the
    alternative - trimming recordings to hit exact proportions - would either
    split a patient or discard data.
    """
    config = config or DataConfig()
    patient_ids = np.asarray(patient_ids)
    if patient_ids.ndim != 1:
        raise ValueError(f"patient_ids must be 1-D, got shape {patient_ids.shape}")
    if patient_ids.size == 0:
        raise ValueError("cannot split an empty dataset")

    unique_patients = np.unique(patient_ids)
    n_patients = unique_patients.size
    if n_patients < 3:
        raise ValueError(
            f"need at least 3 patients to form three splits, got {n_patients}"
        )

    rng = np.random.default_rng(config.split_seed if seed is None else seed)
    shuffled = rng.permutation(unique_patients)

    # Allocate whole patients. Rounding is done on cumulative counts so the
    # three splits always sum to exactly the number of patients.
    n_train = int(round(config.train_frac * n_patients))
    n_val = int(round((config.train_frac + config.val_frac) * n_patients)) - n_train
    # Guarantee every split is non-empty, which rounding can otherwise violate
    # for very small cohorts.
    n_train = max(n_train, 1)
    n_val = max(n_val, 1)
    if n_train + n_val >= n_patients:
        n_train = max(n_patients - 2, 1)
        n_val = 1

    assignment = {
        "train": set(shuffled[:n_train].tolist()),
        "val": set(shuffled[n_train : n_train + n_val].tolist()),
        "test": set(shuffled[n_train + n_val :].tolist()),
    }

    splits = SplitIndices(
        train=np.flatnonzero([pid in assignment["train"] for pid in patient_ids]),
        val=np.flatnonzero([pid in assignment["val"] for pid in patient_ids]),
        test=np.flatnonzero([pid in assignment["test"] for pid in patient_ids]),
    )
    assert_no_patient_leakage(patient_ids, splits)
    return splits


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
@dataclass
class LeadNormalizer:
    """Per-lead standardisation fitted on the training split only.

    Holds one mean and one standard deviation per lead. Fitting on training data
    alone means no property of the test set - not even its overall amplitude -
    can influence the scale of the inputs the model learns from.

    The fitted statistics are saved with the model checkpoint, because applying
    a *different* normalisation at inference than at training would silently
    change every prediction.
    """

    mean: np.ndarray
    std: np.ndarray
    mode: str = "per_lead_train_stats"

    @classmethod
    def fit(
        cls, signals: np.ndarray, mode: str = "per_lead_train_stats"
    ) -> "LeadNormalizer":
        """Compute per-lead statistics from training recordings.

        Parameters
        ----------
        signals:
            ``(n_recordings, n_leads, n_samples)`` training data.
        mode:
            ``per_lead_train_stats``, ``per_recording`` or ``none``. The latter
            two need no fitted statistics and produce an identity-like object.
        """
        signals = np.asarray(signals)
        if signals.ndim != 3:
            raise ValueError(
                f"expected (n_recordings, n_leads, n_samples), got {signals.shape}"
            )
        n_leads = signals.shape[1]

        if mode != "per_lead_train_stats":
            return cls(
                mean=np.zeros(n_leads), std=np.ones(n_leads), mode=mode
            )

        mean = signals.mean(axis=(0, 2))
        std = signals.std(axis=(0, 2))
        # A dead lead has zero variance; dividing by it would produce NaNs that
        # propagate silently through training.
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.astype(np.float64), std=std.astype(np.float64), mode=mode)

    def transform(self, signals: np.ndarray) -> np.ndarray:
        """Apply the normalisation to ``(n_recordings, n_leads, n_samples)`` data."""
        signals = np.asarray(signals, dtype=np.float32)
        if self.mode == "none":
            return signals
        if self.mode == "per_recording":
            mean = signals.mean(axis=-1, keepdims=True)
            std = signals.std(axis=-1, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            return ((signals - mean) / std).astype(np.float32)

        if signals.shape[1] != self.mean.size:
            raise ValueError(
                f"normalizer was fitted for {self.mean.size} leads but received "
                f"{signals.shape[1]}"
            )
        mean = self.mean.astype(np.float32)[None, :, None]
        std = self.std.astype(np.float32)[None, :, None]
        return ((signals - mean) / std).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, stored alongside model weights."""
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LeadNormalizer":
        """Rebuild from :meth:`to_dict`."""
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            mode=payload.get("mode", "per_lead_train_stats"),
        )


# --------------------------------------------------------------------------- #
# Signal conditioning
# --------------------------------------------------------------------------- #
def bandpass_signals(
    signals: np.ndarray, config: DataConfig, sampling_rate_hz: float | None = None
) -> np.ndarray:
    """Apply the configured clinical bandpass (and optional mains notch).

    Zero-phase throughout, so no feature is displaced in time - the same
    requirement that governs R-peak detection, for the same reason.
    """
    signals = np.asarray(signals, dtype=np.float64)
    rate = sampling_rate_hz if sampling_rate_hz is not None else config.sampling_rate_hz
    nyquist = rate / 2.0

    high = min(config.bandpass_high_hz, nyquist * 0.99)
    sos = sp_signal.butter(
        config.bandpass_order,
        [config.bandpass_low_hz / nyquist, high / nyquist],
        btype="bandpass",
        output="sos",
    )
    filtered = sp_signal.sosfiltfilt(sos, signals, axis=-1)

    if config.powerline_notch_hz is not None and config.powerline_notch_hz < nyquist:
        b, a = sp_signal.iirnotch(
            config.powerline_notch_hz / nyquist, config.notch_quality
        )
        filtered = sp_signal.filtfilt(b, a, filtered, axis=-1)

    return np.ascontiguousarray(filtered, dtype=np.float32)


def resample_signals(
    signals: np.ndarray, from_hz: float, to_hz: float
) -> np.ndarray:
    """Resample along the time axis using polyphase filtering.

    Used to derive the model's 100 Hz input from the 500 Hz signal that interval
    measurement requires, so both views describe exactly the same recording.
    ``resample_poly`` applies an anti-aliasing filter as part of the operation,
    which matters when downsampling: without it, content above the new Nyquist
    frequency would fold back into the signal as artefact.
    """
    if from_hz == to_hz:
        return np.asarray(signals, dtype=np.float32)
    ratio = math.gcd(int(round(from_hz)), int(round(to_hz)))
    up = int(round(to_hz)) // ratio
    down = int(round(from_hz)) // ratio
    resampled = sp_signal.resample_poly(
        np.asarray(signals, dtype=np.float64), up, down, axis=-1
    )
    return np.ascontiguousarray(resampled, dtype=np.float32)
