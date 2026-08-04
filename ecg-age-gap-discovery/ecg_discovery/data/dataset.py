"""An in-memory PyTorch dataset shared by the synthetic and PTB-XL paths.

This is a small addition to the repository layout given in the build plan. It
exists so that the training loop is written once against a single interface:
the synthetic cohort and PTB-XL differ in how they are *loaded*, not in what a
batch looks like, and keeping that difference out of the trainer is what lets
the entire pipeline be validated on synthetic data before real data arrives.

Recordings are held in memory as float32. PTB-XL at 100 Hz is about 1 GB in that
form, which is comfortable; the 500 Hz version is only ever needed for interval
measurement, which streams one recording at a time and does not use this class.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["ECGArrayDataset"]


class ECGArrayDataset(Dataset):
    """Waveforms with their age labels, sex covariate and provenance.

    Parameters
    ----------
    signals:
        ``(n_recordings, n_leads, n_samples)`` float array, already preprocessed
        and normalised.
    ages:
        Chronological age in years - the regression target.
    sexes:
        Sex codes (0 = male, 1 = female).
    patient_ids:
        Patient identifier per recording. Carried through so that a split can be
        audited for leakage after the fact, not merely trusted at creation.
    record_ids:
        Optional string identifiers, used to join predictions back to interval
        features in Phase 8.

    Notes
    -----
    ``__getitem__`` returns ``(waveform, age, sex, index)``. The index is
    returned so predictions can be written back into the right row of the
    original arrays regardless of shuffling - reconstructing that order by
    assumption is a classic source of silently misaligned results.
    """

    def __init__(
        self,
        signals: np.ndarray,
        ages: np.ndarray,
        sexes: np.ndarray,
        patient_ids: np.ndarray | None = None,
        record_ids: Sequence[str] | None = None,
    ) -> None:
        signals = np.asarray(signals, dtype=np.float32)
        if signals.ndim != 3:
            raise ValueError(
                f"expected (n_recordings, n_leads, n_samples), got {signals.shape}"
            )
        n = signals.shape[0]
        for name, array in (("ages", ages), ("sexes", sexes)):
            if len(array) != n:
                raise ValueError(
                    f"{name} has {len(array)} entries but there are {n} recordings"
                )
        if record_ids is not None and len(record_ids) != n:
            raise ValueError(
                f"record_ids has {len(record_ids)} entries but there are {n} recordings"
            )

        self.signals = signals
        self.ages = np.asarray(ages, dtype=np.float32)
        self.sexes = np.asarray(sexes, dtype=np.float32)
        self.patient_ids = (
            None if patient_ids is None else np.asarray(patient_ids)
        )
        self.record_ids = None if record_ids is None else list(record_ids)

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.signals[index]),
            torch.tensor(self.ages[index]),
            torch.tensor(self.sexes[index]),
            index,
        )

    def subset(self, indices: np.ndarray) -> "ECGArrayDataset":
        """A new dataset holding only the given recordings.

        Used to materialise a split. Views would be cheaper, but a split must
        also be *normalised* using training-only statistics, so the data is
        copied at this boundary anyway.
        """
        indices = np.asarray(indices, dtype=np.int64)
        return ECGArrayDataset(
            signals=self.signals[indices],
            ages=self.ages[indices],
            sexes=self.sexes[indices],
            patient_ids=None if self.patient_ids is None else self.patient_ids[indices],
            record_ids=(
                None if self.record_ids is None
                else [self.record_ids[i] for i in indices]
            ),
        )
