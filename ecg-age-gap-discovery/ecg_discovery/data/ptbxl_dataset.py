"""Loading PTB-XL waveforms, demographics and diagnostic labels.

**Requires PTB-XL downloaded per ``scripts/download_data.sh``.** Every function
here raises a clear error if it is absent rather than failing obscurely partway
through.

WHAT PTB-XL PROVIDES
--------------------
21,799 clinical 12-lead ECGs of 10 seconds each, from 18,869 patients, with
age, sex, and cardiologist-assigned diagnostic statements in the SCP-ECG
standard. Waveforms ship at both 100 Hz and 500 Hz, which suits this project
exactly: the model trains on the 100 Hz view while classical intervals are
measured on the 500 Hz view of the same recordings.

THREE THINGS THAT WILL SILENTLY CORRUPT RESULTS IF MISHANDLED
-------------------------------------------------------------
**Ages above 89 are recorded as 300.** This is a HIPAA-style anonymisation, not
a data error, and 300 is a plausible-looking float that will pass straight
through any numeric pipeline. Left in, it would drag the regression target and
every age-gap residual with it. Those recordings are *dropped*, not clipped:
inventing an age for them would silently fabricate the very quantity this
project studies.

**Patients recur.** 21,799 recordings come from 18,869 patients, so splitting by
recording puts the same heart in train and test. The dataset's own
``strat_fold`` column is used by default, and
:func:`official_split` verifies that no patient crosses a fold boundary rather
than trusting that it does not.

**The dataset includes children.** Ages run from 0. An age regressor trained
across paediatric and adult ECGs is modelling growth as much as ageing, which is
a different phenomenon. ``DataConfig.min_age_years`` restricts the cohort to
adults by default.

DIAGNOSTIC SUPERCLASSES
-----------------------
Each recording carries SCP-ECG statements with likelihood values. Following the
convention established with the dataset, statements marked as diagnostic in
``scp_statements.csv`` are aggregated into five superclasses - NORM (normal), MI
(myocardial infarction), STTC (ST/T change), CD (conduction disturbance) and HYP
(hypertrophy). Labels are **multi-label**: a recording may carry several, or
none at all when it has no diagnostic statement.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from ecg_discovery.config import DataConfig
from ecg_discovery.data.preprocessing import SplitIndices, assert_no_patient_leakage

__all__ = [
    "DIAGNOSTIC_SUPERCLASSES",
    "AGE_SENTINEL",
    "PTBXLData",
    "load_ptbxl_metadata",
    "load_ptbxl",
    "iter_ptbxl_waveforms",
    "load_waveform_subset",
    "official_split",
    "summarise_ptbxl",
]

#: The five diagnostic superclasses, in a fixed order used throughout.
DIAGNOSTIC_SUPERCLASSES: tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")

#: PTB-XL records every age above 89 as this value, for anonymisation.
AGE_SENTINEL = 300.0


@dataclass
class PTBXLData:
    """Loaded PTB-XL cohort, in the same form the training pipeline expects.

    Attributes
    ----------
    signals:
        ``(n_recordings, 12, n_samples)`` float32 waveforms in millivolts.
    ages, sexes, patient_ids, record_ids:
        Per-recording labels and provenance. ``sexes`` uses PTB-XL's own coding
        (0 = male, 1 = female).
    diagnostic_labels:
        ``(n_recordings, 5)`` multi-hot over :data:`DIAGNOSTIC_SUPERCLASSES`.
        A row may be all zeros: not every recording has a diagnostic statement.
    strat_folds:
        The dataset's own stratified fold assignment (1-10), used for the
        recommended split.
    metadata:
        The full filtered metadata table, for anything else.
    """

    signals: np.ndarray
    ages: np.ndarray
    sexes: np.ndarray
    patient_ids: np.ndarray
    record_ids: list[str]
    diagnostic_labels: np.ndarray
    strat_folds: np.ndarray
    sampling_rate_hz: int
    metadata: pd.DataFrame

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    @property
    def n_patients(self) -> int:
        """Distinct patients in the cohort."""
        return int(np.unique(self.patient_ids).size)


def _require_dataset(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(
            f"PTB-XL directory not found: {root}\n"
            "Download it first:  bash scripts/download_data.sh"
        )
    for name in ("ptbxl_database.csv", "scp_statements.csv"):
        if not (root / name).is_file():
            raise FileNotFoundError(
                f"{root / name} is missing - the download looks incomplete.\n"
                "Re-run:  bash scripts/download_data.sh"
            )


def _diagnostic_superclasses(root: Path) -> dict[str, str]:
    """Map each diagnostic SCP code to its superclass."""
    statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
    diagnostic = statements[statements.diagnostic == 1]
    return diagnostic["diagnostic_class"].dropna().to_dict()


def load_ptbxl_metadata(
    root: str | Path, config: DataConfig | None = None
) -> pd.DataFrame:
    """Load and filter PTB-XL's metadata table, without touching the waveforms.

    Applies the age filtering described in the module docstring and attaches
    multi-hot diagnostic superclass columns.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``ecg_id``, with an added ``dx_<CLASS>`` column per
        superclass. Rows excluded by age filtering are already removed.
    """
    config = config or DataConfig()
    root = Path(root)
    _require_dataset(root)

    frame = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    n_total = len(frame)

    # Age handling. The sentinel is removed first and explicitly, because 300 is
    # a numerically plausible value that would otherwise survive any range check
    # phrased as a maximum.
    if config.drop_age_sentinel_300:
        frame = frame[frame.age != AGE_SENTINEL]
    frame = frame[frame.age.between(config.min_age_years, config.max_age_years)]
    frame = frame[frame.sex.isin([0, 1])]

    if frame.empty:
        raise ValueError(
            f"no recordings survived filtering (age {config.min_age_years}-"
            f"{config.max_age_years}) out of {n_total} in the dataset"
        )

    superclass_of = _diagnostic_superclasses(root)

    def superclasses(raw: str) -> set[str]:
        try:
            codes = ast.literal_eval(raw) if isinstance(raw, str) else {}
        except (ValueError, SyntaxError):
            return set()
        return {superclass_of[code] for code in codes if code in superclass_of}

    assigned = frame.scp_codes.apply(superclasses)
    for name in DIAGNOSTIC_SUPERCLASSES:
        frame[f"dx_{name}"] = assigned.apply(lambda s, n=name: int(n in s))

    return frame


def iter_ptbxl_waveforms(
    root: str | Path,
    metadata: pd.DataFrame,
    sampling_rate_hz: int = 100,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(record_id, signal)`` one recording at a time.

    Streaming rather than loading everything at once, because the 500 Hz set is
    roughly 5 GB in memory as float32 while interval measurement only ever needs
    one recording at a time.
    """
    try:
        import wfdb
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise ImportError(
            "reading PTB-XL requires the `wfdb` package: pip install wfdb"
        ) from error

    root = Path(root)
    column = "filename_lr" if sampling_rate_hz == 100 else "filename_hr"
    for ecg_id, relative in metadata[column].items():
        record = wfdb.rdrecord(str(root / relative))
        # wfdb returns (n_samples, n_leads); the project uses (n_leads, n_samples).
        yield str(ecg_id), np.asarray(record.p_signal, dtype=np.float32).T


def load_ptbxl(
    root: str | Path,
    config: DataConfig | None = None,
    sampling_rate_hz: int | None = None,
    limit: int | None = None,
    progress: bool = False,
) -> PTBXLData:
    """Load PTB-XL waveforms and labels into memory.

    Parameters
    ----------
    root:
        Directory containing ``ptbxl_database.csv`` and the record trees.
    config:
        Supplies age filtering and the default sampling rate.
    sampling_rate_hz:
        Overrides ``config.sampling_rate_hz``. Use 100 for training; 500 is
        better streamed via :func:`iter_ptbxl_waveforms` than loaded whole.
    limit:
        Load only the first N recordings after filtering. For smoke tests.
    progress:
        Print a progress line every 2,000 recordings.

    Returns
    -------
    PTBXLData
        Waveforms, labels and metadata, aligned by row.
    """
    config = config or DataConfig()
    rate = sampling_rate_hz or config.sampling_rate_hz
    root = Path(root)

    metadata = load_ptbxl_metadata(root, config)
    if limit is not None:
        metadata = metadata.iloc[:limit]

    signals: list[np.ndarray] = []
    record_ids: list[str] = []
    for position, (record_id, signal) in enumerate(
        iter_ptbxl_waveforms(root, metadata, rate)
    ):
        signals.append(signal)
        record_ids.append(record_id)
        if progress and position and position % 2000 == 0:
            print(f"  loaded {position}/{len(metadata)} recordings")

    stacked = np.stack(signals).astype(np.float32)
    # PTB-XL contains a small number of recordings with NaN samples where a lead
    # dropped out. Zeroing is deliberate: after per-lead standardisation zero is
    # the training mean, so a dropped lead reads as "no information" rather than
    # poisoning the batch with NaN.
    n_nan = int(np.isnan(stacked).sum())
    if n_nan:
        stacked = np.nan_to_num(stacked, nan=0.0)

    return PTBXLData(
        signals=stacked,
        ages=metadata.age.to_numpy(dtype=np.float64),
        sexes=metadata.sex.to_numpy(dtype=np.int64),
        patient_ids=metadata.patient_id.to_numpy(dtype=np.int64),
        record_ids=record_ids,
        diagnostic_labels=metadata[
            [f"dx_{name}" for name in DIAGNOSTIC_SUPERCLASSES]
        ].to_numpy(dtype=np.int64),
        strat_folds=metadata.strat_fold.to_numpy(dtype=np.int64),
        sampling_rate_hz=rate,
        metadata=metadata,
    )


def load_waveform_subset(
    root: str | Path,
    metadata: pd.DataFrame,
    positions: Sequence[int] | np.ndarray,
    sampling_rate_hz: int,
) -> np.ndarray:
    """Load waveforms for selected rows only, at the requested sampling rate.

    Exists because the whole dataset at 500 Hz is about 5 GB as float32, while
    the interval measurements that need 500 Hz are only ever computed on the
    test split - roughly a tenth of that. Loading everything would push a
    laptop into swap partway through a multi-hour run for no benefit.

    Parameters
    ----------
    metadata:
        The filtered metadata table the positions index into.
    positions:
        Integer positions (not ``ecg_id`` values) within ``metadata``.
    """
    positions = np.asarray(positions, dtype=np.int64)
    subset = metadata.iloc[positions]
    signals = [signal for _, signal in iter_ptbxl_waveforms(root, subset, sampling_rate_hz)]
    stacked = np.stack(signals).astype(np.float32)
    return np.nan_to_num(stacked, nan=0.0)


def official_split(
    strat_folds: np.ndarray, patient_ids: np.ndarray | None = None
) -> SplitIndices:
    """The dataset's recommended split: folds 1-8 train, 9 validation, 10 test.

    Using PTB-XL's own stratified folds keeps results comparable with published
    work, and folds 9 and 10 are the ones that received human over-reading, so
    they are the most reliable choice for validation and test.

    When ``patient_ids`` is supplied the split is *verified* to respect patient
    boundaries rather than assumed to. The folds are constructed to be
    patient-consistent, but this project's central results depend on that being
    true, and checking costs nothing.
    """
    strat_folds = np.asarray(strat_folds)
    splits = SplitIndices(
        train=np.flatnonzero(strat_folds <= 8),
        val=np.flatnonzero(strat_folds == 9),
        test=np.flatnonzero(strat_folds == 10),
    )
    if min(splits.sizes.values()) == 0:
        raise ValueError(
            f"one of the official folds is empty: sizes {splits.sizes}. Expected "
            "strat_fold values in 1-10."
        )
    if patient_ids is not None:
        assert_no_patient_leakage(patient_ids, splits)
    return splits


def summarise_ptbxl(root: str | Path, config: DataConfig | None = None) -> str:
    """A human-readable summary of what is in the downloaded dataset.

    Cheap to run - reads only the metadata - and worth running immediately after
    a download to confirm it is complete and behaves as documented.
    """
    config = config or DataConfig()
    root = Path(root)
    raw = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    filtered = load_ptbxl_metadata(root, config)

    n_sentinel = int((raw.age == AGE_SENTINEL).sum())
    counts = {
        name: int(filtered[f"dx_{name}"].sum()) for name in DIAGNOSTIC_SUPERCLASSES
    }
    unlabelled = int(
        (filtered[[f"dx_{n}" for n in DIAGNOSTIC_SUPERCLASSES]].sum(axis=1) == 0).sum()
    )

    return "\n".join([
        f"PTB-XL at {root}",
        f"  raw:      {len(raw):,} recordings, {raw.patient_id.nunique():,} patients",
        f"            {n_sentinel:,} with the age-300 sentinel (>89 years, dropped)",
        f"            age range {raw.age.min():.0f}-{raw.age[raw.age != AGE_SENTINEL].max():.0f}",
        f"  filtered: {len(filtered):,} recordings, {filtered.patient_id.nunique():,} patients",
        f"            age {filtered.age.min():.0f}-{filtered.age.max():.0f} "
        f"(mean {filtered.age.mean():.1f}), {100 * filtered.sex.mean():.0f}% female",
        f"            repeat patients: "
        f"{len(filtered) - filtered.patient_id.nunique():,} extra recordings",
        f"  diagnostic superclasses: {counts}",
        f"            {unlabelled:,} recordings carry no diagnostic superclass",
        f"  official folds: train {(filtered.strat_fold <= 8).sum():,}, "
        f"val {(filtered.strat_fold == 9).sum():,}, "
        f"test {(filtered.strat_fold == 10).sum():,}",
    ])
