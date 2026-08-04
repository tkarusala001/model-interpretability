"""Loading the Chapman-Shaoxing-Ningbo ECG database, for external replication.

**Requires the dataset downloaded per ``scripts/download_chapman.sh``.**

WHY A SECOND COHORT
-------------------
Every result in this project up to now comes from PTB-XL: one institution, one
country, one set of recording devices. A finding that holds only there is a
finding about that hospital's data. This cohort is the external check.

45,152 twelve-lead ECGs from Chapman University, Shaoxing People's Hospital and
Ningbo First Hospital - a different country, different institutions and
different equipment from PTB-XL's German cohort. Ten seconds at 500 Hz, which
matches this project's design exactly: the model's 100 Hz view is derived by
resampling, and interval measurement uses the native 500 Hz.

Licensed CC BY 4.0, fully open - no credentialing, data use agreement or account
(verified 2026-07-30 against
https://physionet.org/content/ecg-arrhythmia/1.0.0/).

Attribution required by the licence:
    Zheng, J., Zhang, J., Danioko, S., Yao, H., Guo, H., & Rakovski, C. (2020).
    A 12-lead electrocardiogram database for arrhythmia research covering more
    than 10,000 patients. Scientific Data, 7, 48.
    PhysioNet: https://doi.org/10.13026/wgex-er52

DIFFERENCES FROM PTB-XL THAT THE ADAPTER ABSORBS
------------------------------------------------
The point of this module is that *only loading* differs. Everything downstream -
delineation, interval measurement, the decomposition, attribution, occlusion -
receives identical arrays and runs unchanged.

- **Metadata location.** PTB-XL keeps age and sex in a CSV keyed by ``ecg_id``;
  here they live in WFDB header comments (``#Age: 63``, ``#Sex: Male``).
- **No 100 Hz copy.** PTB-XL ships both rates. This cohort is 500 Hz only, so
  the model's view is resampled on load. Recordings are resampled **as they are
  read**, never all at once: 45,152 recordings at 500 Hz would be 10.8 GB as
  float32, against 2.2 GB at 100 Hz.
- **One recording per patient.** There is no patient identifier and no repeat
  visits, so each recording is treated as its own patient. Patient-level
  splitting is therefore trivially satisfied, which is worth stating rather than
  assuming - it is the one hazard this cohort does not have.
- **Diagnoses are SNOMED-CT codes**, not SCP statements, and do not map cleanly
  onto PTB-XL's five superclasses. They are *not* loaded. This cohort is used to
  replicate the age-gap decomposition and the attribution finding, not the
  diagnostic-link experiment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from ecg_discovery.config import DataConfig
from ecg_discovery.data.preprocessing import resample_signals

__all__ = [
    "CHAPMAN_LICENCE",
    "ChapmanData",
    "load_chapman",
    "load_chapman_waveform_subset",
    "summarise_chapman",
    "is_downloaded",
]

CHAPMAN_LICENCE = "CC BY 4.0"
NATIVE_RATE_HZ = 500

_AGE_PATTERN = re.compile(r"#\s*Age:\s*([0-9.]+)", re.IGNORECASE)
_SEX_PATTERN = re.compile(r"#\s*Sex:\s*(\w+)", re.IGNORECASE)


@dataclass
class ChapmanData:
    """Loaded cohort, in the same shape :mod:`ecg_discovery.training.train` expects."""

    signals: np.ndarray
    ages: np.ndarray
    sexes: np.ndarray
    patient_ids: np.ndarray
    record_ids: list[str]
    sampling_rate_hz: int
    metadata: pd.DataFrame

    def __len__(self) -> int:
        return int(self.signals.shape[0])


def is_downloaded(root: str | Path) -> bool:
    """Whether the dataset appears present and usable."""
    root = Path(root)
    records = root / "WFDBRecords"
    return records.is_dir() and any(records.iterdir())


def _require_dataset(root: Path) -> None:
    if not is_downloaded(root):
        raise FileNotFoundError(
            f"Chapman-Shaoxing-Ningbo not found at {root}.\n"
            "Download it first:  bash scripts/download_chapman.sh"
        )


def _header_paths(root: Path) -> list[Path]:
    """Every ``.hea`` file, in a deterministic order.

    Sorted so the cohort is identical between runs - without this, filesystem
    ordering would silently change which recordings a ``--limit`` run sees.
    """
    return sorted((root / "WFDBRecords").rglob("*.hea"))


def _parse_header(path: Path) -> tuple[float, int] | None:
    """Extract ``(age, sex)`` from a WFDB header, or ``None`` if unusable.

    Sex is encoded to match PTB-XL's convention (0 = male, 1 = female) so the
    two cohorts are interchangeable downstream. Records with a missing or
    unparseable age or sex are skipped rather than guessed at.
    """
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return None

    age_match = _AGE_PATTERN.search(text)
    sex_match = _SEX_PATTERN.search(text)
    if not age_match or not sex_match:
        return None
    try:
        age = float(age_match.group(1))
    except ValueError:
        return None

    sex_text = sex_match.group(1).strip().lower()
    if sex_text.startswith("m"):
        sex = 0
    elif sex_text.startswith("f"):
        sex = 1
    else:
        return None
    return age, sex


def load_chapman_metadata(
    root: str | Path, config: DataConfig | None = None
) -> pd.DataFrame:
    """Scan headers and return the filtered cohort table, without waveforms.

    Applies the same age filtering as the PTB-XL loader, so the two cohorts are
    comparable: adults only, since an age model spanning paediatric and adult
    ECGs is modelling growth as much as ageing.
    """
    config = config or DataConfig()
    root = Path(root)
    _require_dataset(root)

    rows = []
    for header in _header_paths(root):
        parsed = _parse_header(header)
        if parsed is None:
            continue
        age, sex = parsed
        rows.append({
            "record_id": header.stem,
            "path": str(header.with_suffix("")),
            "age": age,
            "sex": sex,
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no readable headers found under {root}")

    frame = frame[frame.age.between(config.min_age_years, config.max_age_years)]
    if frame.empty:
        raise ValueError(
            f"no recordings survived age filtering "
            f"({config.min_age_years}-{config.max_age_years})"
        )
    return frame.reset_index(drop=True)


def _read_record(path: str) -> np.ndarray | None:
    """Read one WFDB record as ``(12, n_samples)`` millivolts, or None on failure."""
    try:
        import wfdb
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise ImportError("reading this dataset requires `wfdb`") from error
    try:
        record = wfdb.rdrecord(path)
    except Exception:
        return None
    signal = np.asarray(record.p_signal, dtype=np.float32).T
    if signal.ndim != 2 or signal.shape[0] != 12:
        return None
    return signal


def load_chapman(
    root: str | Path,
    config: DataConfig | None = None,
    limit: int | None = None,
    target_rate_hz: int | None = None,
    progress: bool = False,
) -> ChapmanData:
    """Load the cohort at the model's sampling rate.

    Recordings are resampled **during** loading rather than afterwards: the
    native 500 Hz set is 10.8 GB as float32 across 45k recordings, against
    2.2 GB at 100 Hz. Holding the full-rate array would exhaust a laptop.

    Parameters
    ----------
    limit:
        Load only the first N recordings after filtering, in deterministic
        header order.
    target_rate_hz:
        Defaults to ``config.sampling_rate_hz`` (the model's rate).
    """
    config = config or DataConfig()
    target = target_rate_hz or config.sampling_rate_hz
    root = Path(root)

    metadata = load_chapman_metadata(root, config)
    if limit is not None:
        metadata = metadata.iloc[:limit].reset_index(drop=True)

    signals: list[np.ndarray] = []
    keep: list[int] = []
    for position, path in enumerate(metadata.path):
        signal = _read_record(path)
        if signal is None:
            continue
        if target != NATIVE_RATE_HZ:
            signal = resample_signals(signal[None, ...], NATIVE_RATE_HZ, target)[0]
        signals.append(np.nan_to_num(signal, nan=0.0))
        keep.append(position)
        if progress and position and position % 2000 == 0:
            print(f"  loaded {position}/{len(metadata)} recordings")

    if not signals:
        raise ValueError("no recordings could be read")

    metadata = metadata.iloc[keep].reset_index(drop=True)
    stacked = np.stack(signals).astype(np.float32)

    return ChapmanData(
        signals=stacked,
        ages=metadata.age.to_numpy(dtype=np.float64),
        sexes=metadata.sex.to_numpy(dtype=np.int64),
        # No patient identifier exists and there are no repeat visits, so each
        # recording is its own patient. Stated explicitly because the
        # patient-level split machinery still runs - it simply has nothing to do.
        patient_ids=np.arange(len(metadata), dtype=np.int64),
        record_ids=metadata.record_id.tolist(),
        sampling_rate_hz=target,
        metadata=metadata,
    )


def load_chapman_waveform_subset(
    metadata: pd.DataFrame, positions: Sequence[int] | np.ndarray
) -> np.ndarray:
    """Load selected recordings at the native 500 Hz, for interval measurement.

    Mirrors the PTB-XL loader's lazy subset path: interval features need the
    higher rate, but only for the test split.
    """
    positions = np.asarray(positions, dtype=np.int64)
    signals = []
    for path in metadata.iloc[positions].path:
        signal = _read_record(path)
        if signal is None:
            raise ValueError(f"could not read {path}, needed for interval measurement")
        signals.append(np.nan_to_num(signal, nan=0.0))
    return np.stack(signals).astype(np.float32)


def summarise_chapman(root: str | Path, config: DataConfig | None = None) -> str:
    """Human-readable summary, for verifying the download."""
    config = config or DataConfig()
    root = Path(root)
    total_headers = len(_header_paths(root))
    frame = load_chapman_metadata(root, config)
    return "\n".join([
        f"Chapman-Shaoxing-Ningbo at {root}",
        f"  raw:      {total_headers:,} header files",
        f"  filtered: {len(frame):,} recordings with a usable age and sex,",
        f"            age {frame.age.min():.0f}-{frame.age.max():.0f} "
        f"(mean {frame.age.mean():.1f}), {100 * frame.sex.mean():.0f}% female",
        f"  native rate: {NATIVE_RATE_HZ} Hz, 10 s, 12 leads",
        f"  licence: {CHAPMAN_LICENCE} (open access, no credentialing)",
        "  note: one recording per patient; no repeat visits in this cohort",
    ])
