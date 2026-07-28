"""Tests for PTB-XL loading.

The real dataset is a 3 GB download, so these tests build a miniature stand-in
with the same schema. That is enough to exercise everything that can silently
corrupt results - the age-300 sentinel, diagnostic superclass aggregation, and
the official fold split - without requiring the data to be present.

What these tests cannot verify is anything about the real file contents. Those
checks live in :func:`summarise_ptbxl`, which is meant to be run immediately
after downloading; ``docs/reproducibility.md`` records what it should print.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecg_discovery.config import DataConfig
from ecg_discovery.data.ptbxl_dataset import (
    AGE_SENTINEL,
    DIAGNOSTIC_SUPERCLASSES,
    load_ptbxl_metadata,
    official_split,
)


@pytest.fixture
def fake_ptbxl(tmp_path):
    """A miniature dataset with PTB-XL's schema and its awkward cases."""
    database = pd.DataFrame({
        "ecg_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "patient_id": [10, 10, 11, 12, 13, 14, 15, 16],
        "age": [55.0, 55.0, AGE_SENTINEL, 8.0, 71.0, 64.0, 39.0, 82.0],
        "sex": [0, 0, 1, 1, 1, 0, 1, 0],
        "scp_codes": [
            "{'NORM': 100.0, 'SR': 0.0}",
            "{'IMI': 80.0}",
            "{'NORM': 100.0}",
            "{'NORM': 100.0}",
            "{'NDT': 100.0, 'LVH': 50.0}",
            "{'SR': 0.0}",                 # no diagnostic statement at all
            "{'CLBBB': 100.0}",
            "not-a-dict",                  # malformed, must not crash
        ],
        "strat_fold": [1, 2, 3, 4, 9, 10, 8, 10],
        "filename_lr": [f"records100/00000/{i:05d}_lr" for i in range(1, 9)],
        "filename_hr": [f"records500/00000/{i:05d}_hr" for i in range(1, 9)],
    })
    database.to_csv(tmp_path / "ptbxl_database.csv", index=False)

    statements = pd.DataFrame({
        "Unnamed: 0": ["NORM", "IMI", "NDT", "LVH", "CLBBB", "SR"],
        "diagnostic": [1, 1, 1, 1, 1, 0],
        "diagnostic_class": ["NORM", "MI", "STTC", "HYP", "CD", np.nan],
    })
    statements.to_csv(tmp_path / "scp_statements.csv", index=False)
    return tmp_path


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def test_age_sentinel_is_dropped_not_clipped(fake_ptbxl):
    """300 is anonymisation, not an age, and must never reach the target.

    It is numerically plausible, so it would survive any filter phrased purely
    as a maximum and would drag the regression target and every age-gap
    residual with it.
    """
    frame = load_ptbxl_metadata(fake_ptbxl, DataConfig())
    assert AGE_SENTINEL not in set(frame.age)
    assert 3 not in frame.index          # the sentinel record itself is gone
    assert frame.age.max() <= 89.0


def test_children_are_excluded_by_default(fake_ptbxl):
    """Growth is not ageing; an 8-year-old belongs to a different phenomenon."""
    frame = load_ptbxl_metadata(fake_ptbxl, DataConfig())
    assert frame.age.min() >= 18.0
    assert 4 not in frame.index


def test_age_bounds_are_configurable(fake_ptbxl):
    import dataclasses

    config = dataclasses.replace(DataConfig(), min_age_years=60.0, max_age_years=75.0)
    frame = load_ptbxl_metadata(fake_ptbxl, config)
    assert set(frame.age) == {71.0, 64.0}


def test_everything_filtered_out_raises(fake_ptbxl):
    import dataclasses

    config = dataclasses.replace(DataConfig(), min_age_years=88.0, max_age_years=89.0)
    with pytest.raises(ValueError, match="no recordings survived"):
        load_ptbxl_metadata(fake_ptbxl, config)


# --------------------------------------------------------------------------- #
# Diagnostic superclasses
# --------------------------------------------------------------------------- #
def test_scp_codes_map_to_superclasses(fake_ptbxl):
    frame = load_ptbxl_metadata(fake_ptbxl, DataConfig())
    assert frame.loc[1, "dx_NORM"] == 1
    assert frame.loc[2, "dx_MI"] == 1
    assert frame.loc[7, "dx_CD"] == 1


def test_labels_are_multi_label(fake_ptbxl):
    """One recording can carry several superclasses; PTB-XL is not exclusive."""
    frame = load_ptbxl_metadata(fake_ptbxl, DataConfig())
    assert frame.loc[5, "dx_STTC"] == 1 and frame.loc[5, "dx_HYP"] == 1


def test_recordings_without_a_diagnostic_statement_are_all_zero(fake_ptbxl):
    """Not every ECG has a diagnostic label; that must not become a false NORM."""
    frame = load_ptbxl_metadata(fake_ptbxl, DataConfig())
    columns = [f"dx_{name}" for name in DIAGNOSTIC_SUPERCLASSES]
    assert frame.loc[6, columns].sum() == 0


def test_malformed_scp_codes_do_not_crash(fake_ptbxl):
    frame = load_ptbxl_metadata(fake_ptbxl, DataConfig())
    columns = [f"dx_{name}" for name in DIAGNOSTIC_SUPERCLASSES]
    assert frame.loc[8, columns].sum() == 0


def test_non_diagnostic_codes_are_ignored(fake_ptbxl):
    """SR (sinus rhythm) is a rhythm statement, not a diagnostic superclass."""
    frame = load_ptbxl_metadata(fake_ptbxl, DataConfig())
    assert frame.loc[1, "dx_NORM"] == 1     # from NORM, not from SR


# --------------------------------------------------------------------------- #
# The official split
# --------------------------------------------------------------------------- #
def test_official_split_uses_folds_1_to_8_9_and_10():
    folds = np.array([1, 2, 8, 9, 10, 3, 9, 10])
    splits = official_split(folds)
    assert sorted(splits.train.tolist()) == [0, 1, 2, 5]
    assert sorted(splits.val.tolist()) == [3, 6]
    assert sorted(splits.test.tolist()) == [4, 7]


def test_official_split_verifies_patient_integrity():
    """PTB-XL's folds are patient-consistent; this project checks rather than trusts.

    The central results depend on it, and checking costs nothing.
    """
    folds = np.array([1, 9, 10, 2])
    leaking_patients = np.array([100, 100, 101, 102])   # patient 100 in train and val
    with pytest.raises(ValueError, match="leakage"):
        official_split(folds, leaking_patients)

    clean_patients = np.array([100, 101, 102, 103])
    official_split(folds, clean_patients)               # must not raise


def test_official_split_rejects_missing_folds():
    with pytest.raises(ValueError, match="empty"):
        official_split(np.array([1, 2, 3, 4]))          # no fold 9 or 10


def test_official_split_on_the_fake_dataset(fake_ptbxl):
    frame = load_ptbxl_metadata(fake_ptbxl, DataConfig())
    splits = official_split(
        frame.strat_fold.to_numpy(), frame.patient_id.to_numpy()
    )
    assert sum(splits.sizes.values()) == len(frame)


# --------------------------------------------------------------------------- #
# Missing dataset
# --------------------------------------------------------------------------- #
def test_missing_dataset_gives_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.sh"):
        load_ptbxl_metadata(tmp_path / "nope", DataConfig())


def test_incomplete_download_gives_an_actionable_error(tmp_path):
    (tmp_path / "ptbxl_database.csv").write_text("ecg_id\n1\n")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        load_ptbxl_metadata(tmp_path, DataConfig())


def test_download_helpers_report_the_verified_terms():
    from ecg_discovery.data.download_ptbxl import (
        PTBXL_BASE_URL,
        PTBXL_LICENCE,
        PTBXL_VERSION,
        is_downloaded,
    )

    # Verified against physionet.org on 2026-07-28. The build plan said ODC-BY;
    # the dataset is actually CC BY 4.0, and the paper must cite the right one.
    assert PTBXL_LICENCE == "CC BY 4.0"
    assert PTBXL_VERSION == "1.0.3"
    assert PTBXL_VERSION in PTBXL_BASE_URL
    assert not is_downloaded("/nonexistent/path")
