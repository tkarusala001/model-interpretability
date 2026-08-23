"""Tests for rhythm statements as known features and as a stratification mask.

These feed two things that bear directly on the project's surviving claim: an
enumeration that includes rhythm, and a sinus-only subset in which P-wave
*presence* is constant by construction. A mask that quietly admitted atrial
fibrillation would destroy the argument the subset exists to make, so the
exclusion behaviour is tested harder than the happy path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecg_discovery.data.rhythm_labels import (
    RHYTHM_FEATURE_NAMES,
    is_sinus_rhythm,
    parse_scp_codes,
    rhythm_indicators,
)

SINUS = "{'NORM': 100.0, 'SR': 0.0}"
AFIB = "{'AFIB': 100.0, 'IMI': 15.0}"
FLUTTER = "{'AFLT': 100.0}"
PACED = "{'PACE': 100.0}"
TACHY = "{'STACH': 100.0}"
UNLABELLED = "{'NORM': 100.0}"


def test_parses_ptbxl_stringified_dicts():
    assert parse_scp_codes(SINUS) == {"NORM": 100.0, "SR": 0.0}
    assert parse_scp_codes({"AFIB": 100.0}) == {"AFIB": 100.0}


def test_malformed_rows_carry_no_statements_rather_than_raising():
    """One bad row must not abort a cohort load."""
    for bad in ("{not a dict", "", None, float("nan"), "[1, 2]"):
        assert parse_scp_codes(bad) == {}


def test_indicators_are_multi_hot_over_the_named_categories():
    frame = rhythm_indicators([SINUS, AFIB, FLUTTER, PACED, TACHY])
    assert list(frame.columns) == list(RHYTHM_FEATURE_NAMES)
    assert frame["rhythm_sinus"].tolist() == [1, 0, 0, 0, 1]
    assert frame["rhythm_atrial_fibrillation"].tolist() == [0, 1, 0, 0, 0]
    assert frame["rhythm_atrial_flutter"].tolist() == [0, 0, 1, 0, 0]
    assert frame["rhythm_paced"].tolist() == [0, 0, 0, 1, 0]


def test_a_recording_without_a_rhythm_statement_is_all_zero_not_imputed():
    """An absent statement is unlabelled, not confirmed sinus."""
    frame = rhythm_indicators([UNLABELLED])
    assert frame.iloc[0].sum() == 0


def test_sinus_mask_excludes_every_rhythm_lacking_an_organised_p_wave():
    """The claim the subset supports is that every recording in it has a P wave."""
    codes = [SINUS, AFIB, FLUTTER, PACED, TACHY]
    mask = is_sinus_rhythm(codes)
    assert mask.tolist() == [True, False, False, False, True]


def test_unlabelled_recordings_are_excluded_by_default():
    """Treating unlabelled as sinus would readmit undetected AF."""
    assert is_sinus_rhythm([UNLABELLED]).tolist() == [False]
    assert is_sinus_rhythm([UNLABELLED], require_explicit=False).tolist() == [True]


def test_a_sinus_statement_alongside_af_does_not_count_as_sinus():
    """Mixed statements must fail closed, not open."""
    mixed = "{'SR': 100.0, 'AFIB': 100.0}"
    assert is_sinus_rhythm([mixed]).tolist() == [False]
    assert is_sinus_rhythm([mixed], require_explicit=False).tolist() == [False]


def test_rate_variants_of_sinus_are_kept():
    """Tachycardia and bradycardia still have organised P waves."""
    codes = ["{'STACH': 100.0}", "{'SBRAD': 100.0}", "{'SARRH': 100.0}"]
    assert is_sinus_rhythm(codes).all()


def test_indicators_align_with_the_mask():
    codes = [SINUS, AFIB, PACED, UNLABELLED, TACHY]
    frame = rhythm_indicators(codes)
    mask = is_sinus_rhythm(codes)
    # Anything masked in must be flagged sinus and nothing else disqualifying.
    for i, keep in enumerate(mask):
        if keep:
            assert frame.loc[i, "rhythm_sinus"] == 1
            assert frame.loc[i, "rhythm_atrial_fibrillation"] == 0
            assert frame.loc[i, "rhythm_paced"] == 0


def test_indicators_are_usable_as_known_features():
    """They must drop into a decomposition without further conversion."""
    from ecg_discovery.config import ValidationFrameworkConfig
    from ecg_discovery.validation.residual_decomposition import decompose_age_gap

    rng = np.random.default_rng(0)
    n = 400
    codes = [AFIB if i % 4 == 0 else SINUS for i in range(n)]
    rhythm = rhythm_indicators(codes)
    # An age gap driven entirely by rhythm must be fully attributed to it.
    gap = 6.0 * rhythm["rhythm_atrial_fibrillation"].to_numpy() + rng.normal(size=n)
    features = pd.concat(
        [rhythm, pd.DataFrame({"qrs_duration_ms": rng.normal(95, 10, n)})], axis=1
    )
    config = ValidationFrameworkConfig(
        known_features=RHYTHM_FEATURE_NAMES + ("qrs_duration_ms",),
        explainer_models=("linear",), cv_folds=4, bootstrap_iterations=200,
    )
    result = decompose_age_gap(gap, features, config, compute_univariate=False)
    assert result.most_explanatory.r2_incremental > 0.7
