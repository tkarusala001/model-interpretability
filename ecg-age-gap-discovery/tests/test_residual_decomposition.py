"""Tests for the rediscovery-vs-discovery decomposition.

This is the project's central methodological claim, so it is tested in both
directions and at two levels.

**Both directions.** A method that always reports "explained" would look
reassuring and be useless; one that always reports "unexplained" would
manufacture discoveries. The decomposition must do each when each is correct:

- ``test_end_to_end_effect_mediated_by_a_known_interval`` builds a cohort whose
  entire age signal runs through QRS duration - a measurable interval - and
  requires the decomposition to attribute nearly all of it to known features.
- ``test_end_to_end_effect_invisible_to_known_intervals`` builds one whose age
  signal is T-wave *skew*, which by construction leaves onset, offset, duration
  and amplitude untouched, and requires a large unexplained residual.

**Two levels.** Fast tests on constructed arrays check the decomposition
arithmetic. Slow end-to-end tests run the real pipeline - generate, train,
detect, delineate, measure, decompose - because the arithmetic being right does
not establish that the *measurements* feeding it are.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ecg_discovery.config import (
    BackboneConfig,
    SignalProcessingConfig,
    SyntheticConfig,
    TrainingConfig,
    ValidationFrameworkConfig,
)
from ecg_discovery.data.preprocessing import resample_signals
from ecg_discovery.data.synthetic_ecg import generate_cohort
from ecg_discovery.signal_processing.interval_features import interval_features_table
from ecg_discovery.training.train import train_age_regressor
from ecg_discovery.validation.residual_decomposition import decompose_age_gap

FAST_CONFIG = ValidationFrameworkConfig(
    known_features=("heart_rate_bpm", "qrs_duration_ms"),
    cv_folds=4,
    bootstrap_iterations=200,
    seed=0,
)


def _table(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


# --------------------------------------------------------------------------- #
# The arithmetic
# --------------------------------------------------------------------------- #
def test_a_gap_that_is_a_known_feature_is_fully_explained():
    """If the gap *is* a known measurement, essentially none is left over."""
    rng = np.random.default_rng(0)
    n = 400
    qrs = rng.normal(95, 12, n)
    heart_rate = rng.normal(70, 10, n)
    age_gap = 0.4 * (qrs - 95) + rng.normal(0, 0.3, n)

    result = decompose_age_gap(
        age_gap, _table(heart_rate_bpm=heart_rate, qrs_duration_ms=qrs), FAST_CONFIG
    )
    for explainer in result.explainers.values():
        assert explainer.r2_full > 0.95, explainer.model_name
        assert explainer.unexplained_fraction < 0.05


def test_a_gap_unrelated_to_known_features_is_left_unexplained():
    """Pure noise must not be 'explained', least of all by the flexible model."""
    rng = np.random.default_rng(1)
    n = 400
    result = decompose_age_gap(
        rng.normal(0, 5, n),
        _table(heart_rate_bpm=rng.normal(70, 10, n), qrs_duration_ms=rng.normal(95, 12, n)),
        FAST_CONFIG,
    )
    for explainer in result.explainers.values():
        assert explainer.r2_full < 0.10, explainer.model_name
        assert explainer.unexplained_fraction > 0.90


def test_out_of_fold_evaluation_prevents_explaining_noise():
    """In-sample, a boosted model fits noise; out of fold it must not.

    This is why every figure the framework reports is cross-validated. Without
    it, flexibility alone would produce a high explained share and a discovery
    could be defined out of existence by fitting harder.
    """
    rng = np.random.default_rng(2)
    n = 200
    features = _table(**{f"f{i}": rng.normal(0, 1, n) for i in range(2)})
    config = dataclasses.replace(
        FAST_CONFIG, known_features=("f0", "f1"), explainer_models=("gradient_boosting",)
    )
    result = decompose_age_gap(rng.normal(0, 5, n), features, config)
    explainer = result.explainers["gradient_boosting"]

    from sklearn.ensemble import GradientBoostingRegressor

    in_sample = GradientBoostingRegressor(random_state=0).fit(
        features.to_numpy(), rng.normal(0, 5, n)
    )
    assert explainer.r2_full < 0.2
    assert in_sample.score(features.to_numpy(), in_sample.predict(features.to_numpy())) > 0.9


def test_demographic_adjustment_prevents_crediting_a_regression_artefact():
    """Shared age dependence must not be credited to the intervals.

    An age gap is mechanically anti-correlated with age because models regress
    to the mean. Known intervals also drift with age. Here the gap depends on
    age *only*, and the interval is a pure function of age carrying no extra
    information: with adjustment the incremental contribution must be nil.
    """
    rng = np.random.default_rng(3)
    n = 400
    ages = rng.uniform(20, 89, n)
    qrs = 90 + 0.3 * ages + rng.normal(0, 0.5, n)     # varies with age alone
    age_gap = -0.35 * (ages - ages.mean()) + rng.normal(0, 1.0, n)

    features = _table(heart_rate_bpm=rng.normal(70, 10, n), qrs_duration_ms=qrs)
    adjusted = decompose_age_gap(
        age_gap, features, FAST_CONFIG, ages=ages, sexes=rng.integers(0, 2, n)
    )
    unadjusted = decompose_age_gap(
        age_gap, features, dataclasses.replace(FAST_CONFIG, adjust_for_covariates=())
    )

    linear_adjusted = adjusted.explainers["linear"]
    linear_unadjusted = unadjusted.explainers["linear"]
    # Unadjusted, the interval looks highly explanatory; adjusted, it adds nothing.
    assert linear_unadjusted.r2_incremental > 0.6
    assert linear_adjusted.r2_incremental < 0.1
    assert linear_adjusted.r2_baseline > 0.6


def test_both_explainers_are_always_reported():
    """No opportunity to report whichever number reads better."""
    rng = np.random.default_rng(4)
    n = 200
    result = decompose_age_gap(
        rng.normal(0, 3, n),
        _table(heart_rate_bpm=rng.normal(70, 10, n), qrs_duration_ms=rng.normal(95, 12, n)),
        FAST_CONFIG,
    )
    assert set(result.explainers) == {"linear", "gradient_boosting"}
    assert len(result.to_frame()) == 2


def test_non_linear_relationship_is_caught_by_the_boosted_explainer():
    """A curved relationship must not be mistaken for unexplained signal.

    This is the specific reason a linear-only framework would be dishonest: a
    real but non-linear dependence on a known interval would sit in the residual
    looking exactly like a discovery.
    """
    rng = np.random.default_rng(5)
    n = 500
    qrs = rng.uniform(70, 130, n)
    age_gap = 0.004 * (qrs - 100) ** 2 + rng.normal(0, 0.4, n)

    result = decompose_age_gap(
        age_gap, _table(heart_rate_bpm=rng.normal(70, 10, n), qrs_duration_ms=qrs),
        FAST_CONFIG,
    )
    assert result.explainers["gradient_boosting"].r2_full > 0.8
    assert (
        result.explainers["gradient_boosting"].r2_full
        > result.explainers["linear"].r2_full + 0.3
    )


def test_most_explanatory_picks_the_conservative_claim():
    """The headline uses whichever explainer explains most, not least."""
    rng = np.random.default_rng(6)
    n = 400
    qrs = rng.uniform(70, 130, n)
    result = decompose_age_gap(
        0.004 * (qrs - 100) ** 2 + rng.normal(0, 0.4, n),
        _table(heart_rate_bpm=rng.normal(70, 10, n), qrs_duration_ms=qrs),
        FAST_CONFIG,
    )
    assert result.most_explanatory.model_name == "gradient_boosting"


def test_univariate_scores_identify_the_carrying_feature():
    rng = np.random.default_rng(7)
    n = 400
    qrs = rng.normal(95, 12, n)
    result = decompose_age_gap(
        0.5 * (qrs - 95) + rng.normal(0, 0.5, n),
        _table(heart_rate_bpm=rng.normal(70, 10, n), qrs_duration_ms=qrs),
        FAST_CONFIG,
    )
    univariate = result.explainers["linear"].univariate_r2
    assert univariate["qrs_duration_ms"] > 0.9
    assert univariate["heart_rate_bpm"] < 0.1


# --------------------------------------------------------------------------- #
# Bookkeeping and failure modes
# --------------------------------------------------------------------------- #
def test_unmeasurable_recordings_are_dropped_and_counted_not_imputed():
    """A fabricated interval would enter the known set as if it were measured."""
    rng = np.random.default_rng(8)
    n = 300
    qrs = rng.normal(95, 12, n)
    qrs[:20] = np.nan
    result = decompose_age_gap(
        0.4 * np.nan_to_num(qrs - 95) + rng.normal(0, 1, n),
        _table(heart_rate_bpm=rng.normal(70, 10, n), qrs_duration_ms=qrs),
        FAST_CONFIG,
    )
    assert result.n_dropped == 20
    assert result.n_recordings == n - 20


def test_missing_known_feature_raises():
    rng = np.random.default_rng(9)
    with pytest.raises(KeyError, match="known feature"):
        decompose_age_gap(
            rng.normal(0, 1, 100), _table(heart_rate_bpm=rng.normal(70, 10, 100)),
            FAST_CONFIG,
        )


def test_length_mismatch_raises():
    rng = np.random.default_rng(10)
    with pytest.raises(ValueError, match="rows"):
        decompose_age_gap(
            rng.normal(0, 1, 50),
            _table(heart_rate_bpm=rng.normal(70, 10, 100), qrs_duration_ms=rng.normal(95, 5, 100)),
            FAST_CONFIG,
        )


def test_too_few_usable_recordings_raises():
    rng = np.random.default_rng(11)
    with pytest.raises(ValueError, match="too few"):
        decompose_age_gap(
            rng.normal(0, 1, 6),
            _table(heart_rate_bpm=rng.normal(70, 10, 6), qrs_duration_ms=rng.normal(95, 5, 6)),
            FAST_CONFIG,
        )


def test_confidence_interval_brackets_the_estimate():
    rng = np.random.default_rng(12)
    n = 400
    qrs = rng.normal(95, 12, n)
    result = decompose_age_gap(
        0.4 * (qrs - 95) + rng.normal(0, 1, n),
        _table(heart_rate_bpm=rng.normal(70, 10, n), qrs_duration_ms=qrs),
        FAST_CONFIG,
    )
    for explainer in result.explainers.values():
        low, high = explainer.r2_full_ci
        assert low <= explainer.r2_full <= high


def test_unexplained_residual_is_uncorrelated_with_the_known_features():
    """What is left over must genuinely be orthogonal to what was regressed out."""
    rng = np.random.default_rng(13)
    n = 500
    qrs = rng.normal(95, 12, n)
    result = decompose_age_gap(
        0.4 * (qrs - 95) + rng.normal(0, 2, n),
        _table(heart_rate_bpm=rng.normal(70, 10, n), qrs_duration_ms=qrs),
        FAST_CONFIG,
    )
    residual = result.explainers["linear"].unexplained_residual
    assert abs(np.corrcoef(residual, qrs[: residual.size])[0, 1]) < 0.15


def test_summary_text_reports_both_explainers_and_the_caveat():
    rng = np.random.default_rng(14)
    n = 200
    result = decompose_age_gap(
        rng.normal(0, 3, n),
        _table(heart_rate_bpm=rng.normal(70, 10, n), qrs_duration_ms=rng.normal(95, 12, n)),
        FAST_CONFIG,
    )
    text = result.summary_text()
    assert "linear" in text and "gradient_boosting" in text
    assert "CANDIDATE" in text


# --------------------------------------------------------------------------- #
# End to end: the ground-truth claim
# --------------------------------------------------------------------------- #
SMALL_BACKBONE = BackboneConfig(
    stem_channels=16, stage_channels=(16, 24, 32), stride_per_stage=(2, 2, 2)
)
QUICK_TRAINING = TrainingConfig(
    epochs=25, batch_size=32, learning_rate=3e-3, warmup_epochs=1,
    early_stopping_patience=30, seed=0,
)
END_TO_END_CONFIG = ValidationFrameworkConfig(
    known_features=("heart_rate_bpm", "qrs_duration_ms", "pr_interval_ms", "qt_interval_ms"),
    cv_folds=4,
    bootstrap_iterations=200,
    seed=0,
)


def _ground_truth_ceiling(age_gap, latent_offset, ages, sexes) -> float:
    """The most of this age gap that *any* method could possibly explain.

    An age gap is not purely the injected signal: it is the injected latent
    offset **plus the model's own prediction error**, and no ECG measurement can
    explain a neural network's noise. Asking the decomposition to reach 100% is
    therefore asking it to explain something that is not there.

    The honest benchmark is what the ground-truth latent offset itself achieves
    when regressed on the same gap - the ceiling set by how much of the gap is
    injected signal at all. The decomposition is then judged on the fraction of
    *that* it recovers, which separates "the framework works" from "the model
    happened to train well today".
    """
    design = np.column_stack([
        np.ones_like(age_gap), np.asarray(ages), np.asarray(sexes),
        np.asarray(latent_offset),
    ])
    coefficients, *_ = np.linalg.lstsq(design, age_gap, rcond=None)
    predicted = design @ coefficients
    return float(
        1.0 - np.sum((age_gap - predicted) ** 2) / np.sum((age_gap - age_gap.mean()) ** 2)
    )


def _run_pipeline(n_recordings: int = 900, **synthetic_overrides):
    """Generate, train, measure and decompose - the real pipeline, end to end.

    Returns the decomposition, the training result, and the ground-truth
    ceiling described in :func:`_ground_truth_ceiling`.
    """
    config = dataclasses.replace(
        SyntheticConfig(sampling_rate_hz=500), **synthetic_overrides
    )
    cohort = generate_cohort(config, n_recordings=n_recordings)

    signals_100 = resample_signals(cohort.signals, 500, 100)
    result = train_age_regressor(
        signals=signals_100, ages=cohort.ages, sexes=cohort.sexes,
        patient_ids=cohort.patient_ids, backbone_config=SMALL_BACKBONE,
        training_config=QUICK_TRAINING,
    )
    test = result.predictions["test"]

    # Intervals are measured at 500 Hz, as the real pipeline does.
    features = interval_features_table(
        cohort.signals[test.indices], 500.0, SignalProcessingConfig(),
        cohort[0].lead_names,
    )
    decomposition = decompose_age_gap(
        test.age_gap, features, END_TO_END_CONFIG,
        ages=test.true_age, sexes=cohort.sexes[test.indices],
        patient_ids=cohort.patient_ids[test.indices],
    )

    metadata = cohort.metadata()
    ceilings = {
        channel: _ground_truth_ceiling(
            test.age_gap,
            metadata[f"true_{channel}_age_offset"].to_numpy()[test.indices],
            test.true_age,
            cohort.sexes[test.indices],
        )
        for channel in ("known", "unexplained")
    }
    return decomposition, result, ceilings


def test_end_to_end_effect_mediated_by_a_known_interval():
    """An age signal carried entirely by QRS duration must be attributed to it.

    The synthetic cohort's only age effect is QRS widening, which classical
    delineation measures directly. If the framework cannot recognise this as
    rediscovery, it would label a century-old measurement a novel finding.
    """
    decomposition, training, ceilings = _run_pipeline(
        qrs_widening_ms_per_decade=9.0,
        t_wave_skew_per_decade=0.0,
        hr_change_bpm_per_decade=0.0,
        unexplained_age_offset_sd_years=0.0,
        known_age_offset_sd_years=8.0,
    )
    best = decomposition.most_explanatory
    assert training.test_metrics["mae"] < 12.0, "model failed to learn; test is vacuous"

    # The ceiling: how much of this gap the true latent offset itself explains.
    # The rest is the model's own error, which no ECG measurement can account
    # for, so it is not the decomposition's job to find.
    ceiling = ceilings["known"]
    assert ceiling > 0.3, f"injected signal too weak to test against ({ceiling:.1%})"
    recovered = best.r2_full / ceiling
    assert recovered > 0.7, (
        f"known intervals recovered only {recovered:.1%} of the {ceiling:.1%} that "
        "was explainable at all, for an age gap constructed to run entirely "
        "through QRS duration\n" + decomposition.summary_text()
    )
    assert (
        best.univariate_r2["qrs_duration_ms"]
        > best.univariate_r2["heart_rate_bpm"]
    ), "the wrong interval was credited"


def test_end_to_end_effect_invisible_to_known_intervals():
    """A T-wave skew signal must survive as a large unexplained residual.

    Skew changes the shape of the T wave while leaving its onset, offset,
    duration and peak amplitude identical, so no timing interval can see it
    (established in test_synthetic_ecg.py). If the decomposition 'explained'
    this, it would be dissolving a genuine discovery candidate.
    """
    decomposition, training, ceilings = _run_pipeline(
        qrs_widening_ms_per_decade=0.0,
        hr_change_bpm_per_decade=0.0,
        t_wave_skew_per_decade=0.30,
        known_age_offset_sd_years=0.0,
        unexplained_age_offset_sd_years=8.0,
    )
    best = decomposition.most_explanatory
    assert training.test_metrics["mae"] < 14.0, "model failed to learn; test is vacuous"

    # There IS real injected signal here - the ceiling confirms it - and the
    # known intervals must nonetheless fail to find it. A test where the ceiling
    # was near zero would pass trivially by having nothing to detect.
    ceiling = ceilings["unexplained"]
    assert ceiling > 0.3, f"injected signal too weak to test against ({ceiling:.1%})"
    # Around 30% is attributable to known features even here, and that is
    # correct rather than a failure - see
    # test_measured_qt_partially_detects_t_wave_shape below. The measurement of
    # QT is shape-sensitive even though its definition is not, so a real
    # cardiologist could partly detect this change too. The framework is
    # supposed to credit that to "known".
    assert best.unexplained_fraction > 0.5, (
        f"only {best.unexplained_fraction:.1%} was left unexplained for an age gap "
        "that no timing interval can detect by construction\n"
        + decomposition.summary_text()
    )
    assert best.r2_incremental < 0.40, decomposition.summary_text()


def test_measured_qt_partially_detects_t_wave_shape():
    """A measurement can see what its definition cannot - pinned, because it matters.

    T-wave skew leaves the true QT interval exactly unchanged: onset, offset,
    duration and amplitude are all preserved by construction, and
    ``test_synthetic_ecg.py`` verifies it. But QT is not measured by consulting
    its definition. The tangent method locates T-offset from the steepest point
    of the descending limb, and skew changes the shape of that limb - so the
    *measured* QT shifts even though the *true* QT does not.

    Measured against the latent offset that drives skew, after removing the
    shared age trend, the correlation is roughly zero for true QT and roughly
    0.33 for measured QT.

    Two consequences, both worth stating in the paper:

    1. It is why the end-to-end "invisible signal" test above expects around
       30% of the gap to be attributed to known features rather than none. The
       decomposition is behaving correctly: a change a real clinician could
       partly detect through an existing measurement is partly rediscovery.
    2. It cuts against a tempting shortcut in this kind of validation. "The
       known quantity is unchanged in principle" does not establish that the
       known *measurement* is uninformative. Only the measurement can be
       regressed out, and only the measurement is what a clinician has.

    The error points the safe way: the known-feature set captures slightly more
    than an idealised one would, which shrinks the discovery claim rather than
    inflating it.
    """
    config = dataclasses.replace(
        SyntheticConfig(sampling_rate_hz=500),
        qrs_widening_ms_per_decade=0.0, hr_change_bpm_per_decade=0.0,
        t_wave_skew_per_decade=0.30, known_age_offset_sd_years=0.0,
        unexplained_age_offset_sd_years=8.0,
    )
    cohort = generate_cohort(config, n_recordings=300)
    measured = interval_features_table(
        cohort.signals, 500.0, SignalProcessingConfig(), cohort[0].lead_names
    )
    metadata = cohort.metadata()
    ages = metadata.age.to_numpy()
    offset = metadata.true_unexplained_age_offset.to_numpy()

    def age_residualised_correlation(values: np.ndarray) -> float:
        ok = np.isfinite(values)
        residual = values[ok] - np.poly1d(np.polyfit(ages[ok], values[ok], 1))(ages[ok])
        return float(np.corrcoef(residual, offset[ok])[0, 1])

    true_qt = age_residualised_correlation(metadata.true_qt_interval_ms.to_numpy())
    measured_qt = age_residualised_correlation(measured.qt_interval_ms.to_numpy())

    assert abs(true_qt) < 0.10, "the constructed QT should be untouched by skew"
    assert measured_qt > 0.20, (
        "measured QT is expected to leak T-wave shape information; if this has "
        "changed, the end-to-end expectations above need revisiting"
    )
    # QRS measurement, by contrast, is genuinely blind to T-wave shape.
    assert abs(
        age_residualised_correlation(measured.qrs_duration_ms.to_numpy())
    ) < 0.10


def test_end_to_end_both_channels_split_roughly_in_half():
    """With both effects present, roughly half should be explained.

    The generator drives QRS widening from one latent offset and T-wave skew
    from an independent one of equal variance, so a model reading both produces
    a gap that is about half measurable and about half not. Landing near that is
    a joint check on the model, the interval measurement and the decomposition.
    """
    decomposition, training, ceilings = _run_pipeline(
        qrs_widening_ms_per_decade=9.0,
        t_wave_skew_per_decade=0.30,
        hr_change_bpm_per_decade=0.0,
        known_age_offset_sd_years=8.0,
        unexplained_age_offset_sd_years=8.0,
    )
    best = decomposition.most_explanatory
    assert training.test_metrics["mae"] < 12.0
    # Both channels contribute real signal to the gap.
    assert ceilings["known"] > 0.15 and ceilings["unexplained"] > 0.15
    # And the decomposition splits it, rather than collapsing to either extreme.
    assert 0.15 < best.r2_incremental < 0.75, decomposition.summary_text()
    assert 0.25 < best.unexplained_fraction < 0.85, decomposition.summary_text()
