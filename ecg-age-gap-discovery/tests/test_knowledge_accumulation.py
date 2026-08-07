"""Tests for the knowledge-accumulation curve.

This module exists to answer a question the two-point comparison in the paper
cannot: *how much* of the unexplained residual is an artefact of a short
known-feature list. It must therefore be able to return both answers - a curve
still climbing when enumeration is unfinished, and a flat one when it is not -
because a diagnostic that always reports "incomplete" would block every
discovery claim and one that always reports "saturated" would wave them all
through.

Two things about the estimand shape the tests and are worth stating before
reading them.

**Random subsets are not the same as chosen subsets.** ``delta(k)`` is the
expected attributable share of a *randomly chosen* vocabulary of size k, so the
chance of containing the informative features rises with k. The mean therefore
climbs with k even when only two features matter, and flattening shows up only
at the top end. Saturation is consequently tested as a *contrast* between two
cohorts rather than against an absolute threshold.

**The row set must be fixed.** Every subset is scored on the recordings where
all features are measurable. Letting each subset drop its own unmeasurable rows
would score small subsets on more - and systematically cleaner - recordings
than large ones, bending the curve for a reason that has nothing to do with
knowledge accumulation. That confound has its own test.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ecg_discovery.config import ValidationFrameworkConfig
from ecg_discovery.validation.knowledge_accumulation import (
    AccumulationCurve,
    AccumulationLevel,
    _default_subset_sizes,
    _draw_subsets,
    _fit_saturating_ceiling,
    sweep_known_features,
)
from ecg_discovery.validation.residual_decomposition import decompose_age_gap

FEATURES = tuple(f"f{i}" for i in range(8))

CONFIG = ValidationFrameworkConfig(
    known_features=FEATURES,
    explainer_models=("linear",),
    cv_folds=4,
    bootstrap_iterations=200,
    seed=0,
)


def _cohort(n=500, n_informative=8, noise=1.0, seed=0):
    """Features where exactly ``n_informative`` of them drive the age gap."""
    rng = np.random.default_rng(seed)
    columns = {name: rng.normal(size=n) for name in FEATURES}
    gap = np.zeros(n)
    for name in FEATURES[:n_informative]:
        gap = gap + columns[name]
    gap = gap + rng.normal(scale=noise, size=n)
    return gap, pd.DataFrame(columns)


# --------------------------------------------------------------------------- #
# Both directions
# --------------------------------------------------------------------------- #
def test_curve_climbs_when_every_feature_carries_signal():
    """Enumeration that is genuinely unfinished must look unfinished."""
    gap, features = _cohort(n_informative=8)
    curve = sweep_known_features(
        gap, features, CONFIG, subset_sizes=(1, 2, 4, 6, 8), draws_per_size=6
    )
    means = [level.mean_incremental for level in curve.levels]
    assert means[0] < means[-1]
    assert curve.log_slope > 0.05
    assert curve.tail_gain > 0
    assert not curve.is_saturated()
    assert "still rising" in curve.summary_text()


def test_curve_flattens_when_only_a_few_features_carry_signal():
    """A vocabulary that has been exhausted must look exhausted.

    Compared against the all-informative cohort rather than an absolute
    threshold: with random subsets the mean climbs with k in both cases, and it
    is the *marginal* value of the last measurement that separates them.
    """
    sparse_gap, features = _cohort(n_informative=2)
    dense_gap, _ = _cohort(n_informative=8)
    sizes, draws = (1, 2, 4, 6, 8), 6

    sparse = sweep_known_features(
        sparse_gap, features, CONFIG, subset_sizes=sizes, draws_per_size=draws
    )
    dense = sweep_known_features(
        dense_gap, features, CONFIG, subset_sizes=sizes, draws_per_size=draws
    )
    assert sparse.tail_gain < dense.tail_gain
    assert sparse.log_slope < dense.log_slope


def test_noise_features_accumulate_nothing():
    """Adding measurements that explain nothing must not look like progress."""
    rng = np.random.default_rng(3)
    gap = rng.normal(size=500)
    features = pd.DataFrame({name: rng.normal(size=500) for name in FEATURES})
    curve = sweep_known_features(
        gap, features, CONFIG, subset_sizes=(1, 2, 4, 8), draws_per_size=5
    )
    for level in curve.levels:
        assert level.mean_incremental < 0.05
    assert curve.is_saturated()
    assert "flattened" in curve.summary_text()
    assert "necessary condition" in curve.summary_text()


# --------------------------------------------------------------------------- #
# Confounds that would silently bend the curve
# --------------------------------------------------------------------------- #
def test_every_subset_is_scored_on_the_same_recordings():
    """The row set is fixed to complete cases before any subset is drawn.

    Otherwise a subset excluding the unmeasurable feature would be scored on
    twice as many - and cleaner - recordings than the full set, and the curve
    would bend for a reason unrelated to knowledge accumulation. Checked by
    requiring a named subset to reproduce ``decompose_age_gap`` run on the
    manually filtered rows, not merely by checking a count.
    """
    gap, features = _cohort(n=400, n_informative=8, seed=5)
    dirty = features.copy()
    unmeasurable = np.zeros(len(dirty), dtype=bool)
    unmeasurable[::2] = True
    dirty.loc[unmeasurable, "f7"] = np.nan

    subset = ("f0", "f1")
    curve = sweep_known_features(
        gap, dirty, CONFIG, subset_sizes=(2,), draws_per_size=1,
        named_subsets={"clean": subset},
    )
    assert curve.n_recordings == int((~unmeasurable).sum())
    assert curve.n_dropped == int(unmeasurable.sum())

    expected = decompose_age_gap(
        gap[~unmeasurable],
        dirty.loc[~unmeasurable, :].reset_index(drop=True),
        dataclasses.replace(CONFIG, known_features=subset),
        compute_univariate=False,
    ).most_explanatory.r2_incremental
    assert curve.named_points[0].r2_incremental == pytest.approx(expected, abs=1e-9)


def test_the_full_set_end_of_the_curve_matches_the_headline_figure():
    """The curve must pass through the number a single decomposition reports."""
    gap, features = _cohort(n_informative=6, seed=2)
    curve = sweep_known_features(
        gap, features, CONFIG, subset_sizes=(2, 8), draws_per_size=3
    )
    headline = decompose_age_gap(
        gap, features, CONFIG, compute_univariate=False
    ).most_explanatory.r2_incremental
    assert curve.full_set_incremental == pytest.approx(headline, abs=1e-9)


def test_demographics_are_adjusted_out_at_every_point_on_the_curve():
    """A gap that is a pure function of age must be attributable nowhere.

    The mirror of the single-decomposition guard: an age gap is mechanically
    correlated with age, and if that were credited to the features it would be
    credited at every k, manufacturing a curve out of a regression artefact.
    """
    rng = np.random.default_rng(7)
    n = 500
    ages = rng.uniform(20, 85, n)
    gap = 0.4 * (ages - ages.mean()) + rng.normal(scale=0.5, size=n)
    features = pd.DataFrame({
        name: 0.3 * ages + rng.normal(size=n) for name in FEATURES
    })
    curve = sweep_known_features(
        gap, features, CONFIG, ages=ages, sexes=np.zeros(n),
        subset_sizes=(1, 4, 8), draws_per_size=4,
    )
    for level in curve.levels:
        assert level.mean_incremental < 0.1, level.n_features


# --------------------------------------------------------------------------- #
# How the subsets are drawn
# --------------------------------------------------------------------------- #
def test_draws_are_capped_at_the_number_of_subsets_that_exist():
    """Eight draws of one feature from three cannot be eight distinct answers.

    Without the cap a level would evaluate the same subset repeatedly and its
    spread would read as agreement between independent draws.
    """
    rng = np.random.default_rng(0)
    assert len(_draw_subsets(("a", "b", "c"), 1, 8, rng)) == 3
    assert len(_draw_subsets(("a", "b", "c"), 3, 8, rng)) == 1
    assert len(_draw_subsets(("a", "b", "c"), 2, 8, rng)) == 3


def test_subsets_within_a_level_are_distinct():
    gap, features = _cohort()
    curve = sweep_known_features(
        gap, features, CONFIG, subset_sizes=(2, 3), draws_per_size=5
    )
    for level in curve.levels:
        drawn = [point.features for point in level.points]
        assert len(set(drawn)) == len(drawn)
        assert all(len(f) == level.n_features for f in drawn)


def test_the_full_feature_set_is_a_single_deterministic_draw():
    gap, features = _cohort()
    curve = sweep_known_features(
        gap, features, CONFIG, subset_sizes=(8,), draws_per_size=6
    )
    assert curve.levels[0].n_subsets == 1
    assert curve.levels[0].points[0].features == FEATURES


def test_the_draw_is_reproducible_and_the_seed_argument_moves_only_the_draw():
    """Redrawing the curve must not silently redraw the cross-validation."""
    gap, features = _cohort()
    first = sweep_known_features(
        gap, features, CONFIG, subset_sizes=(3,), draws_per_size=4
    )
    same = sweep_known_features(
        gap, features, CONFIG, subset_sizes=(3,), draws_per_size=4
    )
    other = sweep_known_features(
        gap, features, CONFIG, subset_sizes=(3,), draws_per_size=4, seed=99
    )
    assert [p.features for p in first.levels[0].points] == \
           [p.features for p in same.levels[0].points]
    assert [p.features for p in first.levels[0].points] != \
           [p.features for p in other.levels[0].points]
    # The full-set end does not depend on which subsets were drawn.
    assert first.full_set_incremental == pytest.approx(other.full_set_incremental)


def test_default_sizes_are_dense_at_the_bend_and_include_the_full_set():
    for n in (4, 8, 15, 19):
        sizes = _default_subset_sizes(n)
        assert sizes[0] == 1
        assert sizes[-1] == n
        assert list(sizes) == sorted(set(sizes))
        # Denser at the bottom, where the curve's shape is decided.
        assert sizes[1] - sizes[0] <= sizes[-1] - sizes[-2]


# --------------------------------------------------------------------------- #
# Named subsets
# --------------------------------------------------------------------------- #
def test_named_subsets_are_evaluated_and_labelled():
    """A published analysis must be placeable on the curve it should be read against."""
    gap, features = _cohort(n_informative=8)
    curve = sweep_known_features(
        gap, features, CONFIG, subset_sizes=(4,), draws_per_size=2,
        named_subsets={"timing_only": ("f0", "f1"), "everything": FEATURES},
    )
    labels = {point.label for point in curve.named_points}
    assert labels == {"timing_only", "everything"}
    frame = curve.points_frame()
    assert set(frame["label"]) >= {"timing_only", "everything"}
    everything = next(p for p in curve.named_points if p.label == "everything")
    assert everything.r2_incremental == pytest.approx(
        curve.full_set_incremental, abs=1e-9
    )


# --------------------------------------------------------------------------- #
# Reporting and guards
# --------------------------------------------------------------------------- #
def _curve(log_slope, tail_gain, levels=None, ceiling=None, tail_log_slope=None):
    """A curve object with the diagnostics set directly, for verdict logic."""
    levels = levels or (
        AccumulationLevel(1, 3, 0.10, 0.01, 0.09, 0.11),
        AccumulationLevel(2, 3, 0.12, 0.01, 0.11, 0.13),
    )
    return AccumulationCurve(
        levels=levels, named_points=(), available_features=FEATURES,
        covariates=("age",), n_recordings=500, n_dropped=0,
        full_set_incremental=0.12, full_set_incremental_ci=(0.10, 0.14),
        log_slope=log_slope, tail_gain=tail_gain,
        tail_log_slope=tail_gain if tail_log_slope is None else tail_log_slope,
        extrapolated_ceiling=ceiling,
    )


def test_is_saturated_uses_the_tolerance_it_was_given():
    """There is no principled tolerance, so it must be an argument, not a constant."""
    curve = _curve(log_slope=0.02, tail_gain=0.005)
    assert not curve.is_saturated(tolerance=0.001)
    assert curve.is_saturated(tolerance=0.05)


def test_saturation_is_judged_on_the_tail_not_the_whole_curve():
    """A saturating curve has a steep global slope by definition - it climbed.

    Testing the global slope would report "still rising" for every genuinely
    saturated curve, which is the one case the diagnostic exists to detect.
    Regression test for a synthetic sweep that flattened to +0.3% over its last
    step and was called unfinished on a global slope of +0.247.
    """
    saturated = _curve(log_slope=0.247, tail_gain=0.003, tail_log_slope=0.003)
    assert saturated.is_saturated()
    assert "flattened" in saturated.summary_text()


def test_saturation_requires_both_a_flat_last_step_and_a_flat_rate():
    """The raw step and its per-e-fold rate must agree.

    The raw step alone would call a curve flat merely because its top two sizes
    were adjacent; the rate alone exaggerates a narrow step at the top.
    """
    assert not _curve(0.1, tail_gain=0.08, tail_log_slope=0.001).is_saturated()
    assert not _curve(0.1, tail_gain=0.001, tail_log_slope=0.08).is_saturated()
    assert _curve(0.1, tail_gain=0.001, tail_log_slope=0.001).is_saturated()


def test_an_unmeasurable_tail_is_not_saturation():
    """One level gives no last step, which must not read as a flat one."""
    assert not _curve(0.0, tail_gain=float("nan"),
                      tail_log_slope=float("nan")).is_saturated()


def test_saturation_fraction_is_none_without_a_fitted_ceiling():
    assert _curve(0.001, 0.001).saturation_fraction is None
    assert _curve(0.001, 0.001, ceiling=0.24).saturation_fraction == pytest.approx(0.5)


def test_the_extrapolated_ceiling_is_refused_rather_than_invented():
    """A fabricated asymptote would be read as a measurement of what remains."""
    sizes = np.array([1.0, 2.0, 4.0])
    assert _fit_saturating_ceiling(sizes, np.array([0.1, 0.15, 0.2])) is None

    sizes = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    saturating = np.array([0.05, 0.08, 0.11, 0.12, 0.125])
    ceiling = _fit_saturating_ceiling(sizes, saturating)
    assert ceiling is not None
    assert ceiling >= saturating.max()


def test_the_ceiling_is_refused_when_nothing_was_explained():
    """A fit to a row of near-zeros has no content and must not be reported.

    Found by a smoke run whose model was worse than the mean predictor: every
    level came out at 0.0% and the fit still produced a ceiling, which the
    report then rendered as "0.0% (1% of it reached)".
    """
    sizes = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    nothing = np.array([0.0, 0.0001, -0.0002, 0.0003, 0.0])
    assert _fit_saturating_ceiling(sizes, nothing) is None


def test_a_ceiling_that_ran_to_the_parameter_bound_is_refused():
    """Hitting the bound means the fit found no asymptote, not a ceiling of 100%.

    The hyperbola cannot describe a curve that stays near zero before climbing,
    so the optimiser pushes the ceiling as high as it is allowed. Regression
    test for a synthetic sweep that plateaued at 61% and reported exactly 100%,
    which would tell a reader most of existing knowledge was unenumerated.
    """
    sizes = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 15.0])
    plateau = np.array([0.018, 0.123, 0.097, 0.320, 0.532, 0.609, 0.611])
    ceiling = _fit_saturating_ceiling(sizes, plateau)
    assert ceiling is None or ceiling < 0.999


def test_the_spread_warning_fires_only_when_feature_choice_dominates():
    """Which features were picked can matter more than how many, and that is reportable."""
    wide = (
        AccumulationLevel(2, 4, 0.20, 0.15, 0.02, 0.45),
        AccumulationLevel(8, 1, 0.50, 0.0, 0.50, 0.50),
    )
    narrow = (
        AccumulationLevel(2, 4, 0.20, 0.002, 0.199, 0.203),
        AccumulationLevel(8, 1, 0.50, 0.0, 0.50, 0.50),
    )
    assert _curve(0.1, 0.1, levels=wide).spread_warning() is not None
    assert _curve(0.1, 0.1, levels=narrow).spread_warning() is None
    assert "not a specification" in _curve(0.1, 0.1, levels=wide).summary_text()


def test_frames_carry_every_point_and_level():
    gap, features = _cohort()
    curve = sweep_known_features(
        gap, features, CONFIG, subset_sizes=(1, 2, 8), draws_per_size=3,
        named_subsets={"pair": ("f0", "f1")},
    )
    levels = curve.to_frame()
    assert list(levels["n_features"]) == [1, 2, 8]
    assert {"mean_incremental", "spread", "n_subsets"} <= set(levels.columns)

    points = curve.points_frame()
    assert len(points) == sum(level.n_subsets for level in curve.levels) + 1
    assert {"features", "r2_incremental", "label"} <= set(points.columns)


def test_summary_reports_the_interval_on_the_full_set():
    gap, features = _cohort()
    curve = sweep_known_features(
        gap, features, CONFIG, subset_sizes=(2, 8), draws_per_size=2
    )
    low, high = curve.full_set_incremental_ci
    assert low <= curve.full_set_incremental <= high
    assert "All 8 features" in curve.summary_text()


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def test_missing_known_feature_raises():
    gap, features = _cohort()
    with pytest.raises(KeyError, match="not in the supplied table"):
        sweep_known_features(gap, features.drop(columns=["f3"]), CONFIG)


def test_length_mismatch_raises():
    gap, features = _cohort()
    with pytest.raises(ValueError, match="rows but"):
        sweep_known_features(gap[:-5], features, CONFIG)


def test_named_subset_naming_an_absent_feature_raises():
    gap, features = _cohort()
    with pytest.raises(KeyError, match="absent feature"):
        sweep_known_features(
            gap, features, CONFIG, subset_sizes=(2,),
            named_subsets={"bad": ("f0", "not_measured")},
        )


def test_too_few_usable_recordings_raises():
    gap, features = _cohort(n=6)
    with pytest.raises(ValueError, match="too few"):
        sweep_known_features(gap, features, CONFIG, subset_sizes=(2,))


def test_subset_sizes_outside_the_pool_are_dropped_not_silently_clamped():
    gap, features = _cohort()
    curve = sweep_known_features(
        gap, features, CONFIG, subset_sizes=(2, 99), draws_per_size=2
    )
    assert [level.n_features for level in curve.levels] == [2]
    with pytest.raises(ValueError, match="no valid subset sizes"):
        sweep_known_features(gap, features, CONFIG, subset_sizes=(0, 99))
