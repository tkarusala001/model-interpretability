"""Separating rediscovery from discovery in a model's age-gap residual.

THE PROBLEM
-----------
A model predicts age from an ECG and is wrong by some amount. That error - the
age gap - is routinely interpreted as meaning the heart is biologically older or
younger than the patient. Attribution can then show *where in the signal* the
model looked. Neither step establishes that the model found anything new.

The gap could be nothing more than a repackaging of measurements cardiology has
taken for a century. A model that has quietly learned "wide QRS complexes mean
older" produces a perfectly good age gap, and attribution obligingly highlights
the QRS complex, and none of it is a discovery. Attribution answers *where the
model looked*; it cannot answer *whether what it saw was already known*.

THE METHOD
----------
Regress the age gap on everything already measurable, and keep only what is left
over::

    age gap  ~  demographics (age, sex)          -> baseline
    age gap  ~  demographics + known intervals   -> full model

    explained by known intervals = R2(full) - R2(baseline)
    unexplained                  = 1 - R2(full)

Only the unexplained part is a candidate for discovery, and even then only a
candidate - Phase 9 asks whether it predicts anything independently verifiable.

Three details make the difference between this being a real test and a
formality:

**Demographics are regressed out first.** An age gap is mechanically correlated
with age: models regress towards the mean, over-predicting the young and
under-predicting the old, so the gap carries an age trend that is an artefact of
regression rather than physiology. Known intervals also vary with age. Without
adjusting, that shared age dependence would be credited to the intervals and
inflate the "explained" share. The reported figure is therefore *incremental*
R-squared over demographics alone.

**Everything is evaluated out of fold.** In-sample R-squared rises with model
flexibility whether or not any relationship exists; a gradient-boosted model can
"explain" pure noise. Fits are cross-validated, grouped by patient where patient
identifiers are available, and the unexplained residual passed downstream is
built from out-of-fold predictions so it is not contaminated by its own fit.

**Both a linear and a non-linear explainer are always reported.** The linear
model is interpretable; the boosted model catches curved or interacting
relationships a linear fit would miss and leave sitting in the "unexplained"
residual, looking like a discovery. Reporting both removes the opportunity to
pick whichever number reads better, so the choice is made before seeing results.

WHICH WAY THE ERRORS POINT
--------------------------
Every weakness in this pipeline pushes the same direction: **towards claiming a
discovery.** Noisy interval measurement, too few known features, or an explainer
too rigid to fit a real relationship all shrink the explained share and inflate
the unexplained residual. There is no corresponding mechanism that manufactures
a *false negative*. So an unexplained residual is weak evidence and a fully
explained one is strong evidence - and the honest reading of a large unexplained
share is "we have not shown this is known", never "we have shown this is new".

This asymmetry is why ``docs/limitations.md`` states plainly that the
known-feature set is small, and why a richer one could only ever explain more.

A MEASUREMENT CAN SEE WHAT ITS DEFINITION CANNOT
------------------------------------------------
One finding from validating this framework deserves stating up front, because
it changes how the known-feature set should be understood. In the synthetic
cohort, T-wave *skew* leaves the true QT interval exactly unchanged - onset,
offset, duration and amplitude are all preserved by construction. Yet the
*measured* QT correlates about 0.33 with the skew signal, against roughly zero
for the true QT, because QT is measured by the tangent method, which reads the
shape of the T wave's descending limb.

So a morphology change invisible to the *definition* of a known interval can be
partly visible to its *measurement*. The framework consequently attributes some
of that signal to rediscovery - which is correct, since a clinician measuring
QT would partly detect the change too. What must not be done is the tempting
shortcut of arguing "the known quantity is unchanged in principle, therefore
this is new": only the measurement can be regressed out, and only the
measurement is what a clinician actually has.

GENERALITY
----------
Nothing here is specific to ECG. The template - *regress the novel signal
against everything already known; only the residual is a discovery candidate,
and evaluate it out of fold* - applies to any claim that a model has found
something new in a domain that already has measurements. See
``docs/validation_methodology.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ecg_discovery.config import ValidationFrameworkConfig

__all__ = [
    "ExplainerResult",
    "DecompositionResult",
    "assemble_covariates",
    "decompose_age_gap",
]


def _r2(true: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of variance explained, against always guessing the mean.

    Can be negative out of fold, which is meaningful rather than an error: it
    says the fitted relationship does not generalise and performs worse than a
    constant.
    """
    variance = float(np.var(true))
    if variance <= 0:
        return float("nan")
    return float(1.0 - np.mean((true - predicted) ** 2) / variance)


@dataclass(frozen=True)
class ExplainerResult:
    """How much of the age gap one explainer accounted for.

    Attributes
    ----------
    r2_baseline:
        Out-of-fold R-squared using demographics (age, sex) alone.
    r2_full:
        Out-of-fold R-squared using demographics plus the known intervals.
    r2_incremental:
        ``r2_full - r2_baseline`` - the share of the age gap the known
        intervals explain *beyond* what demographics already account for. This
        is the headline "rediscovery" number.
    r2_incremental_ci:
        Paired bootstrap interval for ``r2_incremental``, clustered by patient
        where identifiers are supplied. This is the interval that belongs on
        the headline number: an attributable share quoted as a bare point
        estimate invites comparison between feature sets whose difference may
        be entirely sampling noise.
    unexplained_fraction:
        ``1 - r2_full``. The share left over, and the ceiling on any discovery
        claim.
    unexplained_residual:
        Out-of-fold residual of the full model, per recording. What Phase 9
        tests for independent signal.
    feature_effects:
        Per-feature summary - standardised coefficients for the linear
        explainer, impurity importances for the boosted one. Indicative only;
        correlated intervals share credit arbitrarily.
    univariate_r2:
        Out-of-fold R-squared of each known feature on its own, above
        demographics. Useful for seeing which single measurement carries the
        signal, and robust to the credit-sharing problem above.
    """

    model_name: str
    r2_baseline: float
    r2_full: float
    r2_incremental: float
    unexplained_fraction: float
    r2_full_ci: tuple[float, float]
    r2_incremental_ci: tuple[float, float]
    unexplained_residual: np.ndarray = field(repr=False)
    feature_effects: dict[str, float] = field(default_factory=dict)
    univariate_r2: dict[str, float] = field(default_factory=dict)


@dataclass
class DecompositionResult:
    """The full decomposition, reported for every explainer without selection."""

    known_features: tuple[str, ...]
    covariates: tuple[str, ...]
    n_recordings: int
    n_dropped: int
    age_gap_sd: float
    explainers: dict[str, ExplainerResult]

    @property
    def most_explanatory(self) -> ExplainerResult:
        """The explainer that accounted for most of the gap.

        Used for the headline "unexplained" figure, because the *smallest*
        credible unexplained residual is the conservative claim: if any
        reasonable model of the known features explains the signal, it is not a
        discovery.
        """
        return max(self.explainers.values(), key=lambda result: result.r2_full)

    def summary_text(self) -> str:
        """A plain-language report of the decomposition, both explainers shown."""
        lines = [
            f"Age-gap residual decomposition over {self.n_recordings} recordings "
            f"(SD {self.age_gap_sd:.2f} years"
            + (f"; {self.n_dropped} dropped for unmeasurable intervals)." if self.n_dropped
               else ")."),
            f"Known features: {', '.join(self.known_features)}",
            f"Adjusted for: {', '.join(self.covariates) or 'nothing'}",
            "",
            f"{'explainer':<20}{'demographics':>14}{'+ intervals':>14}"
            f"{'attributable':>14}{'unexplained':>14}{'attributable 95% CI':>22}",
        ]
        for name, result in self.explainers.items():
            low, high = result.r2_incremental_ci
            lines.append(
                f"{name:<20}{result.r2_baseline:>13.1%} {result.r2_full:>13.1%} "
                f"{result.r2_incremental:>13.1%} {result.unexplained_fraction:>13.1%}"
                f"{f'[{low:.1%}, {high:.1%}]':>22}"
            )
        best = self.most_explanatory
        low, high = best.r2_incremental_ci
        lines += [
            "",
            f"Headline: known ECG intervals explain {best.r2_incremental:.1%} "
            f"[{low:.1%}, {high:.1%}] of the "
            f"age-gap residual beyond demographics; {best.unexplained_fraction:.1%} "
            f"is unexplained ({best.model_name}, out-of-fold).",
            "The unexplained share is a CANDIDATE for new signal, not evidence of it. "
            "Every weakness in this pipeline - measurement noise, a small known-feature "
            "set, an under-powered explainer - inflates it.",
        ]
        if best.r2_incremental < 0:
            lines.append(
                "WARNING: the attributable share is negative, meaning the known "
                "intervals made out-of-fold prediction WORSE than demographics "
                "alone. Read it as zero contribution, not a negative one. This "
                "usually indicates too few recordings for the number of features; "
                "check n_recordings before interpreting anything here."
            )
        if self.n_recordings < 200:
            lines.append(
                f"WARNING: only {self.n_recordings} recordings. These estimates are "
                "unstable at this sample size and should not be reported as results."
            )
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """One row per explainer, for the run directory."""
        return pd.DataFrame([
            {
                "explainer": name,
                "r2_baseline": result.r2_baseline,
                "r2_full": result.r2_full,
                "r2_incremental": result.r2_incremental,
                "unexplained_fraction": result.unexplained_fraction,
                "r2_full_ci_low": result.r2_full_ci[0],
                "r2_full_ci_high": result.r2_full_ci[1],
                "r2_incremental_ci_low": result.r2_incremental_ci[0],
                "r2_incremental_ci_high": result.r2_incremental_ci[1],
            }
            for name, result in self.explainers.items()
        ])


def _build_estimator(name: str, config: ValidationFrameworkConfig):
    """Construct one explainer.

    The linear model is wrapped in a standardiser so its coefficients are
    directly comparable across features measured in different units - a
    millisecond of QRS duration and a beat per minute are not otherwise
    commensurable.
    """
    if name == "linear":
        return Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
    if name == "gradient_boosting":
        gb = config.gradient_boosting
        return GradientBoostingRegressor(
            n_estimators=gb.n_estimators,
            max_depth=gb.max_depth,
            learning_rate=gb.learning_rate,
            subsample=gb.subsample,
            random_state=config.seed,
            # Internal early stopping. Without it, several hundred trees fitted
            # to one or two features on a few hundred recordings overfit so
            # badly that out-of-fold performance falls *below* the mean
            # predictor - which would then corrupt the incremental figure this
            # framework reports. The boosted explainer exists to catch
            # non-linear structure, not to be given enough rope to memorise.
            n_iter_no_change=15,
            validation_fraction=0.15,
            tol=1e-4,
        )
    raise ValueError(f"unknown explainer {name!r}")


def _out_of_fold_predictions(
    features: np.ndarray,
    target: np.ndarray,
    estimator_name: str,
    config: ValidationFrameworkConfig,
    groups: np.ndarray | None,
) -> np.ndarray:
    """Cross-validated predictions, one per recording, never fitted on itself.

    Grouped by patient when identifiers are supplied: a patient with two
    recordings in the same set would otherwise let a flexible model recognise
    the individual across the fold boundary, which is the same leakage hazard
    the train/test split guards against, one level down.
    """
    n = len(target)
    n_splits = min(config.cv_folds, n)
    if groups is not None:
        n_groups = len(np.unique(groups))
        n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        raise ValueError(
            f"need at least 2 cross-validation folds, but only {n_splits} are "
            f"possible with {n} recordings"
        )

    splitter = (
        GroupKFold(n_splits=n_splits) if groups is not None
        else KFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
    )
    predictions = np.zeros(n, dtype=np.float64)
    for train_index, test_index in splitter.split(features, target, groups):
        estimator = _build_estimator(estimator_name, config)
        estimator.fit(features[train_index], target[train_index])
        predictions[test_index] = estimator.predict(features[test_index])
    return predictions


def _cluster_indices(groups: np.ndarray | None) -> list[np.ndarray] | None:
    """Row indices grouped by patient, for a cluster bootstrap.

    Left as ``None`` in the unclustered case rather than filled with a single
    all-rows "cluster", which would look like a valid cluster list while
    standing for a bootstrap that resamples nothing.
    """
    if groups is None:
        return None
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.searchsorted(inverse[order], np.arange(unique_groups.size + 1))
    return [order[boundaries[i] : boundaries[i + 1]] for i in range(unique_groups.size)]


def _paired_bootstrap_r2(
    target: np.ndarray,
    baseline_prediction: np.ndarray,
    full_prediction: np.ndarray,
    groups: np.ndarray | None,
    config: ValidationFrameworkConfig,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Intervals for the full R-squared and for the incremental share.

    Both are read off the *same* resamples, which is what keeps the incremental
    interval paired: baseline and full R-squared share a large component of
    their sampling variance, and differencing two independently bootstrapped
    intervals would be far wider than the truth. Here the shared component
    cancels, as it does in the quantity being estimated.

    The out-of-fold predictions are treated as fixed and the *cohort* is
    resampled, so the interval answers the question actually asked of it: how
    much would this figure move on another sample of patients from the same
    population.

    Resampling is by patient when identifiers are supplied. Two recordings from
    one patient are not two independent observations, and resampling rows in
    that situation yields an interval that is too narrow - the direction that
    makes a feature set look more decisively informative than the data support.

    The zero floor applied to the baseline point estimate is applied inside each
    resample too, so the interval is an interval for the statistic that is
    actually reported rather than for a slightly different one.
    """
    rng = np.random.default_rng(config.seed)
    clusters = _cluster_indices(groups)
    n = target.size

    full_scores = np.empty(config.bootstrap_iterations, dtype=np.float64)
    incremental_scores = np.empty(config.bootstrap_iterations, dtype=np.float64)
    for iteration in range(config.bootstrap_iterations):
        if clusters is None:
            rows = rng.integers(0, n, n)
        else:
            drawn = rng.integers(0, len(clusters), len(clusters))
            rows = np.concatenate([clusters[unit] for unit in drawn])
        resampled_target = target[rows]
        full = _r2(resampled_target, full_prediction[rows])
        baseline = max(_r2(resampled_target, baseline_prediction[rows]), 0.0)
        full_scores[iteration] = full
        incremental_scores[iteration] = full - baseline

    tail = (1.0 - config.confidence_level) / 2.0

    def interval(scores: np.ndarray) -> tuple[float, float]:
        return (
            float(np.nanpercentile(scores, 100 * tail)),
            float(np.nanpercentile(scores, 100 * (1.0 - tail))),
        )

    return interval(full_scores), interval(incremental_scores)


def assemble_covariates(
    config: ValidationFrameworkConfig,
    n_recordings: int,
    ages: np.ndarray | None,
    sexes: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    """The demographic design matrix, and the names of the columns in it.

    Shared with ``knowledge_accumulation`` so that a sweep over known-feature
    subsets adjusts for exactly what a single decomposition adjusts for. A
    curve built against a different baseline than the headline figure would not
    be a curve through that figure.
    """
    columns: list[np.ndarray] = []
    names: list[str] = []
    if "age" in config.adjust_for_covariates and ages is not None:
        columns.append(np.asarray(ages, dtype=np.float64))
        names.append("age")
    if "sex" in config.adjust_for_covariates and sexes is not None:
        columns.append(np.asarray(sexes, dtype=np.float64))
        names.append("sex")
    covariates = (
        np.column_stack(columns) if columns else np.zeros((n_recordings, 0))
    )
    return covariates, names


def decompose_age_gap(
    age_gap: np.ndarray,
    features: pd.DataFrame,
    config: ValidationFrameworkConfig | None = None,
    ages: np.ndarray | None = None,
    sexes: np.ndarray | None = None,
    patient_ids: np.ndarray | None = None,
    compute_univariate: bool = True,
) -> DecompositionResult:
    """Decompose an age-gap residual into explained and unexplained parts.

    Parameters
    ----------
    age_gap:
        Per-recording ``predicted - true`` age, in years.
    features:
        Interval measurements, one row per recording aligned with ``age_gap``.
        Must contain every name in ``config.known_features``.
    config:
        Which features count as known, which explainers to fit, and the
        cross-validation and bootstrap settings.
    ages, sexes:
        Demographic covariates to adjust for. Strongly recommended: an age gap
        is mechanically correlated with age, and leaving that in would credit
        the intervals with variance that is a regression artefact.
    patient_ids:
        Used to group cross-validation folds by patient, and to cluster the
        bootstrap.
    compute_univariate:
        Fit each known feature on its own as well. Informative for a single
        decomposition, but it costs one extra cross-validation per feature per
        explainer, so ``knowledge_accumulation`` turns it off when sweeping
        dozens of feature subsets whose individual breakdowns are not used.

    Returns
    -------
    DecompositionResult
        Results for every configured explainer, none omitted.

    Notes
    -----
    Recordings with any unmeasurable known feature are dropped, and the count is
    reported. They are not imputed: a fabricated interval would enter the
    known-feature set as though it were a measurement and distort the very
    comparison this function exists to make. Note that dropping is not neutral
    either - recordings with unmeasurable intervals are plausibly the noisiest
    ones - so the count belongs in any write-up.
    """
    config = config or ValidationFrameworkConfig()
    age_gap = np.asarray(age_gap, dtype=np.float64)

    missing = [name for name in config.known_features if name not in features.columns]
    if missing:
        raise KeyError(
            f"known feature(s) {missing} are not in the supplied table; available: "
            f"{list(features.columns)}"
        )
    if len(features) != age_gap.size:
        raise ValueError(
            f"features has {len(features)} rows but age_gap has {age_gap.size} entries"
        )

    known = features[list(config.known_features)].to_numpy(dtype=np.float64)

    covariates, covariate_names = assemble_covariates(
        config, age_gap.size, ages, sexes
    )

    usable = np.isfinite(known).all(axis=1) & np.isfinite(age_gap)
    if covariates.shape[1]:
        usable &= np.isfinite(covariates).all(axis=1)
    n_dropped = int((~usable).sum())
    if usable.sum() < config.cv_folds * 2:
        raise ValueError(
            f"only {int(usable.sum())} recordings have all known features "
            f"measurable, too few for {config.cv_folds}-fold cross-validation"
        )

    known = known[usable]
    covariates = covariates[usable]
    target = age_gap[usable]
    groups = None if patient_ids is None else np.asarray(patient_ids)[usable]

    # A baseline with no covariates is the mean, so out-of-fold R-squared is ~0
    # by construction; that is the correct reference in that case.
    baseline_features = (
        covariates if covariates.shape[1] else np.zeros((target.size, 1))
    )
    full_features = np.column_stack([covariates, known])

    explainers: dict[str, ExplainerResult] = {}
    for name in config.explainer_models:
        baseline_prediction = _out_of_fold_predictions(
            baseline_features, target, name, config, groups
        )
        full_prediction = _out_of_fold_predictions(
            full_features, target, name, config, groups
        )
        # A baseline that scores below zero out of fold has failed to beat the
        # mean predictor, which means demographics explain *nothing* - not a
        # negative amount. The mean is always available as a fallback, so the
        # floor is zero. Without this, an overfitting baseline would subtract a
        # negative number and inflate the incremental figure, which is exactly
        # the direction that manufactures a rediscovery claim.
        r2_baseline = max(_r2(target, baseline_prediction), 0.0)
        r2_full = _r2(target, full_prediction)

        # Per-feature summaries, fitted on all usable rows purely for reporting.
        estimator = _build_estimator(name, config)
        estimator.fit(full_features, target)
        n_covariates = covariates.shape[1]
        if name == "linear":
            weights = estimator.named_steps["model"].coef_[n_covariates:]
        else:
            weights = estimator.feature_importances_[n_covariates:]
        feature_effects = {
            feature: float(weight)
            for feature, weight in zip(config.known_features, weights)
        }

        univariate: dict[str, float] = {}
        if compute_univariate:
            for index, feature in enumerate(config.known_features):
                single = np.column_stack([covariates, known[:, index : index + 1]])
                prediction = _out_of_fold_predictions(
                    single, target, name, config, groups
                )
                univariate[feature] = _r2(target, prediction) - r2_baseline

        full_ci, incremental_ci = _paired_bootstrap_r2(
            target, baseline_prediction, full_prediction, groups, config
        )
        explainers[name] = ExplainerResult(
            model_name=name,
            r2_baseline=r2_baseline,
            r2_full=r2_full,
            r2_incremental=r2_full - r2_baseline,
            unexplained_fraction=1.0 - r2_full,
            r2_full_ci=full_ci,
            r2_incremental_ci=incremental_ci,
            unexplained_residual=target - full_prediction,
            feature_effects=feature_effects,
            univariate_r2=univariate,
        )

    return DecompositionResult(
        known_features=tuple(config.known_features),
        covariates=tuple(covariate_names),
        n_recordings=int(usable.sum()),
        n_dropped=n_dropped,
        age_gap_sd=float(np.std(target)),
        explainers=explainers,
    )
