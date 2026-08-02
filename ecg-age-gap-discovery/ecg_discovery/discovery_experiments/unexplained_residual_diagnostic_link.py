"""Does the unexplained residual predict anything a cardiologist recognises?

THE QUESTION
------------
Phase 8 established a ceiling: some share of the model's age gap cannot be
explained by classical ECG intervals. That share is a *candidate* for new
information, and nothing more. An unexplained residual is equally consistent
with the model having found something real and with the model having noise the
interval set happens not to predict.

This module applies the obvious test. If the unexplained residual carries real
cardiac information, it should help predict something independently established
- here, the diagnostic superclasses that cardiologists assigned to these
recordings without reference to any model. If it adds nothing beyond what the
known intervals already provide, that is a negative result, and a useful one.

THE COMPARISON
--------------
Two classifiers per diagnostic superclass, differing by exactly one feature::

    baseline:   demographics + known intervals
    augmented:  demographics + known intervals + unexplained residual

Both are cross-validated on identical folds, so the difference in their
held-out AUC is a paired quantity: the same recordings, the same splits, one
extra column. Each classifier's out-of-fold probability is retained per
recording, and the two AUCs are computed once over the pooled out-of-fold
predictions.

THE INTERVAL IS A PAIRED BOOTSTRAP OVER RECORDINGS, NOT A SPREAD OVER FOLDS
--------------------------------------------------------------------------
An earlier version of this module took percentiles of the *per-fold* AUC
deltas and called the result a confidence interval. It is not one, for two
reasons, and both matter enough to record here so the mistake is not made
again.

First, with five folds and a Bonferroni-corrected tail of 0.5%, the 0.5th
percentile of five numbers is just their minimum - so ``improves`` silently
degraded into "every fold happened to come out positive", which is a sign test
with no stated error rate, not a 95% interval.

Second, and independent of how many folds there are, fold-level estimates are
not independent draws: their training sets overlap in roughly ``(k-2)/(k-1)``
of the data, so their spread estimates neither the standard error of the mean
nor the sampling variability of the statistic. Averaging correlated estimates
does not shrink like ``1/sqrt(k)``, and no percentile of them is a calibrated
interval for anything.

What is used instead is a **paired percentile bootstrap over recordings**. The
out-of-fold predictions are held fixed and the *recordings* are resampled with
replacement; both AUCs are recomputed on each resample and their difference
taken, so the pairing is preserved and the interval reflects the sampling
variability of the cohort. Where patient identifiers exist the resampling is
by **patient**, not recording - a cluster bootstrap - because two recordings
from one patient carry less independent information than two from different
patients, and resampling rows would understate the width.

The per-fold deltas are still retained and reported as a secondary robustness
line ("positive in k of n folds"), which is a genuinely useful thing to know.
It is simply no longer dressed up as an interval.

FOUR WAYS THIS COULD LIE, AND WHAT IS DONE ABOUT THEM
-----------------------------------------------------
**Testing five superclasses and reporting the best one.** With five tests at
the conventional threshold, a spurious "significant" result is likely by
chance. Every superclass is therefore always reported, and the headline verdict
applies a Bonferroni correction across them. Reporting the best of five without
correction is the single easiest way to manufacture a discovery.

**Optimistic evaluation.** Everything is out of fold, stratified so rare
superclasses appear in every fold, and grouped by patient where identifiers
exist so a repeat visit cannot leak across the split.

**Confusing "adds signal" with "is useful".** An improvement in AUC of 0.004
can be statistically distinguishable from zero given enough data and remain
clinically meaningless. Effect size is reported alongside the interval, and the
verdict requires both.

**The residual being a proxy for age.** The residual comes from a model already
adjusted for age and sex, and the baseline classifier contains those same
covariates, so an improvement cannot come from the residual smuggling in
demographics the baseline lacks.

REPORTING
---------
:meth:`DiagnosticLinkReport.summary_text` states whichever outcome occurred, in
the same format either way. A null result here is a genuine finding for this
project - it would say that interpretability plus explicit validation showed a
model's extra signal did *not* correspond to independently verifiable
information, which is a cautionary result about attribution-based biomarker
claims rather than a failed experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ecg_discovery.config import ValidationFrameworkConfig

__all__ = [
    "SuperclassLink",
    "DiagnosticLinkReport",
    "evaluate_diagnostic_link",
]


@dataclass(frozen=True)
class SuperclassLink:
    """Whether the unexplained residual improved prediction of one superclass.

    Attributes
    ----------
    delta_auc:
        Pooled out-of-fold AUC of the augmented classifier minus the baseline.
        Positive means the residual helped.
    delta_auc_ci:
        Paired percentile bootstrap interval for that difference, resampling
        patients where identifiers exist and recordings otherwise, with the
        tail already Bonferroni-corrected for the number of superclasses
        tested. See the module docstring for why this replaced the spread of
        per-fold deltas.
    improves:
        Whether the interval excludes zero *after* correcting for having tested
        several superclasses.
    n_positive:
        Recordings carrying this superclass. Small counts make an AUC unstable,
        which is reported rather than hidden.
    fold_deltas:
        Per-fold AUC differences. Reported as a robustness line - how many folds
        came out positive - and deliberately *not* used to form the interval.
    """

    superclass: str
    n_positive: int
    n_total: int
    auc_baseline: float
    auc_augmented: float
    delta_auc: float
    delta_auc_ci: tuple[float, float]
    improves: bool
    fold_deltas: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    @property
    def prevalence(self) -> float:
        """Fraction of recordings carrying this superclass."""
        return self.n_positive / self.n_total if self.n_total else float("nan")

    @property
    def n_folds_positive(self) -> int:
        """How many cross-validation folds showed a positive difference."""
        return int((self.fold_deltas > 0).sum())

    @property
    def n_folds(self) -> int:
        """How many folds produced a usable AUC for both classifiers."""
        return int(self.fold_deltas.size)


@dataclass
class DiagnosticLinkReport:
    """The full experiment, every superclass reported."""

    links: dict[str, SuperclassLink]
    skipped: dict[str, str]
    n_recordings: int
    correction: str
    minimum_meaningful_delta: float
    #: Whether the bootstrap resampled patients rather than recordings. False
    #: means no patient identifiers were supplied, so repeated visits - if the
    #: cohort contains any - are treated as independent and the intervals are
    #: correspondingly optimistic.
    clustered_bootstrap: bool = False

    @property
    def any_improvement(self) -> bool:
        """Whether any superclass improved after multiplicity correction."""
        return any(link.improves for link in self.links.values())

    @property
    def best(self) -> SuperclassLink | None:
        """The largest improvement, whether or not it is distinguishable from zero."""
        return max(self.links.values(), key=lambda link: link.delta_auc, default=None)

    def to_frame(self) -> pd.DataFrame:
        """One row per superclass, for the run directory."""
        return pd.DataFrame([
            {
                "superclass": link.superclass,
                "n_positive": link.n_positive,
                "prevalence": link.prevalence,
                "auc_baseline": link.auc_baseline,
                "auc_augmented": link.auc_augmented,
                "delta_auc": link.delta_auc,
                "delta_ci_low": link.delta_auc_ci[0],
                "delta_ci_high": link.delta_auc_ci[1],
                "n_folds_positive": link.n_folds_positive,
                "n_folds": link.n_folds,
                "improves": link.improves,
            }
            for link in self.links.values()
        ])

    def summary_text(self) -> str:
        """State the outcome, in the same format whichever way it went."""
        lines = [
            f"Does the unexplained age-gap residual predict diagnosis beyond known "
            f"intervals?  ({self.n_recordings} recordings, pooled out-of-fold AUC, "
            f"{self.correction} correction across {len(self.links)} superclasses)",
            "Intervals are a paired bootstrap over "
            f"{'patients' if self.clustered_bootstrap else 'recordings'}; the "
            "'folds+' column is a separate robustness check, not the basis of the "
            "verdict.",
            "",
            f"{'superclass':<12}{'n+':>7}{'baseline':>11}{'augmented':>11}"
            f"{'delta':>9}{'95% CI':>18}{'folds+':>9}",
        ]
        for link in self.links.values():
            low, high = link.delta_auc_ci
            interval = (
                f"[{low:+.3f}, {high:+.3f}]" if np.isfinite(low) and np.isfinite(high)
                else "not estimable"
            )
            lines.append(
                f"{link.superclass:<12}{link.n_positive:>7}{link.auc_baseline:>11.3f}"
                f"{link.auc_augmented:>11.3f}{link.delta_auc:>+9.3f}"
                f"{interval:>18}"
                f"{f'{link.n_folds_positive}/{link.n_folds}':>9}"
                + ("  *" if link.improves else "")
            )
        for name, reason in self.skipped.items():
            lines.append(f"{name:<12}  skipped: {reason}")

        lines.append("")
        if self.any_improvement:
            improved = [link for link in self.links.values() if link.improves]
            largest = max(improved, key=lambda link: link.delta_auc)
            lines += [
                f"RESULT: the unexplained residual improves prediction of "
                f"{', '.join(link.superclass for link in improved)} beyond known "
                f"intervals (largest gain {largest.delta_auc:+.3f} AUC for "
                f"{largest.superclass}).",
                "",
                "This is a CANDIDATE signal for follow-up clinical validation, not a "
                "validated marker. It shows the residual carries information "
                "correlated with cardiologist labels in this dataset. It does not "
                "establish causation, clinical utility, or that the effect "
                "replicates in another cohort.",
            ]
            if largest.delta_auc < self.minimum_meaningful_delta:
                lines.append(
                    f"NOTE: the improvement is statistically distinguishable from "
                    f"zero but small ({largest.delta_auc:+.3f} AUC, below the "
                    f"{self.minimum_meaningful_delta:.3f} threshold set in advance "
                    "for a clinically meaningful effect). Statistical detectability "
                    "is not clinical relevance."
                )
        else:
            best = self.best
            detail = (
                f" the largest change was {best.delta_auc:+.3f} AUC for "
                f"{best.superclass}, whose interval includes zero."
                if best is not None else ""
            )
            lines += [
                "RESULT: no superclass is better predicted when the unexplained "
                "residual is added;" + detail,
                "",
                "This is a genuine negative result, not a failed experiment. The "
                "model's age gap contains variance that classical intervals do not "
                "explain, and that variance does not correspond to anything the "
                "cardiologist labels can verify. Read together with the "
                "attribution figures, it is a caution: attribution showing a model "
                "attending to a structure, plus an unexplained residual, is NOT "
                "evidence of a discovered biomarker.",
            ]
        return "\n".join(lines)


def _build_classifier(config: ValidationFrameworkConfig) -> Pipeline:
    """Logistic regression, standardised so coefficients are comparable.

    Deliberately simple. The question is whether one extra column carries
    information, and a flexible classifier would blur that by finding structure
    in the baseline features that the baseline classifier could not - changing
    what the comparison means.
    """
    steps = []
    if config.standardize_features:
        steps.append(("scale", StandardScaler()))
    steps.append((
        "model",
        LogisticRegression(max_iter=config.classifier_max_iter, random_state=config.seed),
    ))
    return Pipeline(steps)


def _paired_bootstrap_delta_auc(
    target: np.ndarray,
    baseline_probability: np.ndarray,
    augmented_probability: np.ndarray,
    groups: np.ndarray | None,
    config: ValidationFrameworkConfig,
    tail: float,
) -> tuple[float, float]:
    """Percentile interval for the difference between two paired AUCs.

    The out-of-fold predictions are treated as fixed and the *cohort* is
    resampled, which is what makes this an interval for the quantity actually
    claimed: how much the AUC difference would move on another sample of
    patients from the same population.

    Both AUCs are recomputed on the same resample, so the comparison stays
    paired and the large shared component of their variance cancels - a
    difference of two independently bootstrapped AUCs would be far wider than
    the truth and would hide a real effect.

    Resampling is by patient when identifiers are supplied. Two recordings from
    one patient are not two independent observations, and resampling rows in
    that situation produces an interval that is too narrow, which is the
    direction that manufactures a discovery.

    Parameters
    ----------
    tail:
        One-sided tail probability, already divided by the number of
        superclasses tested.

    Returns
    -------
    tuple[float, float]
        Lower and upper interval bounds. ``(nan, nan)`` if too few resamples
        contained both classes for a percentile to mean anything.
    """
    rng = np.random.default_rng(config.seed)

    # Row indices grouped by patient, precomputed once. Left as None in the
    # unclustered case rather than filled with a single all-rows "cluster",
    # which would look like a valid cluster list while standing for a bootstrap
    # that resamples nothing.
    clusters: list[np.ndarray] | None = None
    if groups is not None:
        unique_groups, inverse = np.unique(groups, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        boundaries = np.searchsorted(inverse[order], np.arange(unique_groups.size + 1))
        clusters = [
            order[boundaries[i] : boundaries[i + 1]] for i in range(unique_groups.size)
        ]

    scores = np.empty(config.bootstrap_iterations, dtype=np.float64)
    for iteration in range(config.bootstrap_iterations):
        if clusters is None:
            rows = rng.integers(0, target.size, target.size)
        else:
            drawn = rng.integers(0, len(clusters), len(clusters))
            rows = np.concatenate([clusters[unit] for unit in drawn])

        resampled_target = target[rows]
        # An AUC needs both classes present. A resample that happens to draw
        # only one is not an extreme value of the statistic, it is an absence of
        # one, so it is excluded rather than being scored as 0.5.
        if resampled_target.min() == resampled_target.max():
            scores[iteration] = np.nan
            continue
        scores[iteration] = (
            roc_auc_score(resampled_target, augmented_probability[rows])
            - roc_auc_score(resampled_target, baseline_probability[rows])
        )

    if np.count_nonzero(np.isfinite(scores)) < 0.5 * config.bootstrap_iterations:
        return (float("nan"), float("nan"))
    return (
        float(np.nanpercentile(scores, 100 * tail)),
        float(np.nanpercentile(scores, 100 * (1.0 - tail))),
    )


def evaluate_diagnostic_link(
    unexplained_residual: np.ndarray,
    features: pd.DataFrame,
    diagnostic_labels: np.ndarray,
    superclass_names: Sequence[str],
    config: ValidationFrameworkConfig | None = None,
    ages: np.ndarray | None = None,
    sexes: np.ndarray | None = None,
    patient_ids: np.ndarray | None = None,
    minimum_meaningful_delta: float = 0.02,
) -> DiagnosticLinkReport:
    """Test whether the unexplained residual improves diagnostic prediction.

    Parameters
    ----------
    unexplained_residual:
        Out-of-fold residual from the Phase 8 decomposition, one per recording.
    features:
        Interval measurements; ``config.known_features`` form the baseline.
    diagnostic_labels:
        ``(n_recordings, n_superclasses)`` multi-hot, cardiologist-assigned.
    superclass_names:
        Column names of ``diagnostic_labels``.
    ages, sexes:
        Included in **both** classifiers, so an improvement cannot come from the
        residual reintroducing demographics the baseline lacked.
    patient_ids:
        Groups cross-validation folds by patient when supplied.
    minimum_meaningful_delta:
        AUC gain considered clinically meaningful, fixed in advance. Used only
        for reporting, never to decide significance.

    Returns
    -------
    DiagnosticLinkReport
        Every superclass, with the verdict corrected for multiple testing.
    """
    config = config or ValidationFrameworkConfig()
    unexplained_residual = np.asarray(unexplained_residual, dtype=np.float64)
    diagnostic_labels = np.asarray(diagnostic_labels)

    if diagnostic_labels.shape[0] != unexplained_residual.size:
        raise ValueError(
            f"diagnostic_labels has {diagnostic_labels.shape[0]} rows but "
            f"unexplained_residual has {unexplained_residual.size} entries"
        )
    if diagnostic_labels.shape[1] != len(superclass_names):
        raise ValueError(
            f"diagnostic_labels has {diagnostic_labels.shape[1]} columns but "
            f"{len(superclass_names)} superclass names were supplied"
        )
    missing = [name for name in config.known_features if name not in features.columns]
    if missing:
        raise KeyError(f"known feature(s) {missing} are not in the supplied table")

    known = features[list(config.known_features)].to_numpy(dtype=np.float64)
    covariates = [
        np.asarray(array, dtype=np.float64)
        for array in (ages, sexes) if array is not None
    ]
    covariate_block = (
        np.column_stack(covariates) if covariates
        else np.zeros((unexplained_residual.size, 0))
    )

    usable = (
        np.isfinite(known).all(axis=1)
        & np.isfinite(unexplained_residual)
        & (np.isfinite(covariate_block).all(axis=1) if covariate_block.shape[1]
           else np.ones(unexplained_residual.size, dtype=bool))
    )
    baseline_matrix = np.column_stack([covariate_block, known])[usable]
    augmented_matrix = np.column_stack(
        [covariate_block, known, unexplained_residual[:, None]]
    )[usable]
    labels = diagnostic_labels[usable]
    groups = None if patient_ids is None else np.asarray(patient_ids)[usable]
    n_recordings = int(usable.sum())

    # Bonferroni across the superclasses actually tested. Applied to the
    # interval width rather than to a p-value, so the reported interval is the
    # one the verdict is based on.
    # A superclass needs enough positive cases in both directions for an AUC to
    # mean anything. Testing a class with a handful of positives produces an
    # estimate dominated by which fold they land in, and a spurious improvement
    # on a rare class is precisely the result that gets written up as a
    # discovery. Untestable classes are named in the report rather than skipped
    # silently.
    minimum = max(config.min_positives_for_link, config.cv_folds)
    testable = [
        index for index in range(labels.shape[1])
        if minimum <= labels[:, index].sum() <= n_recordings - minimum
    ]
    n_tests = max(len(testable), 1)
    tail = (1.0 - config.confidence_level) / n_tests / 2.0

    links: dict[str, SuperclassLink] = {}
    skipped: dict[str, str] = {}

    for index, name in enumerate(superclass_names):
        target = labels[:, index].astype(int)
        n_positive = int(target.sum())
        if index not in testable:
            skipped[name] = (
                f"{n_positive} positive of {n_recordings} - too few for a "
                f"meaningful AUC (need at least {minimum} in each class)"
            )
            continue

        if groups is not None:
            splitter = StratifiedGroupKFold(
                n_splits=config.cv_folds, shuffle=True, random_state=config.seed
            )
            folds = list(splitter.split(baseline_matrix, target, groups))
            # One pass only: StratifiedGroupKFold has no repeat variant, so
            # repeats would need reshuffling by hand for little benefit.
        else:
            splitter = RepeatedStratifiedKFold(
                n_splits=config.cv_folds, n_repeats=config.cv_repeats,
                random_state=config.seed,
            )
            folds = list(splitter.split(baseline_matrix, target))

        # Out-of-fold probabilities are accumulated per recording rather than
        # scored fold by fold, so the AUCs and the bootstrap below are computed
        # over the whole cohort. With repeated splitting a recording is held out
        # more than once; those predictions are averaged.
        probability_sum = np.zeros((2, target.size), dtype=np.float64)
        probability_count = np.zeros(target.size, dtype=np.int64)
        deltas: list[float] = []

        for train_index, test_index in folds:
            if len(np.unique(target[test_index])) < 2:
                continue      # a fold with one class has an undefined AUC
            fold_scores = []
            for position, matrix in enumerate((baseline_matrix, augmented_matrix)):
                classifier = _build_classifier(config)
                classifier.fit(matrix[train_index], target[train_index])
                probability = classifier.predict_proba(matrix[test_index])[:, 1]
                probability_sum[position, test_index] += probability
                fold_scores.append(roc_auc_score(target[test_index], probability))
            probability_count[test_index] += 1
            deltas.append(fold_scores[1] - fold_scores[0])

        covered = probability_count > 0
        if len(deltas) < 2 or covered.sum() < minimum * 2:
            skipped[name] = "too few usable folds to estimate an interval"
            continue

        out_of_fold = probability_sum[:, covered] / probability_count[covered]
        covered_target = target[covered]
        if covered_target.min() == covered_target.max():
            skipped[name] = "only one class among the out-of-fold predictions"
            continue

        auc_baseline = float(roc_auc_score(covered_target, out_of_fold[0]))
        auc_augmented = float(roc_auc_score(covered_target, out_of_fold[1]))
        low, high = _paired_bootstrap_delta_auc(
            covered_target,
            out_of_fold[0],
            out_of_fold[1],
            None if groups is None else groups[covered],
            config,
            tail,
        )
        links[name] = SuperclassLink(
            superclass=name,
            n_positive=n_positive,
            n_total=n_recordings,
            auc_baseline=auc_baseline,
            auc_augmented=auc_augmented,
            delta_auc=auc_augmented - auc_baseline,
            delta_auc_ci=(low, high),
            # NaN bounds mean the interval could not be estimated, which must
            # never read as an improvement.
            improves=bool(np.isfinite(low) and low > 0.0),
            fold_deltas=np.asarray(deltas),
        )

    return DiagnosticLinkReport(
        links=links,
        skipped=skipped,
        n_recordings=n_recordings,
        correction=f"Bonferroni x{n_tests}",
        minimum_meaningful_delta=minimum_meaningful_delta,
        clustered_bootstrap=groups is not None,
    )
