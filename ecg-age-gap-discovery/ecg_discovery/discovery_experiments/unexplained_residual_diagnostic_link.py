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
extra column. The interval is computed across folds, so it reflects the
variability that actually matters - how much the answer moves when the data
moves - rather than the variability of a single split's point estimate.

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
        Mean held-out AUC of the augmented classifier minus the baseline, over
        all folds. Positive means the residual helped.
    delta_auc_ci:
        Interval for that difference, from the spread across folds.
    improves:
        Whether the interval excludes zero *after* correcting for having tested
        several superclasses.
    n_positive:
        Recordings carrying this superclass. Small counts make an AUC unstable,
        which is reported rather than hidden.
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


@dataclass
class DiagnosticLinkReport:
    """The full experiment, every superclass reported."""

    links: dict[str, SuperclassLink]
    skipped: dict[str, str]
    n_recordings: int
    correction: str
    minimum_meaningful_delta: float

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
                "improves": link.improves,
            }
            for link in self.links.values()
        ])

    def summary_text(self) -> str:
        """State the outcome, in the same format whichever way it went."""
        lines = [
            f"Does the unexplained age-gap residual predict diagnosis beyond known "
            f"intervals?  ({self.n_recordings} recordings, out-of-fold AUC, "
            f"{self.correction} correction across {len(self.links)} superclasses)",
            "",
            f"{'superclass':<12}{'n+':>7}{'baseline':>11}{'augmented':>11}"
            f"{'delta':>9}{'95% CI':>18}",
        ]
        for link in self.links.values():
            low, high = link.delta_auc_ci
            lines.append(
                f"{link.superclass:<12}{link.n_positive:>7}{link.auc_baseline:>11.3f}"
                f"{link.auc_augmented:>11.3f}{link.delta_auc:>+9.3f}"
                f"{f'[{low:+.3f}, {high:+.3f}]':>18}"
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

        baseline_scores: list[float] = []
        augmented_scores: list[float] = []
        deltas: list[float] = []
        for train_index, test_index in folds:
            if len(np.unique(target[test_index])) < 2:
                continue      # a fold with one class has an undefined AUC
            fold_scores = []
            for matrix in (baseline_matrix, augmented_matrix):
                classifier = _build_classifier(config)
                classifier.fit(matrix[train_index], target[train_index])
                probability = classifier.predict_proba(matrix[test_index])[:, 1]
                fold_scores.append(roc_auc_score(target[test_index], probability))
            baseline_scores.append(fold_scores[0])
            augmented_scores.append(fold_scores[1])
            deltas.append(fold_scores[1] - fold_scores[0])

        if len(deltas) < 2:
            skipped[name] = "too few usable folds to estimate an interval"
            continue

        delta_array = np.asarray(deltas)
        low = float(np.percentile(delta_array, 100 * tail))
        high = float(np.percentile(delta_array, 100 * (1.0 - tail)))
        links[name] = SuperclassLink(
            superclass=name,
            n_positive=n_positive,
            n_total=n_recordings,
            auc_baseline=float(np.mean(baseline_scores)),
            auc_augmented=float(np.mean(augmented_scores)),
            delta_auc=float(delta_array.mean()),
            delta_auc_ci=(low, high),
            improves=bool(low > 0.0),
            fold_deltas=delta_array,
        )

    return DiagnosticLinkReport(
        links=links,
        skipped=skipped,
        n_recordings=n_recordings,
        correction=f"Bonferroni x{n_tests}",
        minimum_meaningful_delta=minimum_meaningful_delta,
    )
