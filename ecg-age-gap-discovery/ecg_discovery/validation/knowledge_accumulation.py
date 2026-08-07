"""How much of a "discovery" is an artefact of how much knowledge was enumerated.

THE PROBLEM THIS MEASURES
-------------------------
``residual_decomposition`` reports the share of an age gap attributable to
known measurements, and the share left unexplained. That second number is the
ceiling on any discovery claim - and it moves when the *known-feature set*
changes, with nothing about the model changing at all. On PTB-XL, going from
five timing intervals to fifteen measurements quadrupled the attributable share
and took the residual's incremental diagnostic value to essentially zero.

Reported as two points, that is an anecdote about a quantity known in advance to
be monotone: a longer list can only ever explain more. It supports the warning
"your known-feature set might be too short" but gives a reader no way to ask
*how much too short*, and no way to tell a residual that is genuinely resistant
to further enumeration from one that is merely under-enumerated.

THE MEASUREMENT
---------------
Sweep the size of the known-feature set. For each size K, draw random subsets of
the available known features, run the decomposition on each, and record the
attributable share. Plotting the mean against K gives an accumulation curve -
structurally the same object as a species-accumulation curve in ecology, and
read the same way::

    still rising at K_max   ->  enumeration is incomplete; the unexplained
                                residual is not yet admissible as a discovery
                                candidate, because the next feature would have
                                eaten some of it

    flat at K_max           ->  the attributable share has saturated against
                                this *kind* of measurement, and the residual
                                survives the strongest version of the test the
                                available feature vocabulary can mount

This converts the project's central caution into a decision rule with a number
attached, and it is what makes the known-feature set an explicit variable rather
than a fixed input.

WHAT THE CURVE CANNOT DO
------------------------
**Saturation is against the vocabulary, not against knowledge.** A flat curve
says that adding *more of these fifteen kinds of measurement* stops helping. It
says nothing about a sixteenth kind that nobody computed - P-wave dispersion is
not a random draw from the amplitude-and-timing pool. The curve bounds
incompleteness *within* the enumerated vocabulary and is silent outside it, so a
flat curve is a necessary condition for a discovery claim, never a sufficient
one.

**Features are not exchangeable.** A random subset of size five averages over
good and bad choices of five, so the mean at K=5 is not the figure a deliberate
five-feature analysis would report. That difference is itself informative - the
*spread* across draws at fixed K measures how much the attributable share
depends on which features an analyst happened to pick - so named subsets can be
evaluated alongside the random draws and placed on the same axes.

**The extrapolated ceiling is an extrapolation.** The saturating fit is a
two-parameter curve through a handful of noisy means. It is reported because a
reader deserves to know where the trend points, and labelled loudly because a
trend is not a measurement.

WHICH WAY THE ERRORS POINT
--------------------------
As everywhere else in this pipeline, the asymmetry runs toward claiming a
discovery. An under-powered explainer, a noisy measurement or too few
recordings all flatten the curve - the failure mode is a curve that looks
saturated when enumeration is in fact incomplete. So a *rising* curve is strong
evidence against a discovery claim, and a *flat* one is weak evidence for it.
See ``docs/limitations.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ecg_discovery.config import ValidationFrameworkConfig
from ecg_discovery.validation.residual_decomposition import (
    _out_of_fold_predictions,
    _paired_bootstrap_r2,
    _r2,
    assemble_covariates,
)

__all__ = [
    "SubsetPoint",
    "AccumulationLevel",
    "AccumulationCurve",
    "sweep_known_features",
]


@dataclass(frozen=True)
class SubsetPoint:
    """One known-feature subset, and what it explained.

    Attributes
    ----------
    n_features:
        Size of the subset. The x-axis of the curve.
    features:
        Which features, so a surprising point can be traced to its cause.
    label:
        ``None`` for a random draw; the caller's name for a named subset.
    r2_incremental:
        Attributable share above demographics, out of fold, taking the most
        explanatory of the configured explainers - the same conservative
        convention ``DecompositionResult.most_explanatory`` uses, so that
        points on this curve are comparable with the headline figure.
    """

    n_features: int
    features: tuple[str, ...]
    r2_incremental: float
    r2_full: float
    label: str | None = None


@dataclass(frozen=True)
class AccumulationLevel:
    """Every subset drawn at one size, summarised."""

    n_features: int
    n_subsets: int
    mean_incremental: float
    sd_incremental: float
    min_incremental: float
    max_incremental: float
    points: tuple[SubsetPoint, ...] = field(repr=False, default=())

    @property
    def spread(self) -> float:
        """How much the attributable share depends on *which* features are picked.

        At a fixed K this is pure analyst degrees of freedom: same model, same
        cohort, same number of known measurements, different answer.
        """
        return self.max_incremental - self.min_incremental


@dataclass
class AccumulationCurve:
    """The sweep, its saturation diagnostics, and the caveats they carry."""

    levels: tuple[AccumulationLevel, ...]
    named_points: tuple[SubsetPoint, ...]
    available_features: tuple[str, ...]
    covariates: tuple[str, ...]
    n_recordings: int
    n_dropped: int
    #: Attributable share using every available feature, with its paired
    #: bootstrap interval. The right-hand end of the curve, and the figure a
    #: single decomposition would report.
    full_set_incremental: float
    full_set_incremental_ci: tuple[float, float]
    #: Ordinary least squares slope of attributable share on ``ln K``, in
    #: attributable-share units per e-fold increase in the size of the known
    #: feature set. Descriptive: the points are not independent, so this
    #: carries no interval and should be read as a summary of the drawn curve
    #: rather than an estimate of a population quantity.
    log_slope: float
    #: Gain in mean attributable share across the last step of the sweep. The
    #: most direct read on whether enumeration has stopped paying.
    tail_gain: float
    #: ``tail_gain`` divided by the width of that step in log-K, so it is in the
    #: same units as ``log_slope`` and comparable between sweeps whose top step
    #: spans a different number of features.
    #:
    #: This, not ``log_slope``, is what decides saturation. A curve that
    #: saturates has a steep *global* slope almost by definition - it climbed to
    #: get there - so testing the global slope would report "still rising" for
    #: every genuinely saturated curve, which is the one case the diagnostic
    #: exists to detect. Observed on a synthetic sweep that flattened to +0.3%
    #: over its last step and was still called unfinished on a global slope of
    #: +0.247.
    tail_log_slope: float = float("nan")
    #: Asymptote of a saturating fit, or ``None`` if the fit did not converge
    #: or there were too few levels to attempt one. An extrapolation.
    extrapolated_ceiling: float | None = None

    @property
    def saturation_fraction(self) -> float | None:
        """Observed attributable share as a fraction of the extrapolated ceiling.

        Near 1.0 the vocabulary is close to exhausted; well below it, the
        enumeration is visibly unfinished. ``None`` when no ceiling was fitted.
        """
        if self.extrapolated_ceiling is None or self.extrapolated_ceiling <= 0:
            return None
        return self.full_set_incremental / self.extrapolated_ceiling

    def is_saturated(self, tolerance: float = 0.01) -> bool:
        """Whether the curve has flattened, at a tolerance the caller must choose.

        ``tolerance`` is in attributable-share units per e-fold of K: the
        largest slope that still counts as flat. There is no principled value
        for it, which is why it is an argument with a stated default rather
        than a constant buried in the code. The default of 0.01 means "one
        further percentage point of variance per e-fold" - deliberately strict,
        because the error asymmetry means a curve that merely *looks* flat is
        the failure this diagnostic exists to catch.

        Judged on the *tail*, never on the global slope - see
        ``tail_log_slope`` for why. Both the raw last step and its per-e-fold
        rate must be small: the raw step alone would call a curve flat merely
        because its top two sizes were close together, and the rate alone
        exaggerates a narrow step at the top of the sweep.
        """
        if not np.isfinite(self.tail_log_slope):
            return False
        return self.tail_log_slope <= tolerance and self.tail_gain <= tolerance

    def to_frame(self) -> pd.DataFrame:
        """One row per level, for the run directory and for plotting."""
        return pd.DataFrame([
            {
                "n_features": level.n_features,
                "n_subsets": level.n_subsets,
                "mean_incremental": level.mean_incremental,
                "sd_incremental": level.sd_incremental,
                "min_incremental": level.min_incremental,
                "max_incremental": level.max_incremental,
                "spread": level.spread,
            }
            for level in self.levels
        ])

    def points_frame(self) -> pd.DataFrame:
        """One row per subset evaluated, random draws and named subsets alike."""
        rows = [
            {
                "n_features": point.n_features,
                "label": point.label or "",
                "features": ";".join(point.features),
                "r2_incremental": point.r2_incremental,
                "r2_full": point.r2_full,
            }
            for level in self.levels
            for point in level.points
        ]
        rows += [
            {
                "n_features": point.n_features,
                "label": point.label or "",
                "features": ";".join(point.features),
                "r2_incremental": point.r2_incremental,
                "r2_full": point.r2_full,
            }
            for point in self.named_points
        ]
        return pd.DataFrame(rows)

    def summary_text(self) -> str:
        """A plain-language report, verdict included."""
        low, high = self.full_set_incremental_ci
        lines = [
            f"Knowledge-accumulation curve over {self.n_recordings} recordings "
            f"({len(self.available_features)} available known features"
            + (f"; {self.n_dropped} dropped for unmeasurable features)."
               if self.n_dropped else ")."),
            f"Adjusted for: {', '.join(self.covariates) or 'nothing'}",
            "Row set is fixed across every subset, so the curve is not confounded "
            "by smaller subsets being scored on cleaner recordings.",
            "",
            f"{'K':>4}{'subsets':>10}{'mean attributable':>20}{'spread over draws':>20}",
        ]
        for level in self.levels:
            lines.append(
                f"{level.n_features:>4}{level.n_subsets:>10}"
                f"{level.mean_incremental:>19.1%} "
                f"{f'{level.min_incremental:.1%} - {level.max_incremental:.1%}':>19}"
            )
        lines += [
            "",
            f"All {len(self.available_features)} features: "
            f"{self.full_set_incremental:.1%} [{low:.1%}, {high:.1%}]",
            f"Slope over the whole curve: {self.log_slope:+.3f} attributable "
            "share per e-fold increase in K (descriptive only)",
            f"Last step: {self.tail_gain:+.1%} "
            f"({self.tail_log_slope:+.3f} per e-fold) - this is what decides "
            "saturation",
        ]
        if self.extrapolated_ceiling is not None:
            fraction = self.saturation_fraction
            lines.append(
                f"Extrapolated ceiling: {self.extrapolated_ceiling:.1%} "
                f"({fraction:.0%} of it reached) - EXTRAPOLATION from a "
                "two-parameter fit to a handful of noisy means, not a measurement."
            )
        lines.append("")
        if self.is_saturated():
            lines += [
                "VERDICT: the curve has flattened. Adding more measurements OF THE "
                "SAME KIND stops paying, so the unexplained residual survives the "
                "strongest version of this test the available vocabulary can mount.",
                "This is a necessary condition for a discovery claim, NOT a "
                "sufficient one: saturation is against the enumerated vocabulary, "
                "and says nothing about a kind of measurement nobody computed.",
            ]
        else:
            lines += [
                "VERDICT: the curve is still rising at the largest feature set "
                "tested. Enumeration is incomplete, so the unexplained residual is "
                "NOT admissible as a discovery candidate - the next feature would "
                "have eaten some of it. Extend the known-feature set before "
                "reporting any residual as novel.",
            ]
        if self.spread_warning():
            lines.append(self.spread_warning())
        return "\n".join(lines)

    def spread_warning(self) -> str | None:
        """Flag feature sets whose size does not determine what they explain."""
        worst = max(self.levels, key=lambda level: level.spread, default=None)
        if worst is None or worst.n_subsets < 2 or worst.spread < 0.02:
            return None
        return (
            f"NOTE: at K={worst.n_features} the attributable share ranged "
            f"{worst.min_incremental:.1%} - {worst.max_incremental:.1%} depending "
            "purely on which features were drawn. A feature-set size does not "
            "determine what it explains, so 'we adjusted for N known measurements' "
            "is not a specification of how hard a discovery claim was tested."
        )


def _default_subset_sizes(n_available: int) -> tuple[int, ...]:
    """Sizes to sweep: dense where the curve bends, sparse where it flattens.

    A linear sweep wastes fits at the top, where consecutive sizes are nearly
    identical, and under-samples the bottom, where the curve is steepest and
    the shape is decided.
    """
    if n_available <= 3:
        return tuple(range(1, n_available + 1))
    candidates = {1, 2, 3, n_available}
    step = 1
    value = 3
    while value < n_available:
        value += max(1, round(value * 0.6))
        step += 1
        if value < n_available:
            candidates.add(value)
    return tuple(sorted(candidates))


def _draw_subsets(
    features: Sequence[str],
    size: int,
    n_draws: int,
    rng: np.random.Generator,
) -> list[tuple[str, ...]]:
    """Distinct random subsets of the given size.

    Capped at the number of distinct subsets that exist, so small K does not
    silently evaluate the same three features eight times and report the
    repetition as agreement.
    """
    n_available = len(features)
    possible = math.comb(n_available, size)
    wanted = min(n_draws, possible)
    seen: set[tuple[str, ...]] = set()
    # Bounded so an unlucky stream of duplicates cannot spin forever; drawing
    # fewer subsets than asked is reported through n_subsets rather than hidden.
    for _ in range(wanted * 50):
        if len(seen) >= wanted:
            break
        chosen = rng.choice(n_available, size=size, replace=False)
        seen.add(tuple(features[i] for i in sorted(chosen)))
    return sorted(seen)


def _fit_saturating_ceiling(
    sizes: np.ndarray, means: np.ndarray
) -> float | None:
    """Asymptote of ``a * K / (K + h)`` fitted to the level means.

    Returns ``None`` rather than a number whenever the fit is not trustworthy -
    too few levels, no convergence, or a degenerate asymptote - because a
    fabricated ceiling would be read as a measurement of how much knowledge
    remains unenumerated.
    """
    if sizes.size < 4:
        return None
    # A saturating fit to a row of near-zeros has no content: the asymptote is
    # determined by noise, and the reported "fraction of the ceiling reached"
    # becomes a ratio of two numbers that are both nothing. Observed on a smoke
    # run whose model was worse than the mean predictor, where every level came
    # out at 0.0% and the fit still produced a confident-looking ceiling.
    if float(means.max()) < 0.005:
        return None
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None

    def saturating(k, ceiling, half):
        return ceiling * k / (k + half)

    try:
        params, _ = curve_fit(
            saturating,
            sizes.astype(np.float64),
            means.astype(np.float64),
            p0=[max(float(means.max()), 1e-3), 1.0],
            bounds=([0.0, 1e-6], [1.0, 1e4]),
            maxfev=10000,
        )
    except (RuntimeError, ValueError):
        return None
    ceiling = float(params[0])
    if not np.isfinite(ceiling) or ceiling <= 0:
        return None
    # A fit that ran to its upper bound has not found an asymptote, it has run
    # out of room: the hyperbola cannot describe a curve that stays flat near
    # zero before climbing, so the optimiser pushes the ceiling as high as it
    # is allowed. Seen on a synthetic sweep that plateaued at 61% and reported
    # a ceiling of exactly 100%. Reporting that would tell a reader most of
    # existing knowledge was still unenumerated, which the data do not say.
    if ceiling >= 0.999:
        return None
    # An asymptote below what was already observed is the fit failing to
    # describe its own data, not a ceiling.
    if ceiling < float(means.max()):
        return None
    return ceiling


def sweep_known_features(
    age_gap: np.ndarray,
    features: pd.DataFrame,
    config: ValidationFrameworkConfig | None = None,
    ages: np.ndarray | None = None,
    sexes: np.ndarray | None = None,
    patient_ids: np.ndarray | None = None,
    subset_sizes: Sequence[int] | None = None,
    draws_per_size: int = 8,
    named_subsets: Mapping[str, Sequence[str]] | None = None,
    seed: int | None = None,
) -> AccumulationCurve:
    """Measure how the attributable share grows with the known-feature set.

    Parameters
    ----------
    age_gap, features, ages, sexes, patient_ids:
        As for :func:`decompose_age_gap`. ``config.known_features`` defines the
        pool that subsets are drawn from.
    subset_sizes:
        Sizes of K to evaluate. Defaults to a geometric-ish sweep that is dense
        where the curve bends and always includes the full set.
    draws_per_size:
        Random subsets per size, capped at the number that exist.
    named_subsets:
        Specific feature sets to evaluate and report alongside the draws - for
        placing a published analysis on the curve it should have been read
        against. Names appear in ``points_frame``.
    seed:
        Overrides ``config.seed`` for the subset draws only, so a curve can be
        redrawn without changing the cross-validation or bootstrap.

    Returns
    -------
    AccumulationCurve

    Notes
    -----
    **Every subset is scored on the same recordings.** The row set is fixed once
    to the recordings where *all* available features are measurable, before any
    subset is drawn. Letting each subset drop its own unmeasurable rows would
    score small subsets on more - and systematically cleaner - recordings than
    large ones, which bends the curve for a reason that has nothing to do with
    knowledge accumulation.

    **The demographic baseline is fitted once.** It does not depend on which
    known features are in the subset, and refitting it per subset would add
    cross-validation noise to every point on the curve while changing no
    estimate.
    """
    config = config or ValidationFrameworkConfig()
    age_gap = np.asarray(age_gap, dtype=np.float64)

    available = tuple(config.known_features)
    missing = [name for name in available if name not in features.columns]
    if missing:
        raise KeyError(
            f"known feature(s) {missing} are not in the supplied table; available: "
            f"{list(features.columns)}"
        )
    if len(features) != age_gap.size:
        raise ValueError(
            f"features has {len(features)} rows but age_gap has {age_gap.size} entries"
        )
    named_subsets = dict(named_subsets or {})
    for name, subset in named_subsets.items():
        unknown = [f for f in subset if f not in features.columns]
        if unknown:
            raise KeyError(f"named subset {name!r} refers to absent feature(s) {unknown}")

    known = features[list(available)].to_numpy(dtype=np.float64)
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

    target = age_gap[usable]
    covariates = covariates[usable]
    groups = None if patient_ids is None else np.asarray(patient_ids)[usable]
    usable_features = features.loc[usable, :]

    baseline_features = (
        covariates if covariates.shape[1] else np.zeros((target.size, 1))
    )
    # Fitted once: identical for every subset, see Notes.
    baseline_r2: dict[str, float] = {}
    baseline_predictions: dict[str, np.ndarray] = {}
    for explainer in config.explainer_models:
        prediction = _out_of_fold_predictions(
            baseline_features, target, explainer, config, groups
        )
        baseline_predictions[explainer] = prediction
        baseline_r2[explainer] = max(_r2(target, prediction), 0.0)

    def evaluate(subset: Sequence[str], label: str | None = None) -> SubsetPoint:
        """Attributable share for one feature subset, most explanatory explainer."""
        columns = usable_features[list(subset)].to_numpy(dtype=np.float64)
        design = np.column_stack([covariates, columns])
        best_full = -np.inf
        best_incremental = -np.inf
        for explainer in config.explainer_models:
            prediction = _out_of_fold_predictions(
                design, target, explainer, config, groups
            )
            r2_full = _r2(target, prediction)
            # Selected on r2_full, matching DecompositionResult.most_explanatory:
            # the smallest credible unexplained residual is the conservative claim.
            if r2_full > best_full:
                best_full = r2_full
                best_incremental = r2_full - baseline_r2[explainer]
        return SubsetPoint(
            n_features=len(subset),
            features=tuple(subset),
            r2_incremental=float(best_incremental),
            r2_full=float(best_full),
            label=label,
        )

    rng = np.random.default_rng(config.seed if seed is None else seed)
    sizes = tuple(subset_sizes) if subset_sizes else _default_subset_sizes(len(available))
    sizes = tuple(sorted({int(s) for s in sizes if 1 <= int(s) <= len(available)}))
    if not sizes:
        raise ValueError("no valid subset sizes to sweep")

    levels: list[AccumulationLevel] = []
    for size in sizes:
        subsets = _draw_subsets(available, size, draws_per_size, rng)
        points = tuple(evaluate(subset) for subset in subsets)
        scores = np.array([point.r2_incremental for point in points], dtype=np.float64)
        levels.append(AccumulationLevel(
            n_features=size,
            n_subsets=len(points),
            mean_incremental=float(scores.mean()),
            sd_incremental=float(scores.std(ddof=1)) if scores.size > 1 else 0.0,
            min_incremental=float(scores.min()),
            max_incremental=float(scores.max()),
            points=points,
        ))

    named_points = tuple(
        evaluate(subset, label=name) for name, subset in named_subsets.items()
    )

    # The right-hand end, with the interval that belongs on the headline number.
    full_design = np.column_stack([covariates, known[usable]])
    best_explainer = max(
        config.explainer_models,
        key=lambda name: _r2(
            target,
            _out_of_fold_predictions(full_design, target, name, config, groups),
        ),
    )
    full_prediction = _out_of_fold_predictions(
        full_design, target, best_explainer, config, groups
    )
    full_incremental = _r2(target, full_prediction) - baseline_r2[best_explainer]
    _, incremental_ci = _paired_bootstrap_r2(
        target,
        baseline_predictions[best_explainer],
        full_prediction,
        groups,
        config,
    )

    level_sizes = np.array([level.n_features for level in levels], dtype=np.float64)
    level_means = np.array([level.mean_incremental for level in levels])
    if level_sizes.size >= 2:
        log_slope = float(np.polyfit(np.log(level_sizes), level_means, 1)[0])
        tail_gain = float(level_means[-1] - level_means[-2])
        log_step = float(np.log(level_sizes[-1]) - np.log(level_sizes[-2]))
        tail_log_slope = tail_gain / log_step if log_step > 0 else float("nan")
    else:
        log_slope = float("nan")
        tail_gain = float("nan")
        tail_log_slope = float("nan")

    return AccumulationCurve(
        levels=tuple(levels),
        named_points=named_points,
        available_features=available,
        covariates=tuple(covariate_names),
        n_recordings=int(usable.sum()),
        n_dropped=n_dropped,
        full_set_incremental=float(full_incremental),
        full_set_incremental_ci=incremental_ci,
        log_slope=log_slope,
        tail_gain=tail_gain,
        tail_log_slope=tail_log_slope,
        extrapolated_ceiling=_fit_saturating_ceiling(level_sizes, level_means),
    )
