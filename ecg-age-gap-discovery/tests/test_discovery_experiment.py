"""Tests for the discovery experiment.

The experiment must be able to return **either** answer correctly, so it is
tested both ways: on data where a link was built in, and on data where none
exists. A test suite that only checked the positive case would leave the
project unable to distinguish a discovery from a false alarm, which is the one
thing this phase exists to do.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ecg_discovery.config import ValidationFrameworkConfig
from ecg_discovery.discovery_experiments.unexplained_residual_diagnostic_link import (
    evaluate_diagnostic_link,
)

CONFIG = ValidationFrameworkConfig(
    known_features=("heart_rate_bpm", "qrs_duration_ms"),
    cv_folds=5,
    cv_repeats=3,
    seed=0,
)
NAMES = ("NORM", "MI", "STTC", "CD", "HYP")


def _cohort(n=800, link_strength=0.0, seed=0):
    """Build features, labels and a residual with a controllable link."""
    rng = np.random.default_rng(seed)
    heart_rate = rng.normal(70, 10, n)
    qrs = rng.normal(95, 12, n)
    residual = rng.normal(0, 4, n)

    # Abnormality depends on the known intervals always, and on the residual
    # only when link_strength is non-zero.
    logit = (
        -0.4
        + 0.05 * (qrs - 95)
        + link_strength * residual / 4.0
    )
    abnormal = rng.random(n) < 1 / (1 + np.exp(-logit))

    labels = np.zeros((n, len(NAMES)), dtype=int)
    labels[~abnormal, 0] = 1
    for index in np.flatnonzero(abnormal):
        labels[index, 1 + rng.integers(0, 4)] = 1

    features = pd.DataFrame({"heart_rate_bpm": heart_rate, "qrs_duration_ms": qrs})
    return residual, features, labels, rng.uniform(30, 85, n), rng.integers(0, 2, n)


# --------------------------------------------------------------------------- #
# Both directions
# --------------------------------------------------------------------------- #
def test_detects_a_link_that_was_built_in():
    """When the residual genuinely predicts diagnosis, say so."""
    residual, features, labels, ages, sexes = _cohort(link_strength=2.5)
    report = evaluate_diagnostic_link(
        residual, features, labels, NAMES, CONFIG, ages=ages, sexes=sexes
    )
    assert report.any_improvement, report.summary_text()
    assert report.links["NORM"].delta_auc > 0.02
    assert "CANDIDATE" in report.summary_text()


def test_reports_no_link_when_none_exists():
    """When the residual is pure noise, report the null - clearly."""
    residual, features, labels, ages, sexes = _cohort(link_strength=0.0)
    report = evaluate_diagnostic_link(
        residual, features, labels, NAMES, CONFIG, ages=ages, sexes=sexes
    )
    assert not report.any_improvement, report.summary_text()
    text = report.summary_text()
    assert "genuine negative result" in text
    assert "NOT" in text and "evidence of a discovered biomarker" in text


def test_a_residual_that_merely_repeats_a_known_feature_adds_nothing():
    """The comparison is *incremental*: duplicating a baseline column is not signal.

    This is the sharpest form of the false-positive hazard. The residual is a
    perfect copy of QRS duration, which genuinely does predict the label - but
    the baseline already has it, so a correct experiment reports no improvement.
    """
    _, features, labels, ages, sexes = _cohort(link_strength=0.0, seed=3)
    duplicate = features["qrs_duration_ms"].to_numpy().copy()
    report = evaluate_diagnostic_link(
        duplicate, features, labels, NAMES, CONFIG, ages=ages, sexes=sexes
    )
    assert not report.any_improvement, report.summary_text()


# --------------------------------------------------------------------------- #
# Guards against manufacturing a result
# --------------------------------------------------------------------------- #
def test_every_superclass_is_always_reported():
    """No opportunity to report only the superclass that came out best."""
    residual, features, labels, ages, sexes = _cohort(link_strength=2.0)
    report = evaluate_diagnostic_link(
        residual, features, labels, NAMES, CONFIG, ages=ages, sexes=sexes
    )
    assert set(report.links) | set(report.skipped) == set(NAMES)
    assert len(report.to_frame()) == len(report.links)
    for name in report.links:
        assert name in report.summary_text()


def test_multiplicity_correction_widens_the_intervals():
    """Testing five superclasses must not be five chances at a lucky result."""
    residual, features, labels, ages, sexes = _cohort(link_strength=0.6, seed=7)
    corrected = evaluate_diagnostic_link(
        residual, features, labels, NAMES, CONFIG, ages=ages, sexes=sexes
    )
    # A single-superclass run faces no correction, so its interval is narrower.
    single = evaluate_diagnostic_link(
        residual, features, labels[:, :1], NAMES[:1], CONFIG, ages=ages, sexes=sexes
    )
    corrected_width = np.diff(corrected.links["NORM"].delta_auc_ci)[0]
    single_width = np.diff(single.links["NORM"].delta_auc_ci)[0]
    assert corrected_width > single_width
    assert "Bonferroni" in corrected.correction


def test_small_effects_are_flagged_as_possibly_meaningless():
    """Statistical detectability is not clinical relevance, and is labelled so."""
    residual, features, labels, ages, sexes = _cohort(n=3000, link_strength=0.5, seed=11)
    report = evaluate_diagnostic_link(
        residual, features, labels, NAMES, CONFIG, ages=ages, sexes=sexes,
        minimum_meaningful_delta=0.20,      # deliberately unreachable
    )
    if report.any_improvement:
        assert "not clinical relevance" in report.summary_text()


def test_rare_superclasses_are_skipped_with_a_reason():
    """An AUC on four positives is noise; say why it was not computed."""
    residual, features, labels, ages, sexes = _cohort(seed=5)
    labels[:, 4] = 0
    labels[:3, 4] = 1                       # only three positives
    report = evaluate_diagnostic_link(
        residual, features, labels, NAMES, CONFIG, ages=ages, sexes=sexes
    )
    assert "HYP" in report.skipped
    assert "too few" in report.skipped["HYP"]
    assert "HYP" in report.summary_text()


def test_demographics_are_in_both_classifiers():
    """An improvement must not come from smuggling age back in.

    A residual constructed as a pure function of age must add nothing, because
    the baseline already contains age.
    """
    _, features, labels, ages, sexes = _cohort(link_strength=0.0, seed=13)
    report = evaluate_diagnostic_link(
        ages * 1.0, features, labels, NAMES, CONFIG, ages=ages, sexes=sexes
    )
    assert not report.any_improvement, report.summary_text()


# --------------------------------------------------------------------------- #
# Bookkeeping
# --------------------------------------------------------------------------- #
def test_confidence_interval_brackets_the_estimate():
    residual, features, labels, ages, sexes = _cohort(link_strength=1.5)
    report = evaluate_diagnostic_link(
        residual, features, labels, NAMES, CONFIG, ages=ages, sexes=sexes
    )
    for link in report.links.values():
        low, high = link.delta_auc_ci
        assert low <= link.delta_auc <= high
        assert link.improves == (low > 0.0)


def test_patient_grouping_is_respected():
    """A repeat visit must not leak across a fold boundary."""
    residual, features, labels, ages, sexes = _cohort(n=600, link_strength=1.5, seed=17)
    patient_ids = np.repeat(np.arange(300), 2)
    report = evaluate_diagnostic_link(
        residual, features, labels, NAMES, CONFIG,
        ages=ages, sexes=sexes, patient_ids=patient_ids,
    )
    assert report.n_recordings == 600
    assert report.links


def test_shape_mismatches_raise():
    residual, features, labels, ages, sexes = _cohort(n=200)
    with pytest.raises(ValueError, match="rows"):
        evaluate_diagnostic_link(residual[:100], features, labels, NAMES, CONFIG)
    with pytest.raises(ValueError, match="columns"):
        evaluate_diagnostic_link(residual, features, labels, NAMES[:3], CONFIG)
    with pytest.raises(KeyError, match="known feature"):
        evaluate_diagnostic_link(
            residual, features[["heart_rate_bpm"]], labels, NAMES,
            dataclasses.replace(CONFIG, known_features=("heart_rate_bpm", "qt_interval_ms")),
        )


def test_report_frame_has_one_row_per_tested_superclass():
    residual, features, labels, ages, sexes = _cohort(link_strength=1.0)
    report = evaluate_diagnostic_link(
        residual, features, labels, NAMES, CONFIG, ages=ages, sexes=sexes
    )
    frame = report.to_frame()
    assert set(frame.columns) >= {
        "superclass", "n_positive", "auc_baseline", "auc_augmented",
        "delta_auc", "delta_ci_low", "delta_ci_high", "improves",
    }
    assert (frame.prevalence.between(0, 1)).all()


def test_rare_superclass_cannot_produce_a_spurious_discovery():
    """A class with a handful of positives must not be testable at all.

    Caught by a smoke run: a superclass with 7 positives and a baseline AUC of
    0.220 - far worse than chance - was reported as a statistically
    distinguishable improvement. It was noise. An AUC on that few positives is
    dominated by which fold they land in, and a spurious gain on a rare class is
    exactly the result that gets written up as a discovery.
    """
    residual, features, labels, ages, sexes = _cohort(n=400, link_strength=0.0, seed=23)
    labels[:, 3] = 0
    labels[:7, 3] = 1                       # 7 positives, above the old threshold of 5
    report = evaluate_diagnostic_link(
        residual, features, labels, NAMES, CONFIG, ages=ages, sexes=sexes
    )
    assert "CD" in report.skipped
    assert "CD" not in report.links
    assert "too few" in report.skipped["CD"]


def test_minimum_positive_threshold_is_configurable_and_enforced():
    residual, features, labels, ages, sexes = _cohort(n=600, link_strength=1.0, seed=29)
    strict = evaluate_diagnostic_link(
        residual, features, labels, NAMES,
        dataclasses.replace(CONFIG, min_positives_for_link=250),
        ages=ages, sexes=sexes,
    )
    # With a threshold that high almost nothing qualifies.
    assert len(strict.links) < len(NAMES)
    for reason in strict.skipped.values():
        assert "250" in reason
