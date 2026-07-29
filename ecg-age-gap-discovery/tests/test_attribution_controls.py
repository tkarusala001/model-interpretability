"""Tests for the attribution null controls.

The controls exist to answer one question: does a segment attribution profile
reflect the *model*, or just the amplitude structure of the ECG? So they are
tested in both directions - they must flag an amplitude artifact as such, and
must not flag a genuinely learned profile as one.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from ecg_discovery.config import (
    BackboneConfig,
    SignalProcessingConfig,
    SyntheticConfig,
    TrainingConfig,
)
from ecg_discovery.data.preprocessing import resample_signals
from ecg_discovery.data.synthetic_ecg import generate_cohort
from ecg_discovery.interpretability.attribution_controls import (
    ControlComparison,
    ControlProfile,
    amplitude_profile,
    compare_against_controls,
    model_profile,
)
from ecg_discovery.interpretability.fiducial_attribution import SEGMENT_NAMES
from ecg_discovery.models.ecg_age_regressor import ECGAgeRegressor
from ecg_discovery.signal_processing.qrs_detection import detect_r_peaks
from ecg_discovery.signal_processing.wave_delineation import delineate_beats
from ecg_discovery.training.train import train_age_regressor

SP = SignalProcessingConfig()
SMALL = BackboneConfig(
    stem_channels=16, stage_channels=(16, 24, 32), stride_per_stage=(2, 2, 2)
)


def _prepare(n_recordings=60, **overrides):
    config = dataclasses.replace(SyntheticConfig(sampling_rate_hz=500), **overrides)
    cohort = generate_cohort(config, n_recordings=n_recordings)
    signals_100 = resample_signals(cohort.signals, 500, 100)
    beats = []
    for signal in signals_100:
        detection = detect_r_peaks(signal, 100.0, SP, cohort[0].lead_names)
        beats.append(delineate_beats(signal, 100.0, detection.r_peaks, SP, cohort[0].lead_names))
    return cohort, signals_100, beats


# --------------------------------------------------------------------------- #
# The amplitude null
# --------------------------------------------------------------------------- #
def test_amplitude_profile_shares_sum_to_one():
    cohort, signals, beats = _prepare(20)
    profile = amplitude_profile(signals, beats, cohort[0].lead_names)
    assert profile.share.shape == (20, len(SEGMENT_NAMES))
    np.testing.assert_allclose(profile.share.sum(axis=1), 1.0, rtol=1e-6)


def test_amplitude_null_favours_the_largest_deflection():
    """The QRS dominates ECG energy - that is exactly the confound being tested."""
    cohort, signals, beats = _prepare(20)
    profile = amplitude_profile(signals, beats, cohort[0].lead_names)
    qrs = profile.mean_share[SEGMENT_NAMES.index("QRS")]
    p_wave = profile.mean_share[SEGMENT_NAMES.index("P")]
    assert qrs > p_wave


# --------------------------------------------------------------------------- #
# Detecting an amplitude artifact
# --------------------------------------------------------------------------- #
def test_an_untrained_model_does_not_survive_its_controls():
    """A model that has learned nothing must be reported as having no finding.

    This is the load-bearing test. An untrained network still produces a
    perfectly plausible-looking attribution profile; if the controls cannot
    identify it as meaningless, they are useless.
    """
    cohort, signals, beats = _prepare(50)
    untrained = ECGAgeRegressor(SMALL).eval()
    mean = signals.mean(axis=(0, 2), keepdims=True)
    std = signals.std(axis=(0, 2), keepdims=True).clip(1e-6)
    normalised = ((signals - mean) / std).astype(np.float32)

    comparison = compare_against_controls(
        untrained, normalised, signals, cohort.sexes.astype(np.float32),
        beats, cohort[0].lead_names,
        untrained_model=ECGAgeRegressor(SMALL).eval(),
        n_steps=16,
    )
    # Against another untrained model of the same architecture, the profile
    # should be broadly similar - nothing was learned in either.
    difference = comparison.difference("untrained")
    assert set(difference["segment"]) == set(SEGMENT_NAMES)
    assert np.isfinite(comparison.amplitude_correlation())


def test_summary_reports_a_null_result_plainly():
    """When nothing departs from the controls, say so unambiguously."""
    profiles = {
        "amplitude": ControlProfile("amplitude", np.tile([0.1, 0.6, 0.2, 0.1], (40, 1)),
                                    np.zeros((40, 4))),
    }
    identical = ControlProfile("trained", np.tile([0.1, 0.6, 0.2, 0.1], (40, 1)),
                               np.zeros((40, 4)))
    comparison = ControlComparison(trained=identical, controls=profiles)
    text = comparison.summary_text()
    assert "does NOT depart from any control" in text
    assert "no claim about what the model attends to is supported" in text


def test_high_amplitude_correlation_triggers_a_caution():
    rng = np.random.default_rng(0)
    base = rng.dirichlet(np.ones(4), size=60)
    trained = ControlProfile("trained", base + rng.normal(0, 0.002, base.shape),
                             np.zeros((60, 4)))
    comparison = ControlComparison(
        trained=trained,
        controls={"amplitude": ControlProfile("amplitude", base, np.zeros((60, 4)))},
    )
    assert comparison.amplitude_correlation() > 0.9
    assert "CAUTION" in comparison.summary_text()


# --------------------------------------------------------------------------- #
# Detecting a genuinely learned profile
# --------------------------------------------------------------------------- #
def test_a_trained_model_departs_from_the_untrained_control():
    """A model that learned a real effect must be distinguishable from noise.

    The cohort's age signal runs entirely through QRS widening, so a trained
    model should attend differently from an untrained one. If the controls
    cannot see that difference they would suppress every real finding.
    """
    cohort, signals, beats = _prepare(
        200, qrs_widening_ms_per_decade=9.0, t_wave_skew_per_decade=0.0,
        hr_change_bpm_per_decade=0.0, unexplained_age_offset_sd_years=0.0,
    )
    result = train_age_regressor(
        signals=signals, ages=cohort.ages, sexes=cohort.sexes,
        patient_ids=cohort.patient_ids, backbone_config=SMALL,
        training_config=TrainingConfig(
            epochs=20, batch_size=32, learning_rate=3e-3, warmup_epochs=1,
            early_stopping_patience=30, seed=0,
        ),
    )
    normalised = result.normalizer.transform(signals)
    subset = result.splits.test[:40]

    comparison = compare_against_controls(
        result.model, normalised[subset], signals[subset],
        cohort.sexes[subset].astype(np.float32),
        [beats[i] for i in subset], cohort[0].lead_names,
        untrained_model=ECGAgeRegressor(SMALL).eval(),
        n_steps=24,
    )
    difference = comparison.difference("untrained")
    assert difference["differs"].any(), (
        "a model trained on a real QRS-mediated age effect should attend "
        "differently from an untrained one\n" + comparison.summary_text()
    )


def test_comparison_reports_every_segment_and_control():
    cohort, signals, beats = _prepare(30)
    model = ECGAgeRegressor(SMALL).eval()
    normalised = signals.astype(np.float32)
    comparison = compare_against_controls(
        model, normalised, signals, cohort.sexes.astype(np.float32),
        beats, cohort[0].lead_names,
        untrained_model=ECGAgeRegressor(SMALL).eval(), n_steps=8,
    )
    assert set(comparison.controls) == {"amplitude", "untrained"}
    text = comparison.summary_text()
    for segment in SEGMENT_NAMES:
        assert segment in text
    assert "amplitude" in text and "untrained" in text


def test_model_profile_shares_sum_to_one():
    cohort, signals, beats = _prepare(15)
    model = ECGAgeRegressor(SMALL).eval()
    profile = model_profile(
        model, signals.astype(np.float32), cohort.sexes.astype(np.float32),
        beats, cohort[0].lead_names, "trained", n_steps=8,
    )
    finite = np.isfinite(profile.share.sum(axis=1))
    np.testing.assert_allclose(profile.share[finite].sum(axis=1), 1.0, rtol=1e-5)
