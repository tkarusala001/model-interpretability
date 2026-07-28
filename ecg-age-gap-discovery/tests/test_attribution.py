"""Tests for the hand-implemented attribution methods.

Integrated Gradients is defined by axioms, not by an implementation, so the
tests check the axioms directly:

*Completeness* - attributions sum to the model's output change between baseline
and input. This is the property that caught a silent backend failure during
development (see the module docstring in ``attribution.py``), so it is checked
in several regimes.

*Sensitivity* - an input feature the model ignores receives zero attribution.

*Exactness on linear models* - for a linear model the path integral is
analytic, and Integrated Gradients must equal input x gradient exactly. That
gives a closed-form check against a hand-computable answer, independent of any
reference implementation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from torch import nn

from ecg_discovery.config import BackboneConfig
from ecg_discovery.interpretability.attribution import (
    _resolve_attribution_device,
    input_x_gradient,
    integrated_gradients,
    make_baseline,
)
from ecg_discovery.models.ecg_age_regressor import ECGAgeRegressor
from ecg_discovery.runtime import set_global_seed


class LinearModel(nn.Module):
    """A model whose attributions can be computed by hand: y = sum(w * x) + b."""

    def __init__(self, n_leads: int = 3, n_samples: int = 20, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(n_leads, n_samples, generator=generator))
        self.bias = nn.Parameter(torch.tensor(5.0))

    def forward(self, waveform: torch.Tensor, sex: torch.Tensor | None = None):
        return (waveform * self.weight).sum(dim=(1, 2)) + self.bias


class IgnoresOneLead(nn.Module):
    """Reads every lead except lead 0, for testing the sensitivity axiom."""

    def forward(self, waveform: torch.Tensor, sex: torch.Tensor | None = None):
        return waveform[:, 1:, :].sum(dim=(1, 2))


class CubicModel(nn.Module):
    """A strongly non-linear model: y = (sum of inputs)^3.

    Needed because the untrained regressor is very nearly linear over the
    integration path - its final layer starts at a small scale - so Integrated
    Gradients is essentially exact on it at any step count and cannot
    demonstrate Riemann convergence. A cubic has genuine curvature, so a coarse
    integral is measurably wrong and refining it measurably helps.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, waveform: torch.Tensor, sex: torch.Tensor | None = None):
        return waveform.sum(dim=(1, 2)) ** 3 + 0 * self.dummy


class ConstantModel(nn.Module):
    """Output independent of the input."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, waveform: torch.Tensor, sex: torch.Tensor | None = None):
        return torch.full((waveform.shape[0],), 42.0) + 0 * self.dummy


def _small_regressor() -> ECGAgeRegressor:
    set_global_seed(0)
    return ECGAgeRegressor(
        BackboneConfig(stem_channels=8, stage_channels=(8, 12), stride_per_stage=(2, 2))
    ).eval()


# --------------------------------------------------------------------------- #
# Axioms
# --------------------------------------------------------------------------- #
def test_completeness_on_a_nonlinear_model():
    """Attributions must sum to prediction minus baseline prediction."""
    model = _small_regressor()
    waveform = torch.randn(4, 12, 400)
    sex = torch.zeros(4)

    result = integrated_gradients(model, waveform, sex, n_steps=128)
    summed = result.attributions.sum(axis=(1, 2))
    expected = result.prediction - result.baseline_prediction
    np.testing.assert_allclose(summed, expected, rtol=0, atol=0.05)


def test_completeness_error_shrinks_with_more_steps():
    """A genuine Riemann error must fall as the integration is refined.

    Distinguishes an under-resolved integral - which more steps fix - from a
    broken gradient, which they do not. That distinction is what let the MPS
    failure be identified: its error stayed flat from 16 steps to 1024.
    """
    model = CubicModel()
    waveform = torch.randn(2, 3, 40) * 0.2

    coarse = integrated_gradients(model, waveform, n_steps=2, max_relative_error=None)
    fine = integrated_gradients(model, waveform, n_steps=512, max_relative_error=None)

    coarse_error = np.abs(coarse.convergence_delta).mean()
    fine_error = np.abs(fine.convergence_delta).mean()
    assert coarse_error > 1e-4, "expected a coarse integral to be measurably wrong"
    assert fine_error < 0.05 * coarse_error


def test_integrated_gradients_equals_input_x_gradient_for_a_linear_model():
    """On a linear model the path integral is analytic, so the two must agree.

    A closed-form check against hand-computable truth, needing no reference
    implementation to compare against.
    """
    model = LinearModel()
    waveform = torch.randn(3, 3, 20)

    ig = integrated_gradients(model, waveform, n_steps=32)
    ixg = input_x_gradient(model, waveform)
    np.testing.assert_allclose(ig.attributions, ixg.attributions, rtol=1e-4, atol=1e-5)

    # And both equal w * x exactly, computed by hand.
    expected = (waveform * model.weight).detach().numpy()
    np.testing.assert_allclose(ig.attributions, expected, rtol=1e-4, atol=1e-5)


def test_sensitivity_an_ignored_input_gets_zero_attribution():
    """A lead the model never reads must receive exactly no credit."""
    model = IgnoresOneLead()
    waveform = torch.randn(2, 4, 30)

    result = integrated_gradients(model, waveform, n_steps=16, max_relative_error=None)
    np.testing.assert_allclose(result.attributions[:, 0, :], 0.0, atol=1e-6)
    assert np.abs(result.attributions[:, 1:, :]).sum() > 1.0


def test_constant_model_gets_zero_attribution():
    model = ConstantModel()
    waveform = torch.randn(2, 3, 30)
    result = integrated_gradients(model, waveform, n_steps=8)
    np.testing.assert_allclose(result.attributions, 0.0, atol=1e-6)
    np.testing.assert_allclose(result.convergence_delta, 0.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# The completeness check itself
# --------------------------------------------------------------------------- #
class BrokenGradientModel(nn.Module):
    """Stands in for a backend that returns wrong gradients without erroring.

    Its forward output depends on the input, but the gradient path is severed
    with ``detach``, so autograd reports zero. This is exactly the failure mode
    observed on Apple MPS during development: plausible outputs, silently dead
    gradients.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, waveform: torch.Tensor, sex: torch.Tensor | None = None):
        return waveform.sum(dim=(1, 2)).detach() + 0 * self.dummy


def test_completeness_check_fires_on_silently_broken_gradients():
    """The guard must catch the failure mode it was written for."""
    model = BrokenGradientModel()
    waveform = torch.randn(3, 4, 50) * 10.0

    with pytest.warns(RuntimeWarning, match="completeness"):
        result = integrated_gradients(model, waveform, n_steps=16)
    # Attributions are zero while the model output plainly moved.
    np.testing.assert_allclose(result.attributions, 0.0, atol=1e-8)
    assert np.abs(result.prediction - result.baseline_prediction).mean() > 1.0


def test_completeness_check_can_be_disabled():
    model = BrokenGradientModel()
    waveform = torch.randn(2, 4, 50) * 10.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        integrated_gradients(model, waveform, n_steps=8, max_relative_error=None)


def test_completeness_check_tolerates_a_tiny_span():
    """A near-zero output change must not be flagged on relative error alone.

    Without an absolute term, a recording whose prediction happens to sit near
    its baseline prediction divides a negligible delta by a near-zero span and
    reports a huge relative error - a false alarm, not a failure.
    """
    model = _small_regressor()
    waveform = torch.randn(3, 12, 400) * 1e-4     # barely moves the model
    sex = torch.zeros(3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = integrated_gradients(model, waveform, sex, n_steps=32)
    assert np.abs(result.prediction - result.baseline_prediction).max() < 1.0


# --------------------------------------------------------------------------- #
# Implementation details that must not change the answer
# --------------------------------------------------------------------------- #
def test_step_batching_does_not_change_the_result():
    """Path points are stacked into one pass; that must be purely a speed-up.

    Safe only because the model runs in eval mode, where batch normalisation
    uses fixed running statistics and stacked points cannot influence each
    other. If that ever stopped holding, this test would catch it.
    """
    model = _small_regressor()
    waveform = torch.randn(3, 12, 300)
    sex = torch.zeros(3)

    one_at_a_time = integrated_gradients(model, waveform, sex, n_steps=32, step_batch=1)
    stacked = integrated_gradients(model, waveform, sex, n_steps=32, step_batch=32)
    np.testing.assert_allclose(
        one_at_a_time.attributions, stacked.attributions, rtol=1e-4, atol=1e-6
    )


def test_attribution_is_deterministic():
    model = _small_regressor()
    waveform = torch.randn(2, 12, 300)
    sex = torch.zeros(2)
    first = integrated_gradients(model, waveform, sex, n_steps=16)
    second = integrated_gradients(model, waveform, sex, n_steps=16)
    np.testing.assert_array_equal(first.attributions, second.attributions)


def test_model_is_left_in_its_original_mode():
    """Attribution switches to eval mode and must restore what it found."""
    model = _small_regressor().train()
    integrated_gradients(model, torch.randn(2, 12, 200), torch.zeros(2), n_steps=4)
    assert model.training
    model.eval()
    integrated_gradients(model, torch.randn(2, 12, 200), torch.zeros(2), n_steps=4)
    assert not model.training


def test_dropout_does_not_perturb_attribution():
    """Eval mode must be genuinely active, or attributions would be random."""
    model = ECGAgeRegressor(
        BackboneConfig(
            stem_channels=8, stage_channels=(8, 12), stride_per_stage=(2, 2), dropout=0.5
        )
    ).train()
    waveform = torch.randn(2, 12, 300)
    sex = torch.zeros(2)
    first = integrated_gradients(model, waveform, sex, n_steps=8)
    second = integrated_gradients(model, waveform, sex, n_steps=8)
    np.testing.assert_allclose(first.attributions, second.attributions, rtol=1e-5, atol=1e-7)


# --------------------------------------------------------------------------- #
# Devices
# --------------------------------------------------------------------------- #
def test_mps_is_avoided_by_default():
    """MPS returned silently zeroed gradients during development; avoid it."""
    model = _small_regressor()
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this machine")
    model = model.to("mps")
    assert _resolve_attribution_device(model, None).type == "cpu"
    # An explicit request is still honoured.
    assert _resolve_attribution_device(model, "mps").type == "mps"


def test_cpu_model_stays_on_cpu():
    model = _small_regressor()
    assert _resolve_attribution_device(model, None).type == "cpu"


def test_inputs_on_a_different_device_are_accepted():
    """Callers hold CPU arrays even when training ran elsewhere."""
    model = _small_regressor()
    result = integrated_gradients(model, torch.randn(2, 12, 200), torch.zeros(2), n_steps=8)
    assert result.attributions.shape == (2, 12, 200)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def test_zero_baseline_is_all_zeros():
    waveform = torch.randn(2, 3, 10) + 5.0
    np.testing.assert_allclose(make_baseline(waveform, "zero").numpy(), 0.0)


def test_flatline_baseline_is_constant_per_lead():
    waveform = torch.randn(2, 3, 50)
    baseline = make_baseline(waveform, "flatline")
    assert baseline.shape == waveform.shape
    # Constant along time within each lead.
    assert torch.allclose(baseline.std(dim=-1), torch.zeros(2, 3), atol=1e-6)
    # And equal to that lead's median.
    torch.testing.assert_close(baseline[..., 0], waveform.median(dim=-1).values)


def test_mean_signal_baseline_requires_a_reference():
    waveform = torch.randn(2, 3, 10)
    with pytest.raises(ValueError, match="requires a reference"):
        make_baseline(waveform, "mean_signal")
    reference = torch.ones(3, 10)
    np.testing.assert_allclose(
        make_baseline(waveform, "mean_signal", reference).numpy(), 1.0
    )


def test_unknown_baseline_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown baseline kind"):
        make_baseline(torch.randn(1, 3, 10), "gaussian_noise")  # type: ignore[arg-type]


def test_baseline_choice_changes_attributions():
    """The baseline is a real methodological choice, not a formality."""
    model = _small_regressor()
    waveform = torch.randn(2, 12, 300) + 2.0
    sex = torch.zeros(2)
    zero = integrated_gradients(model, waveform, sex, baseline_kind="zero", n_steps=32)
    flat = integrated_gradients(model, waveform, sex, baseline_kind="flatline", n_steps=32)
    assert not np.allclose(zero.attributions, flat.attributions, atol=1e-4)


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
def test_single_recording_input_is_accepted():
    model = _small_regressor()
    result = integrated_gradients(model, torch.randn(12, 300), torch.zeros(1), n_steps=8)
    assert result.attributions.shape == (1, 12, 300)


def test_input_x_gradient_reports_no_completeness_guarantee():
    """It has none, so the field must be NaN rather than a misleading number."""
    model = _small_regressor()
    result = input_x_gradient(model, torch.randn(2, 12, 200), torch.zeros(2))
    assert np.isnan(result.convergence_delta).all()
    assert result.method == "input_x_gradient"


def test_bad_shapes_and_sizes_are_rejected():
    model = _small_regressor()
    with pytest.raises(ValueError, match="batch, n_leads, n_samples"):
        integrated_gradients(model, torch.randn(2, 3, 12, 200), n_steps=4)
    with pytest.raises(ValueError, match="sex has"):
        integrated_gradients(model, torch.randn(2, 12, 200), torch.zeros(3), n_steps=4)
    with pytest.raises(ValueError, match="n_steps"):
        integrated_gradients(model, torch.randn(2, 12, 200), torch.zeros(2), n_steps=0)
    with pytest.raises(ValueError, match="step_batch"):
        integrated_gradients(
            model, torch.randn(2, 12, 200), torch.zeros(2), n_steps=4, step_batch=0
        )
