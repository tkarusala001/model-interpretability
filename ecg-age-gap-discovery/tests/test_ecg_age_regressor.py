"""Tests for the from-scratch ECG age regressor.

Three things need establishing before Phase 5 trains this model in earnest:

1. **It is genuinely from scratch.** Every parameter is determined by the random
   seed and nothing else - no weights are loaded from anywhere. This is a
   load-bearing claim of the paper, so it is tested rather than asserted.
2. **Gradients flow, including to the input.** Phase 6 differentiates the
   predicted age with respect to individual signal samples, so input
   differentiability is a hard requirement, not a nicety.
3. **It can actually learn the injected age signal.** The synthetic cohort has a
   known age-morphology relationship built into it; a model that cannot recover
   that has no business being trusted on real ECGs.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from ecg_discovery.config import BackboneConfig, SyntheticConfig
from ecg_discovery.data.synthetic_ecg import generate_cohort
from ecg_discovery.models.ecg_age_regressor import ECGAgeRegressor, ResidualBlock
from ecg_discovery.runtime import set_global_seed

DEFAULT = BackboneConfig()
NO_SEX = dataclasses.replace(DEFAULT, use_sex_input=False)


def _batch(batch_size: int = 4, n_samples: int = 1000):
    waveform = torch.randn(batch_size, 12, n_samples)
    sex = torch.randint(0, 2, (batch_size,)).float()
    return waveform, sex


# --------------------------------------------------------------------------- #
# Shapes and interface
# --------------------------------------------------------------------------- #
def test_output_shape_is_one_age_per_recording():
    model = ECGAgeRegressor(DEFAULT).eval()
    waveform, sex = _batch(5)
    assert model(waveform, sex).shape == (5,)


@pytest.mark.parametrize("n_samples", [1000, 2500, 5000])
def test_any_recording_length_is_accepted(n_samples):
    """Global pooling makes the model length-independent.

    This matters concretely: the model trains on 100 Hz recordings but must also
    accept the 500 Hz signals used for interval measurement without rebuilding.
    """
    model = ECGAgeRegressor(DEFAULT).eval()
    waveform, sex = _batch(2, n_samples)
    assert model(waveform, sex).shape == (2,)


def test_wrong_lead_count_is_rejected():
    model = ECGAgeRegressor(DEFAULT).eval()
    with pytest.raises(ValueError, match="expects 12 leads"):
        model(torch.randn(2, 8, 1000), torch.zeros(2))


def test_missing_time_dimension_is_rejected():
    model = ECGAgeRegressor(DEFAULT).eval()
    with pytest.raises(ValueError, match="batch, channels, time"):
        model(torch.randn(12, 1000), torch.zeros(1))


def test_missing_sex_is_rejected_when_the_model_expects_it():
    model = ECGAgeRegressor(DEFAULT).eval()
    with pytest.raises(ValueError, match="use_sex_input"):
        model(torch.randn(2, 12, 1000))


def test_mismatched_sex_batch_size_is_rejected():
    model = ECGAgeRegressor(DEFAULT).eval()
    with pytest.raises(ValueError, match="batch size"):
        model(torch.randn(4, 12, 1000), torch.zeros(3))


def test_sex_can_be_ablated():
    """``use_sex_input: false`` must give a working model that ignores sex."""
    model = ECGAgeRegressor(NO_SEX).eval()
    waveform, _ = _batch(3)
    assert model(waveform).shape == (3,)
    # Passing sex anyway is harmless and changes nothing.
    torch.testing.assert_close(model(waveform), model(waveform, torch.ones(3)))


def test_sex_input_changes_the_prediction():
    """Guard: sex must actually reach the head, not be silently dropped."""
    set_global_seed(0)
    model = ECGAgeRegressor(DEFAULT).eval()
    # The final layer starts at a deliberately small scale, so train briefly to
    # give the head a non-degenerate mapping before checking that sex matters.
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    waveform, sex = _batch(8)
    target = torch.rand(8) * 40 + 40
    for _ in range(5):
        optimiser.zero_grad()
        torch.nn.functional.mse_loss(model(waveform, sex), target).backward()
        optimiser.step()

    model.eval()
    male = model(waveform, torch.zeros(8))
    female = model(waveform, torch.ones(8))
    assert not torch.allclose(male, female)


# --------------------------------------------------------------------------- #
# From-scratch initialisation
# --------------------------------------------------------------------------- #
def test_parameters_are_fully_determined_by_the_seed():
    """No pretrained weights: the seed alone fixes every parameter.

    If any weight were loaded from a checkpoint or downloaded, two models built
    under the same seed could not agree bit for bit while models under different
    seeds differed.
    """
    set_global_seed(11)
    first = ECGAgeRegressor(DEFAULT)
    set_global_seed(11)
    second = ECGAgeRegressor(DEFAULT)
    for (name_a, a), (name_b, b) in zip(
        first.named_parameters(), second.named_parameters()
    ):
        assert name_a == name_b
        torch.testing.assert_close(a, b)

    set_global_seed(12)
    different = ECGAgeRegressor(DEFAULT)
    assert not torch.allclose(
        first.stem[0].weight, different.stem[0].weight
    )


def test_initial_predictions_start_near_the_population_mean():
    """The final bias starts at the age prior, so training does not begin at zero.

    Predictions start *near* the prior rather than exactly on it. The final
    weight is deliberately small but non-zero: zeroing it would give an exact
    prior at initialisation but would also zero the gradient to every upstream
    parameter, leaving the backbone dead on the first step. See
    ``test_gradient_reaches_every_parameter``.
    """
    model = ECGAgeRegressor(DEFAULT).eval()
    waveform, sex = _batch(6)
    predictions = model(waveform, sex)
    assert torch.all((predictions - DEFAULT.age_prior_mean).abs() < 5.0)
    assert not torch.allclose(predictions, torch.full((6,), DEFAULT.age_prior_mean))


def test_convolution_weights_are_not_degenerate():
    """Kaiming initialisation must produce real spread, not zeros or constants."""
    model = ECGAgeRegressor(DEFAULT)
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d):
            assert module.weight.std().item() > 1e-4


# --------------------------------------------------------------------------- #
# Capacity and temporal context
# --------------------------------------------------------------------------- #
def test_model_is_small_enough_for_the_dataset():
    """~18.9k PTB-XL patients cannot support an arbitrarily large model."""
    model = ECGAgeRegressor(DEFAULT)
    assert model.n_parameters == 326_497
    assert model.n_parameters < 500_000


def test_receptive_field_spans_several_cardiac_cycles():
    """A feature must see more than one beat, or it cannot represent rhythm.

    At 100 Hz one cardiac cycle is roughly 100 samples.
    """
    model = ECGAgeRegressor(DEFAULT)
    assert model.receptive_field_samples == 637
    assert model.receptive_field_samples > 3 * 100


def test_receptive_field_is_computed_from_the_configuration():
    """The figure must track the architecture, not be a stale constant."""
    shallow = ECGAgeRegressor(
        BackboneConfig(stage_channels=(16, 16), stride_per_stage=(2, 2), blocks_per_stage=1)
    )
    deep = ECGAgeRegressor(
        BackboneConfig(
            stage_channels=(16, 16, 16), stride_per_stage=(2, 2, 2), blocks_per_stage=1
        )
    )
    assert deep.receptive_field_samples > shallow.receptive_field_samples

    wider_kernel = ECGAgeRegressor(
        BackboneConfig(
            stage_channels=(16, 16), stride_per_stage=(2, 2),
            blocks_per_stage=1, kernel_size=11,
        )
    )
    assert wider_kernel.receptive_field_samples > shallow.receptive_field_samples


# --------------------------------------------------------------------------- #
# Gradients
# --------------------------------------------------------------------------- #
def test_gradient_reaches_every_parameter():
    """Every parameter must receive gradient on the very first step.

    This is not a formality. An earlier version of this model zero-initialised
    the final layer's weight so that initial predictions were exactly the age
    prior. That silently zeroed the gradient to *every* upstream parameter,
    leaving the whole backbone untrained until the head drifted away from zero.
    The model looked fine and trained slowly for no visible reason. This test is
    what caught it.
    """
    model = ECGAgeRegressor(DEFAULT).train()
    waveform, sex = _batch(4)
    model(waveform, sex).sum().backward()

    missing = [
        name for name, parameter in model.named_parameters()
        if parameter.grad is None or torch.all(parameter.grad == 0)
    ]
    assert missing == [], f"parameters receiving no gradient: {missing}"


def test_gradient_flows_to_the_input_waveform():
    """Phase 6 differentiates predicted age with respect to signal samples.

    Without this the entire attribution stage is impossible, so it is checked
    here rather than discovered later.
    """
    model = ECGAgeRegressor(DEFAULT).eval()
    waveform, sex = _batch(2)
    waveform.requires_grad_(True)
    model(waveform, sex).sum().backward()

    assert waveform.grad is not None
    assert waveform.grad.shape == waveform.shape
    assert torch.isfinite(waveform.grad).all()


def test_gradients_are_finite_on_realistic_input():
    """Real ECG amplitudes, not standard normal noise."""
    config = SyntheticConfig(sampling_rate_hz=100)
    cohort = generate_cohort(config, n_recordings=4)
    waveform = torch.tensor(cohort.signals, dtype=torch.float32, requires_grad=True)
    sex = torch.tensor(cohort.sexes, dtype=torch.float32)

    model = ECGAgeRegressor(DEFAULT).train()
    loss = torch.nn.functional.mse_loss(
        model(waveform, sex), torch.tensor(cohort.ages, dtype=torch.float32)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert waveform.grad is not None and torch.isfinite(waveform.grad).all()


# --------------------------------------------------------------------------- #
# Train / eval behaviour
# --------------------------------------------------------------------------- #
def test_eval_mode_is_deterministic_and_train_mode_is_not():
    """Dropout must be active in training and disabled in evaluation."""
    set_global_seed(3)
    model = ECGAgeRegressor(DEFAULT)
    waveform, sex = _batch(8)

    model.eval()
    torch.testing.assert_close(model(waveform, sex), model(waveform, sex))

    model.train()
    assert not torch.allclose(model(waveform, sex), model(waveform, sex))


def test_residual_block_preserves_shape_and_downsamples_correctly():
    x = torch.randn(2, 16, 400)
    same = ResidualBlock(16, 16, kernel_size=7, stride=1, dropout=0.0).eval()
    assert same(x).shape == (2, 16, 400)

    changed = ResidualBlock(16, 32, kernel_size=7, stride=2, dropout=0.0).eval()
    assert changed(x).shape == (2, 32, 200)


# --------------------------------------------------------------------------- #
# The training-signal sanity check
# --------------------------------------------------------------------------- #
def test_model_learns_the_injected_age_signal():
    """The model must recover the age relationship built into the synthetic data.

    This is the sanity check the build plan requires before the architecture is
    trusted: the synthetic cohort has QRS widening, heart-rate change and T-wave
    skew deliberately tied to age, so a working model should predict age far
    better than the best constant guess.

    A deliberately hard bar is used - beating the *mean-prediction* baseline,
    which is what a model that has learned nothing would converge to - rather
    than merely observing that the loss went down, since a loss can fall simply
    by fitting the mean.
    """
    set_global_seed(0)
    config = SyntheticConfig(sampling_rate_hz=100)
    cohort = generate_cohort(config, n_recordings=256)

    signals = torch.tensor(cohort.signals, dtype=torch.float32)
    ages = torch.tensor(cohort.ages, dtype=torch.float32)
    sexes = torch.tensor(cohort.sexes, dtype=torch.float32)

    # Per-lead standardisation, as the real pipeline applies.
    mean = signals.mean(dim=(0, 2), keepdim=True)
    std = signals.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    signals = (signals - mean) / std

    train, test = slice(0, 192), slice(192, 256)
    model = ECGAgeRegressor(DEFAULT).train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    # 30 epochs is enough: held-out MAE reaches ~5 years by epoch 20 against a
    # 17-year constant-prediction baseline, and running longer only lengthens the
    # test suite.
    first_loss = None
    for epoch in range(30):
        permutation = torch.randperm(192)
        epoch_losses = []
        for start in range(0, 192, 32):
            batch = permutation[start : start + 32]
            optimiser.zero_grad()
            predicted = model(signals[train][batch], sexes[train][batch])
            loss = torch.nn.functional.smooth_l1_loss(
                predicted, ages[train][batch], beta=5.0
            )
            loss.backward()
            optimiser.step()
            epoch_losses.append(loss.item())
        if epoch == 0:
            first_loss = float(np.mean(epoch_losses))
    last_loss = float(np.mean(epoch_losses))

    assert last_loss < first_loss, "training loss did not decrease at all"

    model.eval()
    with torch.no_grad():
        predictions = model(signals[test], sexes[test])
    model_mae = (predictions - ages[test]).abs().mean().item()
    # What a model that learned nothing would achieve: always guess the mean.
    baseline_mae = (ages[train].mean() - ages[test]).abs().mean().item()

    assert model_mae < 0.75 * baseline_mae, (
        f"held-out MAE {model_mae:.1f} years is not meaningfully better than the "
        f"constant-prediction baseline {baseline_mae:.1f} years - the model is "
        "not learning the injected age signal"
    )
    # Predictions must vary with the input, not collapse onto one value.
    assert predictions.std().item() > 2.0
