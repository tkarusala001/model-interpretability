"""Tests for the configuration system.

The config layer's job is to fail loudly on nonsense rather than let a run
proceed with a physically implausible setting, so most of these tests assert
that bad input raises rather than that good input parses.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from ecg_discovery.config import (
    BackboneConfig,
    ConfigError,
    DataConfig,
    SyntheticConfig,
    TrainingConfig,
    ValidationFrameworkConfig,
    config_to_dict,
    load_config,
)
from ecg_discovery.runtime import RunContext, set_global_seed

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


# --------------------------------------------------------------------------- #
# The shipped configs must actually load
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("filename", "cls"),
    [
        ("data.yaml", DataConfig),
        ("synthetic.yaml", SyntheticConfig),
        ("backbone.yaml", BackboneConfig),
        ("training.yaml", TrainingConfig),
        ("validation_framework.yaml", ValidationFrameworkConfig),
    ],
)
def test_shipped_configs_load(filename, cls):
    config = load_config(cls, CONFIG_DIR / filename)
    assert isinstance(config, cls)


def test_shipped_configs_are_json_serialisable():
    """Run directories store configs as JSON, so every value must survive it."""
    for filename, cls in [
        ("data.yaml", DataConfig),
        ("synthetic.yaml", SyntheticConfig),
        ("backbone.yaml", BackboneConfig),
        ("training.yaml", TrainingConfig),
        ("validation_framework.yaml", ValidationFrameworkConfig),
    ]:
        payload = config_to_dict(load_config(cls, CONFIG_DIR / filename))
        json.dumps(payload)


def test_configs_are_frozen():
    """Configs are immutable so a run cannot mutate its own recorded settings."""
    config = load_config(TrainingConfig, CONFIG_DIR / "training.yaml")
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.learning_rate = 0.5  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Type coercion
# --------------------------------------------------------------------------- #
def test_yaml_lists_become_tuples(tmp_path):
    """Sequence fields are declared as tuples so configs stay hashable/immutable."""
    path = tmp_path / "backbone.yaml"
    path.write_text(
        yaml.safe_dump({"stage_channels": [16, 32], "stride_per_stage": [2, 2]}),
        encoding="utf-8",
    )
    config = load_config(BackboneConfig, path)
    assert config.stage_channels == (16, 32)
    assert config.stride_per_stage == (2, 2)


def test_optional_field_accepts_null(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump({"powerline_notch_hz": None}), encoding="utf-8")
    assert load_config(DataConfig, path).powerline_notch_hz is None


def test_integer_yaml_value_coerced_to_float(tmp_path):
    """`window_seconds: 10` must not leave an int where a float is declared."""
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump({"window_seconds": 10}), encoding="utf-8")
    config = load_config(DataConfig, path)
    assert isinstance(config.window_seconds, float)


def test_nested_dataclass_is_parsed(tmp_path):
    path = tmp_path / "validation.yaml"
    path.write_text(
        yaml.safe_dump({"gradient_boosting": {"n_estimators": 42, "max_depth": 2}}),
        encoding="utf-8",
    )
    config = load_config(ValidationFrameworkConfig, path)
    assert config.gradient_boosting.n_estimators == 42
    assert config.gradient_boosting.max_depth == 2


# --------------------------------------------------------------------------- #
# Rejection of bad input
# --------------------------------------------------------------------------- #
def test_unknown_key_is_rejected(tmp_path):
    """A typo'd key must raise, not be silently ignored in favour of a default."""
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump({"learning_rate_": 0.01}), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(TrainingConfig, path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(TrainingConfig, tmp_path / "nope.yaml")


def test_non_mapping_yaml_is_rejected(tmp_path):
    path = tmp_path / "training.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(TrainingConfig, path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"learning_rate": -1e-3},
        {"learning_rate": 0.0},
        {"epochs": 0},
        {"batch_size": -8},
        {"weight_decay": -0.1},
        {"optimizer": "lbfgs"},
        {"loss": "logcosh"},
        {"warmup_epochs": 60},  # equals epochs, must be strictly less
        {"device": "tpu"},
        {"num_workers": -1},
    ],
)
def test_invalid_training_values_rejected(overrides):
    with pytest.raises(ConfigError):
        TrainingConfig(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"window_seconds": 0.0},          # zero-length input window
        {"window_seconds": -10.0},
        {"sampling_rate_hz": 250},        # PTB-XL ships only 100 and 500
        {"n_leads": 8},
        {"bandpass_low_hz": 45.0},        # low >= high
        {"bandpass_high_hz": 60.0},       # above Nyquist at 100 Hz
        {"normalization": "minmax"},
        {"train_frac": 0.8, "val_frac": 0.3, "test_frac": 0.2},  # sums to 1.3
        {"train_frac": 0.9, "val_frac": 0.1, "test_frac": 0.0},  # empty test split
        {"min_age_years": 95.0},          # min above max
    ],
)
def test_invalid_data_values_rejected(overrides):
    with pytest.raises(ConfigError):
        DataConfig(**overrides)


def test_notch_above_nyquist_rejected():
    """50 Hz mains cannot be notched out of a 100 Hz signal at Nyquist == 50."""
    with pytest.raises(ConfigError, match="Nyquist"):
        DataConfig(sampling_rate_hz=100, powerline_notch_hz=55.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"kernel_size": 8},                       # even kernel
        {"kernel_size": 1},                       # too small to see a QRS
        {"stage_channels": ()},
        {"stride_per_stage": (2, 2)},             # length mismatch with 4 stages
        {"dropout": 1.4},
        {"blocks_per_stage": 0},
    ],
)
def test_invalid_backbone_values_rejected(overrides):
    with pytest.raises(ConfigError):
        BackboneConfig(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"known_features": ()},                                   # nothing "known"
        {"known_features": ("qrs_duration_ms", "qrs_duration_ms")},
        {"explainer_models": ("random_forest",)},
        {"cv_folds": 1},
        {"confidence_level": 1.0},
        {"bootstrap_iterations": 10},
        {"adjust_for_covariates": ("bmi",)},
    ],
)
def test_invalid_validation_values_rejected(overrides):
    with pytest.raises(ConfigError):
        ValidationFrameworkConfig(**overrides)


def test_synthetic_cycle_must_fit_in_rr_interval():
    """A P-QRS-T cycle longer than the RR interval would make beats overlap."""
    with pytest.raises(ConfigError, match="does not fit"):
        SyntheticConfig(heart_rate_bpm_mean=200.0, heart_rate_bpm_sd=0.0)


def test_synthetic_pr_must_exceed_p_duration():
    with pytest.raises(ConfigError, match="pr_interval_ms"):
        SyntheticConfig(p_duration_ms=200.0, pr_interval_ms=160.0)


# --------------------------------------------------------------------------- #
# Derived properties
# --------------------------------------------------------------------------- #
def test_n_samples_derived_from_rate_and_duration():
    assert DataConfig(sampling_rate_hz=100, window_seconds=10.0).n_samples == 1000
    assert DataConfig(sampling_rate_hz=500, window_seconds=10.0).n_samples == 5000


def test_total_downsampling():
    assert BackboneConfig(
        stage_channels=(8, 16, 32), stride_per_stage=(2, 2, 4)
    ).total_downsampling == 16


# --------------------------------------------------------------------------- #
# Runtime: seeding and run directories
# --------------------------------------------------------------------------- #
def test_set_global_seed_is_reproducible():
    import random

    import numpy as np

    set_global_seed(123)
    first = (random.random(), np.random.rand(3).tolist())
    set_global_seed(123)
    second = (random.random(), np.random.rand(3).tolist())
    assert first == second


def test_negative_seed_rejected():
    with pytest.raises(ValueError):
        set_global_seed(-1)


def test_run_context_records_provenance_and_metrics(tmp_path):
    config = DataConfig()
    with RunContext("unit_test", tmp_path, seed=3, configs={"data": config}) as run:
        run.log({"epoch": 0, "train_loss": 12.5})
        run.log({"epoch": 1, "train_loss": 9.25})
        run.save_json("result.json", {"mae": 7.1})

    provenance = json.loads((run.dir / "config.json").read_text())
    assert provenance["seed"] == 3
    assert provenance["configs"]["data"]["sampling_rate_hz"] == 100
    assert "python" in provenance["packages"]

    lines = (run.dir / "metrics.jsonl").read_text().strip().split("\n")
    assert [json.loads(line)["epoch"] for line in lines] == [0, 1]
    assert json.loads((run.artifacts_dir / "result.json").read_text())["mae"] == 7.1


def test_run_context_serialises_numpy_values(tmp_path):
    """Metrics arrive as numpy scalars from torch/sklearn; they must not crash."""
    import numpy as np

    with RunContext("unit_test", tmp_path, seed=0) as run:
        run.log({"r2": np.float64(0.42), "counts": np.arange(3)})
    record = json.loads((run.dir / "metrics.jsonl").read_text().strip())
    assert record == {"r2": 0.42, "counts": [0, 1, 2]}
