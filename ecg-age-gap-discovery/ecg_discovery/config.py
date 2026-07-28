"""Typed, validated configuration objects for every stage of the pipeline.

Every experiment in this repository is driven by a YAML file in ``configs/``
that is parsed into one of the frozen dataclasses below. The dataclasses do two
jobs beyond holding values:

1. **They reject nonsense early.** A negative learning rate, a zero-length input
   window, or split fractions that do not sum to one raise at load time rather
   than producing a silently-wrong result eight hours into a training run.
2. **They are the unit of provenance.** Every run directory records the exact
   config it was constructed from (see :mod:`ecg_discovery.runtime`), so a
   number in the paper can always be traced back to the settings that produced
   it.

Unknown keys in a YAML file are an error, not a warning. A typo'd key
(``learning_rate_`` instead of ``learning_rate``) would otherwise be silently
ignored and the default used instead, which is exactly the class of bug that
makes a result irreproducible.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

__all__ = [
    "ConfigError",
    "DataConfig",
    "SyntheticConfig",
    "BackboneConfig",
    "TrainingConfig",
    "ValidationFrameworkConfig",
    "load_config",
    "config_to_dict",
]


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or physically implausible."""


T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _positive(value: float, name: str) -> None:
    _require(
        value is not None and value > 0,
        f"{name} must be strictly positive, got {value!r}",
    )


def _in_range(value: float, low: float, high: float, name: str) -> None:
    _require(
        value is not None and low <= value <= high,
        f"{name} must lie in [{low}, {high}], got {value!r}",
    )


def _one_of(value: Any, allowed: tuple[Any, ...], name: str) -> None:
    _require(value in allowed, f"{name} must be one of {allowed}, got {value!r}")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DataConfig:
    """How PTB-XL recordings are read, filtered, normalised and split.

    Attributes
    ----------
    ptbxl_root:
        Directory holding ``ptbxl_database.csv`` and the ``records100`` /
        ``records500`` waveform trees, as unpacked from PhysioNet.
    sampling_rate_hz:
        Which PTB-XL sampling rate to read. Only 100 and 500 exist.
    window_seconds:
        Length of the analysed window. PTB-XL recordings are exactly 10 s.
    bandpass_low_hz, bandpass_high_hz:
        Passband of the pre-filter. Below ~0.5 Hz sits baseline wander (the slow
        drift caused by patient breathing and electrode movement); above ~40 Hz
        sits muscle noise. Filtering outside this band removes artefact without
        distorting the P, QRS and T waves that the analysis depends on.
    powerline_notch_hz:
        Mains frequency to notch out (50 Hz in Europe, where PTB-XL was
        recorded), or ``None`` to disable. Note that a notch is only possible
        strictly below the Nyquist frequency: at the default 100 Hz sampling
        rate, Nyquist *is* 50 Hz, so the notch must be disabled - which is
        harmless, since the 0.5-40 Hz bandpass already excludes the mains band.
    normalization:
        ``per_lead_train_stats`` z-scores each lead using mean/std computed on
        the training split only, so no test-set statistics leak into the model's
        input scale. ``per_recording`` normalises each recording independently
        (removes inter-patient amplitude information, which may itself be
        age-informative). ``none`` leaves millivolts untouched.
    min_age_years, max_age_years, drop_age_sentinel_300:
        PTB-XL's anonymisation records every patient older than 89 with the
        sentinel age 300. Those recordings are dropped rather than clipped,
        because inventing an age for them would corrupt both the regression
        target and every age-gap residual computed downstream.
    train_frac, val_frac, test_frac:
        Split proportions, applied at the **patient** level. PTB-XL contains
        patients with more than one recording; splitting by recording would put
        the same heart in train and test and silently inflate measured accuracy.
    """

    ptbxl_root: str = "data/ptbxl"
    sampling_rate_hz: int = 100
    window_seconds: float = 10.0
    n_leads: int = 12

    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    bandpass_order: int = 3
    powerline_notch_hz: float | None = None
    notch_quality: float = 30.0

    normalization: str = "per_lead_train_stats"

    min_age_years: float = 18.0
    max_age_years: float = 89.0
    drop_age_sentinel_300: bool = True

    split_seed: int = 1337
    train_frac: float = 0.7
    val_frac: float = 0.1
    test_frac: float = 0.2

    def __post_init__(self) -> None:
        _one_of(self.sampling_rate_hz, (100, 500), "sampling_rate_hz")
        _positive(self.window_seconds, "window_seconds")
        _require(self.n_leads == 12, f"n_leads must be 12 for PTB-XL, got {self.n_leads}")

        _positive(self.bandpass_low_hz, "bandpass_low_hz")
        _positive(self.bandpass_high_hz, "bandpass_high_hz")
        _require(
            self.bandpass_low_hz < self.bandpass_high_hz,
            "bandpass_low_hz must be below bandpass_high_hz "
            f"(got {self.bandpass_low_hz} >= {self.bandpass_high_hz})",
        )
        nyquist = self.sampling_rate_hz / 2.0
        _require(
            self.bandpass_high_hz < nyquist,
            f"bandpass_high_hz ({self.bandpass_high_hz}) must be below the Nyquist "
            f"frequency ({nyquist}) for sampling_rate_hz={self.sampling_rate_hz}",
        )
        _require(1 <= self.bandpass_order <= 10, "bandpass_order must be in [1, 10]")

        if self.powerline_notch_hz is not None:
            _positive(self.powerline_notch_hz, "powerline_notch_hz")
            _require(
                self.powerline_notch_hz < nyquist,
                f"powerline_notch_hz ({self.powerline_notch_hz}) must be below the "
                f"Nyquist frequency ({nyquist}); disable it with `null` at 100 Hz "
                "if the mains frequency exceeds it",
            )
            _positive(self.notch_quality, "notch_quality")

        _one_of(
            self.normalization,
            ("per_lead_train_stats", "per_recording", "none"),
            "normalization",
        )

        _require(
            0 < self.min_age_years < self.max_age_years,
            f"require 0 < min_age_years < max_age_years, got "
            f"{self.min_age_years} and {self.max_age_years}",
        )

        for name in ("train_frac", "val_frac", "test_frac"):
            _in_range(getattr(self, name), 0.0, 1.0, name)
        total = self.train_frac + self.val_frac + self.test_frac
        _require(
            abs(total - 1.0) < 1e-6,
            f"train/val/test fractions must sum to 1.0, got {total:.6f}",
        )
        _positive(self.train_frac, "train_frac")
        _positive(self.test_frac, "test_frac")

    @property
    def n_samples(self) -> int:
        """Number of time samples in one recording window."""
        return int(round(self.window_seconds * self.sampling_rate_hz))


# --------------------------------------------------------------------------- #
# Synthetic cohort
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters of the procedurally generated ECG cohort used for ground truth.

    The synthetic generator exists so that every downstream component can be
    checked against an answer we constructed ourselves. Two age effects are
    injected on purpose, and they are designed to fall on opposite sides of the
    Phase 8 validation framework:

    ``qrs_widening_ms_per_decade``
        Older synthetic patients get a wider QRS complex. This is *entirely*
        mediated by a classically measurable interval, so the residual
        decomposition must attribute essentially all of it to known features.

    ``t_wave_skew_per_decade``
        Older synthetic patients get an asymmetric (skewed) T wave of unchanged
        duration and amplitude. No timing interval captures it, so the
        decomposition must leave it in the unexplained residual.

    Getting both directions right is the correctness check for the method that
    the paper's central claim rests on.

    Interval attributes are baseline values at age 40; the per-decade terms are
    applied relative to that reference age.
    """

    n_recordings: int = 2000
    sampling_rate_hz: int = 100
    duration_seconds: float = 10.0
    n_leads: int = 12
    seed: int = 7

    age_min_years: float = 20.0
    age_max_years: float = 89.0
    p_female: float = 0.5

    heart_rate_bpm_mean: float = 70.0
    heart_rate_bpm_sd: float = 10.0
    hr_sinus_arrhythmia_frac: float = 0.03

    p_duration_ms: float = 100.0
    p_amplitude_mv: float = 0.15
    pr_interval_ms: float = 160.0
    qrs_duration_ms: float = 90.0
    r_amplitude_mv: float = 1.2
    q_amplitude_mv: float = -0.1
    s_amplitude_mv: float = -0.25
    st_segment_ms: float = 100.0
    t_duration_ms: float = 160.0
    t_amplitude_mv: float = 0.3

    qrs_widening_ms_per_decade: float = 4.0
    t_wave_skew_per_decade: float = 0.06
    hr_change_bpm_per_decade: float = -1.5

    noise_mv_sd: float = 0.02
    baseline_wander_mv: float = 0.05
    baseline_wander_hz: float = 0.3
    powerline_mv: float = 0.005
    powerline_hz: float = 50.0
    label_noise_years: float = 2.0

    #: Reference age at which the baseline morphology parameters apply.
    reference_age_years: float = 40.0

    def __post_init__(self) -> None:
        _require(self.n_recordings > 0, "n_recordings must be positive")
        _positive(self.sampling_rate_hz, "sampling_rate_hz")
        _positive(self.duration_seconds, "duration_seconds")
        _require(self.n_leads == 12, "n_leads must be 12")
        _require(
            0 < self.age_min_years < self.age_max_years,
            "require 0 < age_min_years < age_max_years",
        )
        _in_range(self.p_female, 0.0, 1.0, "p_female")

        _positive(self.heart_rate_bpm_mean, "heart_rate_bpm_mean")
        _require(self.heart_rate_bpm_sd >= 0, "heart_rate_bpm_sd must be non-negative")
        _in_range(self.hr_sinus_arrhythmia_frac, 0.0, 0.5, "hr_sinus_arrhythmia_frac")

        for name in ("p_duration_ms", "pr_interval_ms", "qrs_duration_ms",
                     "st_segment_ms", "t_duration_ms"):
            _positive(getattr(self, name), name)
        _require(
            self.pr_interval_ms > self.p_duration_ms,
            "pr_interval_ms (P onset -> QRS onset) must exceed p_duration_ms; "
            f"got {self.pr_interval_ms} <= {self.p_duration_ms}",
        )

        for name in ("noise_mv_sd", "baseline_wander_mv", "powerline_mv",
                     "label_noise_years"):
            _require(getattr(self, name) >= 0, f"{name} must be non-negative")

        # The whole cardiac cycle must fit inside the shortest plausible RR
        # interval, or beats would overlap and the "known" fiducial points would
        # stop being meaningful.
        fastest_hr = self.heart_rate_bpm_mean + 4 * self.heart_rate_bpm_sd
        min_rr_ms = 60_000.0 / max(fastest_hr, 1e-6)
        cycle_ms = (
            self.pr_interval_ms
            + self.qrs_duration_ms
            + self.st_segment_ms
            + self.t_duration_ms
        )
        _require(
            cycle_ms < min_rr_ms,
            f"synthetic P-QRS-T cycle ({cycle_ms:.0f} ms) does not fit inside the "
            f"shortest plausible RR interval ({min_rr_ms:.0f} ms); lower the heart "
            "rate or shorten the intervals",
        )

    @property
    def n_samples(self) -> int:
        """Number of time samples per synthetic recording."""
        return int(round(self.duration_seconds * self.sampling_rate_hz))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BackboneConfig:
    """Architecture of the from-scratch 1D CNN age regressor.

    No field here refers to pretrained weights, because none are ever loaded:
    the network is randomly initialised and trained only on the dataset in this
    repository. See :mod:`ecg_discovery.models.ecg_age_regressor` for why the
    network is deliberately small.
    """

    in_channels: int = 12
    stem_channels: int = 32
    stage_channels: tuple[int, ...] = (32, 64, 64, 128)
    blocks_per_stage: int = 2
    kernel_size: int = 7
    stride_per_stage: tuple[int, ...] = (2, 2, 2, 2)
    dropout: float = 0.3
    use_sex_input: bool = True
    head_hidden: int = 64

    def __post_init__(self) -> None:
        _positive(self.in_channels, "in_channels")
        _positive(self.stem_channels, "stem_channels")
        _require(len(self.stage_channels) > 0, "stage_channels must be non-empty")
        _require(
            all(c > 0 for c in self.stage_channels),
            "every entry in stage_channels must be positive",
        )
        _require(
            len(self.stride_per_stage) == len(self.stage_channels),
            f"stride_per_stage has {len(self.stride_per_stage)} entries but "
            f"stage_channels has {len(self.stage_channels)}; they must match",
        )
        _require(
            all(s >= 1 for s in self.stride_per_stage),
            "every stride must be >= 1",
        )
        _positive(self.blocks_per_stage, "blocks_per_stage")
        _require(
            self.kernel_size >= 3 and self.kernel_size % 2 == 1,
            f"kernel_size must be an odd integer >= 3, got {self.kernel_size}",
        )
        _in_range(self.dropout, 0.0, 1.0, "dropout")
        _positive(self.head_hidden, "head_hidden")

    @property
    def total_downsampling(self) -> int:
        """Product of all stage strides: input samples per output timestep."""
        out = 1
        for s in self.stride_per_stage:
            out *= int(s)
        return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation settings and run-logging behaviour for the age regressor."""

    experiment_name: str = "ecg_age_regressor"
    seed: int = 0
    device: str = "auto"

    epochs: int = 60
    batch_size: int = 128
    num_workers: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 2
    grad_clip_norm: float | None = 5.0

    loss: str = "huber"
    huber_delta: float = 5.0

    early_stopping_patience: int = 12
    checkpoint_metric: str = "val_mae"
    log_every_n_steps: int = 20
    scatter_every_n_epochs: int = 10

    runs_dir: str = "runs"

    def __post_init__(self) -> None:
        _require(bool(self.experiment_name), "experiment_name must be non-empty")
        _one_of(self.device, ("auto", "cpu", "cuda", "mps"), "device")
        _positive(self.epochs, "epochs")
        _positive(self.batch_size, "batch_size")
        _require(self.num_workers >= 0, "num_workers must be non-negative")
        _positive(self.learning_rate, "learning_rate")
        _require(self.weight_decay >= 0, "weight_decay must be non-negative")
        _one_of(self.optimizer, ("adamw", "adam", "sgd"), "optimizer")
        _one_of(self.lr_scheduler, ("cosine", "none"), "lr_scheduler")
        _require(
            0 <= self.warmup_epochs < self.epochs,
            f"warmup_epochs must be in [0, epochs); got {self.warmup_epochs} "
            f"with epochs={self.epochs}",
        )
        if self.grad_clip_norm is not None:
            _positive(self.grad_clip_norm, "grad_clip_norm")
        _one_of(self.loss, ("huber", "mse", "mae"), "loss")
        _positive(self.huber_delta, "huber_delta")
        _positive(self.early_stopping_patience, "early_stopping_patience")
        _one_of(self.checkpoint_metric, ("val_mae", "val_loss"), "checkpoint_metric")
        _positive(self.log_every_n_steps, "log_every_n_steps")
        _positive(self.scatter_every_n_epochs, "scatter_every_n_epochs")


# --------------------------------------------------------------------------- #
# Validation framework
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GradientBoostingConfig:
    """Hyperparameters for the non-linear explainer in the residual decomposition."""

    n_estimators: int = 300
    max_depth: int = 3
    learning_rate: float = 0.05
    subsample: float = 0.9

    def __post_init__(self) -> None:
        _positive(self.n_estimators, "gradient_boosting.n_estimators")
        _positive(self.max_depth, "gradient_boosting.max_depth")
        _positive(self.learning_rate, "gradient_boosting.learning_rate")
        _require(
            0 < self.subsample <= 1.0,
            f"gradient_boosting.subsample must lie in (0, 1], got {self.subsample}",
        )


@dataclass(frozen=True)
class ValidationFrameworkConfig:
    """Settings for separating rediscovery from discovery (Phases 8 and 9).

    ``known_features`` is the list of classically measurable ECG quantities that
    define what counts as "already known". The central number this framework
    reports - the fraction of age-gap residual variance those features explain -
    is only meaningful relative to this list, which is why it is configuration
    rather than a hardcoded constant, and why ``docs/limitations.md`` states
    plainly that a richer known-feature set could explain more.
    """

    known_features: tuple[str, ...] = (
        "heart_rate_bpm",
        "qrs_duration_ms",
        "pr_interval_ms",
        "qt_interval_ms",
        "qtc_bazett_ms",
    )
    explainer_models: tuple[str, ...] = ("linear", "gradient_boosting")
    gradient_boosting: GradientBoostingConfig = field(default_factory=GradientBoostingConfig)
    adjust_for_covariates: tuple[str, ...] = ("age", "sex")

    cv_folds: int = 5
    cv_repeats: int = 3
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    seed: int = 0

    diagnostic_superclasses: tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")
    classifier: str = "logistic_regression"
    classifier_max_iter: int = 2000
    standardize_features: bool = True

    def __post_init__(self) -> None:
        _require(len(self.known_features) > 0, "known_features must be non-empty")
        _require(
            len(set(self.known_features)) == len(self.known_features),
            f"known_features contains duplicates: {self.known_features}",
        )
        _require(len(self.explainer_models) > 0, "explainer_models must be non-empty")
        for model in self.explainer_models:
            _one_of(model, ("linear", "gradient_boosting"), "explainer_models entry")
        for cov in self.adjust_for_covariates:
            _one_of(cov, ("age", "sex"), "adjust_for_covariates entry")
        _require(self.cv_folds >= 2, f"cv_folds must be >= 2, got {self.cv_folds}")
        _positive(self.cv_repeats, "cv_repeats")
        _require(
            self.bootstrap_iterations >= 100,
            "bootstrap_iterations must be >= 100 for a usable interval, got "
            f"{self.bootstrap_iterations}",
        )
        _require(
            0.5 < self.confidence_level < 1.0,
            f"confidence_level must lie in (0.5, 1.0), got {self.confidence_level}",
        )
        _require(
            len(self.diagnostic_superclasses) > 0,
            "diagnostic_superclasses must be non-empty",
        )
        _one_of(self.classifier, ("logistic_regression",), "classifier")
        _positive(self.classifier_max_iter, "classifier_max_iter")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _coerce(value: Any, annotation: Any, path: str) -> Any:
    """Coerce a YAML-parsed value to the type a dataclass field declares.

    Handles the three cases that actually occur in this repository: nested
    dataclasses (parsed from nested mappings), tuples (YAML gives lists), and
    optional scalars (YAML ``null``).
    """
    origin = get_origin(annotation)

    # Optional[X] / X | None
    if origin is not None and type(None) in get_args(annotation):
        if value is None:
            return None
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) == 1:
            return _coerce(value, inner[0], path)
        return value

    if is_dataclass(annotation) and isinstance(value, dict):
        return _from_dict(annotation, value, path)

    if origin in (tuple,):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path} must be a list, got {type(value).__name__}")
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], f"{path}[]") for v in value)
        return tuple(value)

    if origin in (list,):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path} must be a list, got {type(value).__name__}")
        args = get_args(annotation)
        return [_coerce(v, args[0], f"{path}[]") for v in value] if args else list(value)

    if annotation is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)

    return value


def _from_dict(cls: type[T], data: dict[str, Any], path: str = "") -> T:
    """Build a dataclass from a mapping, rejecting unknown keys."""
    if not isinstance(data, dict):
        raise ConfigError(
            f"expected a mapping for {cls.__name__}{' at ' + path if path else ''}, "
            f"got {type(data).__name__}"
        )
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"unknown key(s) for {cls.__name__}: {', '.join(unknown)}. "
            f"Valid keys are: {', '.join(sorted(known))}"
        )
    # `from __future__ import annotations` makes ``Field.type`` a string, so the
    # annotations have to be resolved to real types before they can be matched
    # against ``tuple[...]``, ``X | None`` and nested dataclasses.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        prefix = f"{path}.{name}" if path else name
        kwargs[name] = _coerce(value, hints[name], prefix)
    return cls(**kwargs)


def load_config(cls: type[T], path: str | Path) -> T:
    """Load and validate a YAML configuration file into ``cls``.

    Parameters
    ----------
    cls:
        One of the config dataclasses in this module.
    path:
        Path to the YAML file.

    Raises
    ------
    ConfigError
        If the file is missing, is not a mapping, contains unknown keys, or
        holds values that fail the dataclass's physical-plausibility checks.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping, got {type(raw).__name__}")
    return _from_dict(cls, raw, path=str(path))


def config_to_dict(config: Any) -> dict[str, Any]:
    """Convert a config dataclass to a JSON/YAML-serialisable dictionary.

    Used when writing the exact configuration into a run directory, so any
    reported number can be traced to the settings that produced it.
    """
    if not is_dataclass(config):
        raise ConfigError(f"expected a dataclass, got {type(config).__name__}")
    return dataclasses.asdict(config)
