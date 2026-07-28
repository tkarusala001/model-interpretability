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
    "SignalProcessingConfig",
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
        Which PTB-XL sampling rate the *model* consumes. Only 100 and 500 exist.
        100 Hz is the standard choice for PTB-XL benchmarks and is ample for a
        CNN predicting age.
    interval_sampling_rate_hz:
        Which sampling rate the *classical interval measurements* are made at.
        This is deliberately separate from, and higher than, the model's rate.
        One sample at 100 Hz is 10 ms, which is coarser than the interval
        differences the validation framework must resolve (the injected effect is
        a few ms per decade). Measuring intervals at 100 Hz would systematically
        understate how much variance the *known* features explain, and that error
        points in the dangerous direction: it inflates the "unexplained" residual
        and biases the analysis toward a false discovery claim. Interval features
        are therefore always computed at 500 Hz. See docs/validation_methodology.md.
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
    interval_sampling_rate_hz: int = 500
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
        _one_of(self.interval_sampling_rate_hz, (100, 500), "interval_sampling_rate_hz")
        _require(
            self.interval_sampling_rate_hz >= self.sampling_rate_hz,
            "interval_sampling_rate_hz must be at least sampling_rate_hz; measuring "
            "intervals more coarsely than the model's own input would understate "
            "what known features explain and bias the decomposition toward a false "
            f"discovery claim (got {self.interval_sampling_rate_hz} < {self.sampling_rate_hz})",
        )
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
    checked against an answer we constructed ourselves.

    **The generative story.** Each synthetic patient has a chronological age -
    the label the model is trained to predict - and two *independent* latent
    offsets that shift the apparent age of their heart. Morphology is driven by
    the offset ages, not by chronological age, so a model that reads the
    waveform correctly will disagree with the label by an amount determined by
    those offsets. That disagreement is the synthetic "ECG age gap", and unlike
    label noise it has a known cause that downstream analysis can be scored
    against.

    The two offsets exist because the Phase 8 framework's whole job is telling
    them apart:

    ``known_age_offset_sd_years``
        Drives QRS widening and heart-rate change - effects that are *entirely*
        mediated by classically measurable intervals. The residual decomposition
        must attribute this portion to known features.

    ``unexplained_age_offset_sd_years``
        Drives T-wave *skew*: the T wave becomes asymmetric while keeping the
        exact same onset, offset, duration and peak amplitude. No timing
        interval captures it by construction, so the decomposition must leave
        this portion in the unexplained residual.

    Setting one SD to zero produces the two extreme cases the Phase 8 tests
    require (~100% explained, and ~0% explained). Getting both directions right
    is the correctness check that the paper's central claim rests on.

    Interval attributes are baseline values at ``reference_age_years``; the
    per-decade terms are applied relative to that reference.

    Note that the latent offsets are recorded in the cohort metadata as ground
    truth *for tests only*. No component of the pipeline may read them - doing
    so would be reading the answer key.
    """

    n_recordings: int = 2000
    sampling_rate_hz: int = 500
    duration_seconds: float = 10.0
    n_leads: int = 12
    seed: int = 7

    age_min_years: float = 20.0
    age_max_years: float = 89.0
    p_female: float = 0.5
    repeat_patient_frac: float = 0.1

    # Latent ECG-age offsets: the cause of the synthetic age gap.
    known_age_offset_sd_years: float = 6.0
    unexplained_age_offset_sd_years: float = 6.0

    heart_rate_bpm_mean: float = 70.0
    heart_rate_bpm_sd: float = 10.0
    hr_sinus_arrhythmia_frac: float = 0.03

    p_duration_ms: float = 100.0
    p_amplitude_mv: float = 0.15
    pr_interval_ms: float = 160.0
    # Between-person spread in PR. Without it PR would be identical in every
    # recording, so it would contribute no variance to the known-feature set and
    # its measurement accuracy could not be validated at all (a correlation
    # against a constant is undefined). Real PR intervals vary substantially
    # between healthy people.
    pr_interval_sd_ms: float = 20.0
    qrs_duration_ms: float = 90.0
    r_amplitude_mv: float = 1.2
    q_amplitude_mv: float = -0.1
    s_amplitude_mv: float = -0.25
    st_segment_ms: float = 80.0
    t_duration_ms: float = 160.0
    t_amplitude_mv: float = 0.3

    # Injected age effects, per decade away from reference_age_years.
    qrs_widening_ms_per_decade: float = 5.0     # mediated by a known interval
    hr_change_bpm_per_decade: float = -2.0      # mediated by a known interval
    t_wave_skew_per_decade: float = 0.10        # NOT captured by any timing interval

    noise_mv_sd: float = 0.02
    baseline_wander_mv: float = 0.05
    baseline_wander_hz: float = 0.3
    powerline_mv: float = 0.005
    powerline_hz: float = 50.0
    lead_amplitude_jitter: float = 0.10

    #: Reference age at which the baseline morphology parameters apply.
    reference_age_years: float = 40.0

    # Synthetic stand-in for PTB-XL's diagnostic superclasses, used to exercise
    # the Phase 9 discovery experiment. `diagnostic_link_source` selects which
    # latent offset (if any) raises the probability of an abnormal label, so the
    # experiment can be verified to detect a link when one exists by
    # construction AND to report nothing when none does. This is machinery
    # validation only; it says nothing about real ECGs.
    diagnostic_link_source: str = "unexplained"   # unexplained | known | none
    diagnostic_link_strength: float = 0.8         # log-odds per SD of the source offset
    diagnostic_abnormal_base_rate: float = 0.45   # PTB-XL is roughly 44% non-NORM

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
        _in_range(self.repeat_patient_frac, 0.0, 1.0, "repeat_patient_frac")
        _require(
            self.known_age_offset_sd_years >= 0,
            "known_age_offset_sd_years must be non-negative",
        )
        _require(
            self.unexplained_age_offset_sd_years >= 0,
            "unexplained_age_offset_sd_years must be non-negative",
        )

        _one_of(
            self.diagnostic_link_source,
            ("unexplained", "known", "none"),
            "diagnostic_link_source",
        )
        _require(
            self.diagnostic_link_strength >= 0,
            "diagnostic_link_strength must be non-negative",
        )
        _require(
            0.0 < self.diagnostic_abnormal_base_rate < 1.0,
            "diagnostic_abnormal_base_rate must lie in (0, 1), got "
            f"{self.diagnostic_abnormal_base_rate}",
        )

        _positive(self.heart_rate_bpm_mean, "heart_rate_bpm_mean")
        _require(self.heart_rate_bpm_sd >= 0, "heart_rate_bpm_sd must be non-negative")
        _in_range(self.hr_sinus_arrhythmia_frac, 0.0, 0.5, "hr_sinus_arrhythmia_frac")

        for name in ("p_duration_ms", "pr_interval_ms", "qrs_duration_ms",
                     "st_segment_ms", "t_duration_ms"):
            _positive(getattr(self, name), name)
        _require(
            self.pr_interval_sd_ms >= 0, "pr_interval_sd_ms must be non-negative"
        )
        _require(
            self.pr_interval_ms > self.p_duration_ms,
            "pr_interval_ms (P onset -> QRS onset) must exceed p_duration_ms; "
            f"got {self.pr_interval_ms} <= {self.p_duration_ms}",
        )

        for name in ("noise_mv_sd", "baseline_wander_mv", "powerline_mv",
                     "lead_amplitude_jitter"):
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
# Signal processing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SignalProcessingConfig:
    """Parameters of the from-scratch QRS detection and wave delineation stages.

    Defaults follow Pan & Tompkins (1985) where that algorithm specifies a
    value, and standard electrophysiological ranges elsewhere. The values that
    encode real physiology are called out below, because changing them changes
    what the detector considers a plausible heartbeat.

    Attributes
    ----------
    detection_lead:
        Which lead R-peak detection runs on. Lead II is the conventional rhythm
        lead: its axis is closest to the heart's normal depolarisation
        direction, so the R wave is tall and positive there in most people.
    polarity:
        Whether the R wave is expected to deflect up, down, or whichever is
        larger (``auto``). Needed because the same beat is positive in lead II
        and negative in aVR and V1.
    qrs_band_low_hz, qrs_band_high_hz:
        Passband isolating QRS energy. The QRS complex is the fastest normal
        feature of the ECG, so 5-15 Hz suppresses P and T waves (slower) and
        muscle noise (faster) while keeping the complex itself.
    refine_band_low_hz, refine_band_high_hz:
        A second, wider passband used only to locate the R peak precisely once a
        beat has been found. The 5-15 Hz detection band distorts the QRS shape,
        so peak position is measured on a morphology-preserving 0.5-40 Hz
        version instead.
    integration_window_ms:
        Width of the moving-window integrator. Pan & Tompkins set this to
        approximately the width of the widest normal QRS complex (150 ms): wide
        enough to merge the Q, R and S deflections into one energy bump, narrow
        enough not to merge adjacent beats.
    refractory_ms:
        Minimum interval between accepted beats. The heart is physiologically
        incapable of re-depolarising within ~200 ms, so any second detection
        inside that window is an artefact.
    t_wave_discrimination_ms, t_wave_slope_fraction:
        Below this RR interval, a candidate is checked against the T-wave rule:
        a tall T wave can exceed the detection threshold on amplitude, but rises
        more slowly than a QRS complex, so a candidate whose slope falls below
        ``t_wave_slope_fraction`` of the previous beat's is rejected.

        Pan & Tompkins use 0.5; this project defaults to 0.7 on the evidence of
        a sweep over synthetic cohorts spanning 45-180 bpm and T-wave amplitudes
        up to 83% of the R wave. Raising the fraction from 0.5 to 0.7 cost
        nothing in sensitivity in any scenario tested (including 180 bpm, where
        the rule fires on genuine consecutive beats) while eliminating T-wave
        false positives for hyperacute T waves at 50% of R amplitude.

        Known limitation: for a T wave at ~83% of R amplitude *and* comparable
        steepness, the rule still admits some T waves (positive predictive value
        plateaus near 0.97 regardless of this parameter). Amplitude-and-slope
        thresholding cannot separate two waves that differ in neither. This was
        tuned on synthetic T waves, which are smooth raised cosines rather than
        the peaked "tented" T waves of real hyperkalaemia, so it must be
        re-validated on PTB-XL. See docs/limitations.md.
    searchback_factor:
        If no beat is found within this multiple of the recent average RR
        interval, the detector goes back over the interval with a halved
        threshold. This recovers beats that a transiently high threshold missed.
    edge_guard_ms:
        Detections closer than this to either end of the recording are
        discarded. A recording starts and stops mid-rhythm, so its first and
        last complexes are usually cut in half; a truncated complex has no
        well-defined peak and, if kept, injects a spuriously short RR interval
        that biases heart rate. 50 ms is roughly half a maximal normal QRS
        width, so a beat closer than that to the boundary cannot be complete.
        The value was also chosen empirically: sweeping it over the synthetic
        cohort, 50 ms is the point at which both sensitivity and positive
        predictive value reach 1.000 at 100 Hz and 500 Hz. Below it truncated
        complexes survive as false positives; above it genuine beats start being
        discarded (sensitivity falls to 0.997 by 75 ms).
    match_tolerance_ms:
        How close a detected peak must be to a reference peak to count as the
        same beat when scoring detection performance.
    """

    detection_lead: str = "II"
    polarity: str = "auto"

    qrs_band_low_hz: float = 5.0
    qrs_band_high_hz: float = 15.0
    filter_order: int = 2
    refine_band_low_hz: float = 0.5
    refine_band_high_hz: float = 40.0

    integration_window_ms: float = 150.0
    refractory_ms: float = 200.0
    t_wave_discrimination_ms: float = 360.0
    t_wave_slope_fraction: float = 0.7

    threshold_adaptation: float = 0.125
    threshold_fraction: float = 0.25
    searchback_factor: float = 1.66
    rr_low_factor: float = 0.92
    rr_high_factor: float = 1.16
    rr_history: int = 8
    learning_seconds: float = 2.0
    refine_window_ms: float = 100.0
    edge_guard_ms: float = 50.0

    match_tolerance_ms: float = 50.0

    # -- Wave delineation (Phase 3) ------------------------------------------
    # These heuristics locate the P and T waves relative to each detected R
    # peak. They are an approximation, not clinical-grade delineation; the
    # limitations are set out in wave_delineation.py and docs/limitations.md.
    delineation_lead: str = "II"
    qrs_boundary_source: str = "vector_magnitude"
    qrs_search_ms: float = 120.0
    qrs_envelope_smooth_ms: float = 16.0
    qrs_boundary_threshold: float = 0.10
    quiet_run_ms: float = 12.0

    baseline_window_ms: float = 40.0

    p_search_min_ms: float = 40.0
    p_search_max_ms: float = 320.0
    p_boundary_threshold: float = 0.20
    p_min_amplitude_mv: float = 0.02

    t_search_min_ms: float = 40.0
    t_search_frac_rr: float = 0.65
    t_search_max_ms: float = 500.0
    t_boundary_threshold: float = 0.15
    t_min_amplitude_mv: float = 0.04
    t_offset_method: str = "tangent"

    # -- Interval feature aggregation ----------------------------------------
    aggregation: str = "median"
    min_beats_for_features: int = 3

    def __post_init__(self) -> None:
        _one_of(self.polarity, ("auto", "positive", "negative"), "polarity")
        _require(bool(self.detection_lead), "detection_lead must be non-empty")

        for low, high, name in (
            (self.qrs_band_low_hz, self.qrs_band_high_hz, "qrs_band"),
            (self.refine_band_low_hz, self.refine_band_high_hz, "refine_band"),
        ):
            _positive(low, f"{name}_low_hz")
            _require(
                low < high,
                f"{name}_low_hz must be below {name}_high_hz, got {low} >= {high}",
            )
        _require(1 <= self.filter_order <= 10, "filter_order must be in [1, 10]")

        for name in ("integration_window_ms", "refractory_ms",
                     "t_wave_discrimination_ms", "learning_seconds",
                     "refine_window_ms", "match_tolerance_ms"):
            _positive(getattr(self, name), name)
        _require(self.edge_guard_ms >= 0, "edge_guard_ms must be non-negative")

        _in_range(self.threshold_adaptation, 0.0, 1.0, "threshold_adaptation")
        _in_range(self.threshold_fraction, 0.0, 1.0, "threshold_fraction")
        _in_range(self.t_wave_slope_fraction, 0.0, 1.0, "t_wave_slope_fraction")
        _require(
            self.searchback_factor > 1.0,
            "searchback_factor must exceed 1.0; a value at or below 1 would "
            f"trigger search-back on every beat (got {self.searchback_factor})",
        )
        _require(
            0 < self.rr_low_factor < 1.0 < self.rr_high_factor,
            "require 0 < rr_low_factor < 1 < rr_high_factor, got "
            f"{self.rr_low_factor} and {self.rr_high_factor}",
        )
        _require(self.rr_history >= 2, f"rr_history must be >= 2, got {self.rr_history}")
        _require(
            self.refractory_ms < self.t_wave_discrimination_ms,
            "refractory_ms must be shorter than t_wave_discrimination_ms; "
            "otherwise the T-wave rule could never fire",
        )

        # -- Delineation ------------------------------------------------------
        _require(bool(self.delineation_lead), "delineation_lead must be non-empty")
        _one_of(
            self.qrs_boundary_source,
            ("vector_magnitude", "lead"),
            "qrs_boundary_source",
        )
        _one_of(self.t_offset_method, ("tangent", "threshold"), "t_offset_method")
        for name in ("qrs_search_ms", "qrs_envelope_smooth_ms", "quiet_run_ms",
                     "baseline_window_ms", "p_search_min_ms", "p_search_max_ms",
                     "t_search_min_ms", "t_search_max_ms"):
            _positive(getattr(self, name), name)
        for name in ("qrs_boundary_threshold", "p_boundary_threshold",
                     "t_boundary_threshold", "t_search_frac_rr"):
            _require(
                0.0 < getattr(self, name) < 1.0,
                f"{name} must lie in (0, 1), got {getattr(self, name)}",
            )
        _require(
            self.p_search_min_ms < self.p_search_max_ms,
            "p_search_min_ms must be below p_search_max_ms",
        )
        _require(
            self.t_search_min_ms < self.t_search_max_ms,
            "t_search_min_ms must be below t_search_max_ms",
        )
        for name in ("p_min_amplitude_mv", "t_min_amplitude_mv"):
            _require(getattr(self, name) >= 0, f"{name} must be non-negative")

        _one_of(self.aggregation, ("median", "mean"), "aggregation")
        _require(
            self.min_beats_for_features >= 1,
            "min_beats_for_features must be at least 1",
        )


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
