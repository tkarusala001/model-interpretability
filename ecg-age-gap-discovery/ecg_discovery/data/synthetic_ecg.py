"""Procedurally generated 12-lead ECGs with known fiducial points and known age effects.

This module has no external data dependency: it runs anywhere, immediately.

WHY IT EXISTS
-------------
Every claim this project makes depends on machinery - R-peak detection, wave
delineation, interval measurement, attribution, residual decomposition - whose
correctness cannot be checked on real ECGs, because on real ECGs nobody knows
the true answer. So the whole pipeline is first validated against a cohort we
constructed ourselves, where the right answer is known by construction.

THE GENERATIVE STORY
--------------------
Each synthetic patient has a **chronological age** (the label the regression
model is trained to predict) and two **independent latent offsets** that shift
the apparent age of their heart::

    qrs_age = chronological_age + known_offset
    t_age   = chronological_age + unexplained_offset

Beat morphology is driven by the offset ages, never by chronological age
directly. A model that reads the waveform correctly therefore disagrees with the
label, and that disagreement - the synthetic ECG age gap - has a known cause.
This is deliberately *not* label noise: noise would make the age gap
unpredictable from anything, and the entire Phase 8 analysis would have nothing
to find.

The two offsets sit on opposite sides of the validation framework:

``known_offset``
    Widens the QRS complex and changes heart rate. Both are classically
    measurable, so the residual decomposition must attribute this share of the
    age gap to known features. This is *rediscovery*.

``unexplained_offset``
    Skews the T wave: the wave becomes asymmetric while keeping an identical
    onset, offset, duration and peak amplitude (see :func:`_bump`). No timing
    interval can see it, so the decomposition must leave this share unexplained.
    This is the synthetic analogue of a *discovery candidate*.

Setting either standard deviation to zero yields the two extreme cases that the
Phase 8 correctness tests require.

WAVEFORM MODEL
--------------
Each beat is a sum of three compact-support components - P wave, QRS complex,
T wave - each built from raised-cosine bumps that are exactly zero outside their
own interval. Compact support is the key property: it means a wave's onset and
offset are exact known quantities rather than "wherever a Gaussian tail becomes
small", so a measured interval can be compared against ground truth without
arguing about thresholds.

The 12 leads are linear projections of those three components. Limb leads III,
aVR, aVL and aVF are *derived* from leads I and II via the standard Einthoven
and Goldberger relations, so the generated data satisfies the same algebraic
constraints as a real 12-lead recording (III = II - I, and so on). That holds
exactly in the clean signal; per-lead noise breaks it, as it does in reality.

WHAT THIS IS NOT
----------------
This is not a physiological simulator. It has no conduction model, no realistic
pathology and no inter-lead timing dispersion. It is a test fixture with known
answers, and no clinical conclusion should be drawn from it. Its only job is to
tell us whether our measurement code returns the numbers we put in.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from ecg_discovery.config import SyntheticConfig, config_to_dict

__all__ = [
    "LEAD_NAMES",
    "DIAGNOSTIC_CLASSES",
    "BeatFiducials",
    "SyntheticRecording",
    "SyntheticCohort",
    "generate_recording",
    "generate_cohort",
    "save_cohort",
    "load_cohort",
]

#: Standard 12-lead order, matching PTB-XL's channel order.
LEAD_NAMES: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

#: Synthetic stand-in for PTB-XL's diagnostic superclasses, same names and order.
DIAGNOSTIC_CLASSES: tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")

#: Reference heart rate at which ``t_duration_ms`` applies, in beats per minute.
_REFERENCE_HR_BPM = 70.0

# QRS internal geometry as fractions of the total QRS width. The Q, R and S
# lobes tile [0, 1] exactly, so the complex spans precisely `qrs_duration_ms`
# and the R peak sits at a known offset from QRS onset.
_Q_SPAN = (0.00, 0.20)
_R_SPAN = (0.20, 0.65)
_S_SPAN = (0.65, 1.00)
#: Position of the R peak within the QRS complex, as a fraction of its width.
_R_PEAK_FRAC = (_R_SPAN[0] + _R_SPAN[1]) / 2.0  # 0.425

# Base lead projection weights for leads I and II, and for the six precordial
# leads, given separately for each wave component. Values are chosen so the
# result reads like a normal 12-lead ECG: dominant R in V4-V6, rS pattern in
# V1-V2, inverted everything in aVR, T wave inverted in V1.
_W_I = {"p": 0.70, "qrs": 0.65, "t": 0.75}
_W_II = {"p": 1.00, "qrs": 1.00, "t": 1.00}
_W_PRECORDIAL = {
    "p": (0.25, 0.40, 0.45, 0.40, 0.35, 0.30),
    "qrs": (-0.55, -0.30, 0.45, 1.15, 1.20, 0.90),
    "t": (-0.30, 0.60, 0.90, 1.00, 0.85, 0.65),
}


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BeatFiducials:
    """Ground-truth fiducial sample indices for one complete cardiac cycle.

    Every field is an index into the recording's time axis. These are the exact
    positions the waveform was *constructed* from, rounded to the nearest
    sample - not the output of any detector. Detection code is scored against
    them.

    The clinical intervals follow directly:
    PR = ``qrs_onset - p_onset``, QRS duration = ``qrs_offset - qrs_onset``,
    QT = ``t_offset - qrs_onset``.
    """

    p_onset: int
    p_peak: int
    p_offset: int
    qrs_onset: int
    q_peak: int
    r_peak: int
    s_peak: int
    qrs_offset: int
    t_onset: int
    t_peak: int
    t_offset: int

    def as_dict(self) -> dict[str, int]:
        """Fiducial names mapped to sample indices."""
        return {
            "p_onset": self.p_onset, "p_peak": self.p_peak, "p_offset": self.p_offset,
            "qrs_onset": self.qrs_onset, "q_peak": self.q_peak, "r_peak": self.r_peak,
            "s_peak": self.s_peak, "qrs_offset": self.qrs_offset,
            "t_onset": self.t_onset, "t_peak": self.t_peak, "t_offset": self.t_offset,
        }


@dataclass(frozen=True)
class SyntheticRecording:
    """One synthetic 12-lead recording with its full ground truth.

    Attributes
    ----------
    signal:
        ``(12, n_samples)`` float32 array in millivolts, lead order
        :data:`LEAD_NAMES`.
    age_years:
        Chronological age - the label a regression model is trained to predict.
    sex:
        0 = male, 1 = female.
    known_age_offset_years, unexplained_age_offset_years:
        The two latent offsets that generated this recording's age gap.
        **Ground truth for tests only.** No pipeline component may read these;
        doing so is reading the answer key.
    r_peak_samples:
        Sample index of every R peak whose **entire QRS complex** lies inside
        the window. Beats whose P or T wave is clipped by the window edge are
        included - their complex is intact and detectable - but beats whose
        complex itself is cut in half are not, since no detector could find
        them. This is what an R-peak detector should find, exactly.
    beats:
        Full fiducial sets for the beats whose *entire* cycle fits inside the
        window. A strict subset of ``r_peak_samples``; interval measurements are
        scored against these.
    true_intervals:
        Ground-truth interval values in the same units the measurement code
        reports (ms, and bpm for heart rate).
    diagnostic_labels:
        Multi-hot vector over :data:`DIAGNOSTIC_CLASSES`.
    """

    record_id: str
    patient_id: int
    signal: np.ndarray
    sampling_rate_hz: int
    age_years: float
    sex: int
    known_age_offset_years: float
    unexplained_age_offset_years: float
    r_peak_samples: np.ndarray
    beats: tuple[BeatFiducials, ...]
    true_intervals: dict[str, float]
    diagnostic_labels: np.ndarray
    lead_names: tuple[str, ...] = LEAD_NAMES

    @property
    def n_samples(self) -> int:
        """Number of time samples in the recording."""
        return int(self.signal.shape[1])

    @property
    def duration_seconds(self) -> float:
        """Recording duration in seconds."""
        return self.n_samples / self.sampling_rate_hz

    def lead(self, name: str) -> np.ndarray:
        """Return a single lead's waveform by clinical name, e.g. ``"II"``."""
        try:
            index = self.lead_names.index(name)
        except ValueError:
            raise KeyError(
                f"unknown lead {name!r}; expected one of {self.lead_names}"
            ) from None
        return self.signal[index]


@dataclass
class SyntheticCohort:
    """A collection of synthetic recordings plus the config that produced them."""

    recordings: list[SyntheticRecording]
    config: SyntheticConfig

    def __len__(self) -> int:
        return len(self.recordings)

    def __iter__(self) -> Iterator[SyntheticRecording]:
        return iter(self.recordings)

    def __getitem__(self, index: int) -> SyntheticRecording:
        return self.recordings[index]

    @property
    def signals(self) -> np.ndarray:
        """All waveforms stacked into ``(n_recordings, 12, n_samples)`` float32."""
        return np.stack([r.signal for r in self.recordings]).astype(np.float32)

    @property
    def ages(self) -> np.ndarray:
        """Chronological ages - the regression targets."""
        return np.array([r.age_years for r in self.recordings], dtype=np.float64)

    @property
    def sexes(self) -> np.ndarray:
        """Sex codes (0 = male, 1 = female)."""
        return np.array([r.sex for r in self.recordings], dtype=np.int64)

    @property
    def patient_ids(self) -> np.ndarray:
        """Patient identifiers, used to verify patient-level split integrity."""
        return np.array([r.patient_id for r in self.recordings], dtype=np.int64)

    @property
    def diagnostic_labels(self) -> np.ndarray:
        """Multi-hot diagnostic labels, ``(n_recordings, len(DIAGNOSTIC_CLASSES))``."""
        return np.stack([r.diagnostic_labels for r in self.recordings]).astype(np.int64)

    def metadata(self) -> pd.DataFrame:
        """Per-recording metadata and ground truth as a dataframe.

        Includes the latent offsets and true intervals. These columns are ground
        truth for testing; production pipeline code must derive its own interval
        measurements from the waveform rather than reading them from here.
        """
        rows: list[dict[str, Any]] = []
        for rec in self.recordings:
            row: dict[str, Any] = {
                "record_id": rec.record_id,
                "patient_id": rec.patient_id,
                "age": rec.age_years,
                "sex": rec.sex,
                "n_beats_complete": len(rec.beats),
                "n_r_peaks": int(rec.r_peak_samples.size),
                "true_known_age_offset": rec.known_age_offset_years,
                "true_unexplained_age_offset": rec.unexplained_age_offset_years,
            }
            row.update({f"true_{k}": v for k, v in rec.true_intervals.items()})
            row.update(
                {f"dx_{name}": int(rec.diagnostic_labels[i])
                 for i, name in enumerate(DIAGNOSTIC_CLASSES)}
            )
            rows.append(row)
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Waveform primitives
# --------------------------------------------------------------------------- #
def _bump(u: np.ndarray, skew: float = 0.0) -> np.ndarray:
    """A raised-cosine bump on ``u in [0, 1]``, optionally skewed.

    Defined as ``0.5 * (1 - cos(2*pi * u**k))`` with ``k = exp(skew)``.

    Three properties make this the right primitive for generating testable ECG
    waves, and the third is the one the whole Phase 8 design depends on:

    1. It is exactly zero at ``u = 0`` and ``u = 1``, so the wave has compact
       support and its onset and offset are exact, not threshold-dependent.
    2. Its peak value is exactly 1 for any skew, so skewing a wave does not
       change its amplitude.
    3. **Skew moves the peak without moving the endpoints.** The peak sits at
       ``u = 0.5**(1/k)``, but the support stays ``[0, 1]`` for every ``k``.
       A skewed T wave therefore has an identical onset, offset, duration and
       peak amplitude to an unskewed one - which means *no timing interval can
       detect it*. That is precisely what makes T-wave skew a valid synthetic
       stand-in for a discovery candidate: a real morphological change that the
       classical measurement toolkit is blind to.

    Parameters
    ----------
    u:
        Normalised position within the wave, in ``[0, 1]``.
    skew:
        Zero is symmetric. Positive values push the peak later in the wave
        (a slow upstroke and fast downstroke); negative values push it earlier.
    """
    k = math.exp(skew)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.power(np.clip(u, 0.0, 1.0), k)))


def _peak_position(skew: float) -> float:
    """Normalised position of a skewed bump's peak within its support."""
    return float(0.5 ** (1.0 / math.exp(skew)))


def _rasterize(
    out: np.ndarray,
    fs: float,
    t_start: float,
    t_end: float,
    amplitude: float,
    skew: float = 0.0,
) -> None:
    """Add one bump spanning ``[t_start, t_end]`` seconds into ``out`` in place.

    The wave is treated as a continuous function of time and evaluated at
    whichever sample instants fall inside its support, so the same wave
    generated at 100 Hz and at 500 Hz describes the identical underlying signal.
    Samples outside the array are clipped, which is how beats at the window edge
    end up partially observed - exactly as in a real 10-second recording.
    """
    if t_end <= t_start:
        return
    n_total = out.shape[0]
    first = max(int(math.ceil(t_start * fs)), 0)
    last = min(int(math.floor(t_end * fs)), n_total - 1)
    if last < first:
        return
    idx = np.arange(first, last + 1)
    u = (idx / fs - t_start) / (t_end - t_start)
    out[idx] += amplitude * _bump(u, skew)


def _lead_weights(rng: np.random.Generator, jitter: float) -> dict[str, np.ndarray]:
    """Per-wave projection weights for all 12 leads.

    Leads I and II and the six precordial leads get independent random amplitude
    jitter; the remaining limb leads are then *derived* from I and II using the
    standard relations, so the clean signal satisfies them exactly::

        III = II - I                (Einthoven's law)
        aVR = -(I + II) / 2
        aVL = I - II / 2
        aVF = II - I / 2

    Reproducing those constraints matters because they are a property real
    12-lead ECGs have, and a test can check for them.
    """
    weights: dict[str, np.ndarray] = {}
    for wave in ("p", "qrs", "t"):
        def jittered(value: float) -> float:
            return float(value * (1.0 + jitter * rng.standard_normal()))

        w_i = jittered(_W_I[wave])
        w_ii = jittered(_W_II[wave])
        precordial = [jittered(v) for v in _W_PRECORDIAL[wave]]
        weights[wave] = np.array(
            [
                w_i,                    # I
                w_ii,                   # II
                w_ii - w_i,             # III
                -(w_i + w_ii) / 2.0,    # aVR
                w_i - w_ii / 2.0,       # aVL
                w_ii - w_i / 2.0,       # aVF
                *precordial,            # V1..V6
            ],
            dtype=np.float64,
        )
    return weights


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def _patient_attributes(
    config: SyntheticConfig, patient_id: int
) -> tuple[float, int, float, float]:
    """Draw a patient's age, sex and two latent ECG-age offsets.

    Keyed on ``patient_id`` alone, so a patient contributing two recordings gets
    identical demographics and offsets in both - which is what makes them a
    genuine patient-level split hazard, the thing Phase 5 must handle correctly.
    """
    rng = np.random.default_rng([config.seed, 0xA9E, patient_id])
    age = float(rng.uniform(config.age_min_years, config.age_max_years))
    sex = int(rng.random() < config.p_female)
    known_offset = float(rng.standard_normal() * config.known_age_offset_sd_years)
    unexplained_offset = float(
        rng.standard_normal() * config.unexplained_age_offset_sd_years
    )
    return age, sex, known_offset, unexplained_offset


def _diagnostic_labels(
    config: SyntheticConfig,
    rng: np.random.Generator,
    known_offset: float,
    unexplained_offset: float,
) -> np.ndarray:
    """Draw a multi-hot diagnostic label, optionally linked to a latent offset.

    ``diagnostic_link_source`` selects which latent offset raises the odds of an
    abnormal label. This exists to verify the Phase 9 machinery in both
    directions: that it detects a link when one was built in, and that it
    reports nothing when none was. It carries no clinical meaning whatsoever.
    """
    source = config.diagnostic_link_source
    if source == "unexplained":
        sd = config.unexplained_age_offset_sd_years
        z = unexplained_offset / sd if sd > 0 else 0.0
    elif source == "known":
        sd = config.known_age_offset_sd_years
        z = known_offset / sd if sd > 0 else 0.0
    else:
        z = 0.0

    base = config.diagnostic_abnormal_base_rate
    logit = math.log(base / (1.0 - base)) + config.diagnostic_link_strength * z
    labels = np.zeros(len(DIAGNOSTIC_CLASSES), dtype=np.int64)
    if rng.random() < _sigmoid(logit):
        # Abnormal: pick one of the four non-NORM superclasses uniformly. Real
        # PTB-XL labels are multi-label and correlated; this simplification is
        # fine because Phase 9 fits one-vs-rest classifiers per superclass.
        labels[1 + int(rng.integers(0, len(DIAGNOSTIC_CLASSES) - 1))] = 1
    else:
        labels[0] = 1
    return labels


def generate_recording(
    config: SyntheticConfig,
    record_index: int,
    patient_id: int,
) -> SyntheticRecording:
    """Generate one synthetic 12-lead recording with complete ground truth.

    Randomness is drawn from two independent streams: a patient stream keyed on
    ``patient_id`` (age, sex, latent offsets) and a recording stream keyed on
    ``record_index`` (heart rate, lead amplitudes, noise). Because morphology
    parameters are drawn before any sampling-rate-dependent quantity, generating
    the same ``record_index`` at 100 Hz and at 500 Hz yields two rasterisations
    of the *same underlying recording* - mirroring how PTB-XL ships both rates
    for each of its recordings.

    Parameters
    ----------
    config:
        Cohort parameters.
    record_index:
        Index of this recording within the cohort; seeds the recording stream.
    patient_id:
        Which synthetic patient this recording belongs to.
    """
    fs = float(config.sampling_rate_hz)
    n_samples = config.n_samples
    duration = config.duration_seconds

    age, sex, known_offset, unexplained_offset = _patient_attributes(config, patient_id)
    rng = np.random.default_rng([config.seed, 0x5EC, record_index])

    # -- Age-driven morphology ------------------------------------------------
    # Morphology follows the *offset* ages, so the model's disagreement with the
    # chronological label is caused by the offsets rather than being noise.
    qrs_decades = (age + known_offset - config.reference_age_years) / 10.0
    t_decades = (age + unexplained_offset - config.reference_age_years) / 10.0

    qrs_ms = config.qrs_duration_ms + config.qrs_widening_ms_per_decade * qrs_decades
    qrs_ms = max(qrs_ms, 20.0)
    t_skew = config.t_wave_skew_per_decade * t_decades
    # Between-person variation in PR, unrelated to age or to either latent
    # offset. It exists so PR carries real variance in the known-feature set and
    # its measurement can be validated against something other than a constant.
    pr_ms = config.pr_interval_ms + config.pr_interval_sd_ms * rng.standard_normal()
    pr_ms = float(np.clip(pr_ms, config.p_duration_ms + 20.0, 320.0))

    heart_rate = (
        config.heart_rate_bpm_mean
        + config.heart_rate_bpm_sd * rng.standard_normal()
        + config.hr_change_bpm_per_decade * qrs_decades
    )
    heart_rate = float(np.clip(heart_rate, 35.0, 150.0))
    mean_rr_ms = 60_000.0 / heart_rate

    lead_weights = _lead_weights(rng, config.lead_amplitude_jitter)

    # Beat-to-beat RR jitter (sinus arrhythmia). Drawn at a fixed size so the
    # number of random draws does not depend on the sampling rate.
    max_beats = int(duration * 150.0 / 60.0) + 8
    rr_jitter = rng.standard_normal(max_beats) * config.hr_sinus_arrhythmia_frac

    # -- Beat timing ----------------------------------------------------------
    qrs_pre_s = (_R_PEAK_FRAC * qrs_ms) / 1000.0     # QRS onset -> R peak
    qrs_post_s = ((1.0 - _R_PEAK_FRAC) * qrs_ms) / 1000.0
    earliest_r = (pr_ms / 1000.0) + qrs_pre_s
    t_r = earliest_r + float(rng.uniform(0.0, 0.15))

    p_component = np.zeros(n_samples, dtype=np.float64)
    qrs_component = np.zeros(n_samples, dtype=np.float64)
    t_component = np.zeros(n_samples, dtype=np.float64)

    r_peaks: list[int] = []
    beats: list[BeatFiducials] = []
    qt_values: list[float] = []
    beat_index = 0

    while t_r < duration and beat_index < max_beats:
        rr_s = (mean_rr_ms / 1000.0) * (1.0 + rr_jitter[beat_index])
        rr_s = max(rr_s, 0.3)

        # T-wave duration shortens at faster heart rates (Bazett-style), so that
        # QT varies with rate and QTc is a meaningful derived quantity.
        t_dur_s = (config.t_duration_ms / 1000.0) * math.sqrt(
            rr_s / (60.0 / _REFERENCE_HR_BPM)
        )

        qrs_onset = t_r - qrs_pre_s
        qrs_offset = t_r + qrs_post_s
        p_onset = qrs_onset - pr_ms / 1000.0
        p_offset = p_onset + config.p_duration_ms / 1000.0
        t_onset = qrs_offset + config.st_segment_ms / 1000.0
        t_offset = t_onset + t_dur_s

        # P wave
        _rasterize(p_component, fs, p_onset, p_offset, config.p_amplitude_mv)
        # QRS: three lobes tiling [qrs_onset, qrs_offset] exactly
        qrs_s = qrs_ms / 1000.0
        for (lo, hi), amp in (
            (_Q_SPAN, config.q_amplitude_mv),
            (_R_SPAN, config.r_amplitude_mv),
            (_S_SPAN, config.s_amplitude_mv),
        ):
            _rasterize(
                qrs_component, fs,
                qrs_onset + lo * qrs_s, qrs_onset + hi * qrs_s,
                amp,
            )
        # T wave, skewed but with unchanged support
        _rasterize(t_component, fs, t_onset, t_offset, config.t_amplitude_mv, t_skew)

        # A beat counts as present only if its whole QRS complex is inside the
        # window. A complex cut in half by the recording's edge carries too
        # little energy for any detector to find, and has no well-defined peak
        # to localise, so listing it as ground truth would score detectors
        # against something that is not in the data. Beats whose P or T wave is
        # clipped are still included: their QRS is intact and detectable.
        r_sample = int(round(t_r * fs))
        if qrs_onset >= 0.0 and qrs_offset <= duration and 0 <= r_sample < n_samples:
            r_peaks.append(r_sample)

        # Record full fiducials only for beats entirely inside the window.
        if p_onset >= 0.0 and t_offset <= duration:
            t_peak_s = t_onset + _peak_position(t_skew) * t_dur_s
            beats.append(
                BeatFiducials(
                    p_onset=int(round(p_onset * fs)),
                    p_peak=int(round((p_onset + p_offset) / 2.0 * fs)),
                    p_offset=int(round(p_offset * fs)),
                    qrs_onset=int(round(qrs_onset * fs)),
                    q_peak=int(round((qrs_onset + np.mean(_Q_SPAN) * qrs_s) * fs)),
                    r_peak=r_sample,
                    s_peak=int(round((qrs_onset + np.mean(_S_SPAN) * qrs_s) * fs)),
                    qrs_offset=int(round(qrs_offset * fs)),
                    t_onset=int(round(t_onset * fs)),
                    t_peak=int(round(t_peak_s * fs)),
                    t_offset=int(round(t_offset * fs)),
                )
            )
            qt_values.append((t_offset - qrs_onset) * 1000.0)

        t_r += rr_s
        beat_index += 1

    # -- Project onto 12 leads ------------------------------------------------
    signal = (
        np.outer(lead_weights["p"], p_component)
        + np.outer(lead_weights["qrs"], qrs_component)
        + np.outer(lead_weights["t"], t_component)
    )

    # -- Measurement realism --------------------------------------------------
    # Added per lead independently, which is why it breaks the exact Einthoven
    # relations that hold in the clean signal - as it does in real recordings.
    time_axis = np.arange(n_samples) / fs
    if config.baseline_wander_mv > 0:
        phases = rng.uniform(0, 2 * np.pi, size=config.n_leads)
        signal += config.baseline_wander_mv * np.sin(
            2 * np.pi * config.baseline_wander_hz * time_axis[None, :] + phases[:, None]
        )
    if config.powerline_mv > 0:
        phases = rng.uniform(0, 2 * np.pi, size=config.n_leads)
        signal += config.powerline_mv * np.sin(
            2 * np.pi * config.powerline_hz * time_axis[None, :] + phases[:, None]
        )
    if config.noise_mv_sd > 0:
        signal += rng.standard_normal(signal.shape) * config.noise_mv_sd

    # -- Ground-truth intervals ----------------------------------------------
    r_peak_array = np.array(r_peaks, dtype=np.int64)
    if r_peak_array.size >= 2:
        observed_rr_ms = float(np.mean(np.diff(r_peak_array)) / fs * 1000.0)
    else:
        observed_rr_ms = mean_rr_ms
    observed_hr = 60_000.0 / observed_rr_ms
    qt_ms = float(np.mean(qt_values)) if qt_values else float("nan")

    true_intervals = {
        "heart_rate_bpm": observed_hr,
        "rr_interval_ms": observed_rr_ms,
        "qrs_duration_ms": float(qrs_ms),
        "pr_interval_ms": float(pr_ms),
        "qt_interval_ms": qt_ms,
        "qtc_bazett_ms": qt_ms / math.sqrt(observed_rr_ms / 1000.0),
        "t_wave_skew": float(t_skew),
    }

    return SyntheticRecording(
        record_id=f"syn_{record_index:06d}",
        patient_id=patient_id,
        signal=signal.astype(np.float32),
        sampling_rate_hz=config.sampling_rate_hz,
        age_years=age,
        sex=sex,
        known_age_offset_years=known_offset,
        unexplained_age_offset_years=unexplained_offset,
        r_peak_samples=r_peak_array,
        beats=tuple(beats),
        true_intervals=true_intervals,
        diagnostic_labels=_diagnostic_labels(config, rng, known_offset, unexplained_offset),
    )


def _assign_patients(config: SyntheticConfig, n_recordings: int) -> list[int]:
    """Assign a patient id to each recording, with some patients repeating.

    PTB-XL contains patients with more than one recording, which is what makes
    recording-level splitting a silent correctness bug. The synthetic cohort
    reproduces that hazard so the Phase 5 split logic can be tested against it.
    """
    rng = np.random.default_rng([config.seed, 0xB0B])
    assignments: list[int] = []
    next_pid = 0
    while len(assignments) < n_recordings:
        pid = next_pid
        next_pid += 1
        assignments.append(pid)
        if len(assignments) < n_recordings and rng.random() < config.repeat_patient_frac:
            assignments.append(pid)
    return assignments


def generate_cohort(
    config: SyntheticConfig, n_recordings: int | None = None
) -> SyntheticCohort:
    """Generate a full synthetic cohort.

    Parameters
    ----------
    config:
        Cohort parameters.
    n_recordings:
        Override for ``config.n_recordings``, useful in tests that need a small
        cohort. Recording ``i`` is identical whatever the cohort size, so a
        small cohort is a genuine prefix of a large one.
    """
    total = int(n_recordings if n_recordings is not None else config.n_recordings)
    if total <= 0:
        raise ValueError(f"n_recordings must be positive, got {total}")
    patients = _assign_patients(config, total)
    recordings = [generate_recording(config, i, patients[i]) for i in range(total)]
    return SyntheticCohort(recordings=recordings, config=config)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_cohort(cohort: SyntheticCohort, directory: str | Path) -> Path:
    """Write a cohort to disk so pipeline stages can run as separate processes.

    Produces ``signals.npy`` (float32), ``metadata.csv``, ``ground_truth.json``
    (fiducials and R peaks) and ``config.json`` in ``directory``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    np.save(directory / "signals.npy", cohort.signals)
    cohort.metadata().to_csv(directory / "metadata.csv", index=False)

    ground_truth = {
        rec.record_id: {
            "r_peak_samples": rec.r_peak_samples.tolist(),
            "beats": [beat.as_dict() for beat in rec.beats],
            "true_intervals": rec.true_intervals,
            "diagnostic_labels": rec.diagnostic_labels.tolist(),
        }
        for rec in cohort.recordings
    }
    (directory / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=1), encoding="utf-8"
    )
    (directory / "config.json").write_text(
        json.dumps(config_to_dict(cohort.config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return directory


def load_cohort(directory: str | Path) -> SyntheticCohort:
    """Read back a cohort written by :func:`save_cohort`."""
    directory = Path(directory)
    signals = np.load(directory / "signals.npy")
    metadata = pd.read_csv(directory / "metadata.csv")
    ground_truth = json.loads((directory / "ground_truth.json").read_text())
    config = SyntheticConfig(**json.loads((directory / "config.json").read_text()))

    recordings: list[SyntheticRecording] = []
    for i, row in metadata.iterrows():
        truth = ground_truth[row["record_id"]]
        recordings.append(
            SyntheticRecording(
                record_id=str(row["record_id"]),
                patient_id=int(row["patient_id"]),
                signal=signals[i],
                sampling_rate_hz=config.sampling_rate_hz,
                age_years=float(row["age"]),
                sex=int(row["sex"]),
                known_age_offset_years=float(row["true_known_age_offset"]),
                unexplained_age_offset_years=float(row["true_unexplained_age_offset"]),
                r_peak_samples=np.array(truth["r_peak_samples"], dtype=np.int64),
                beats=tuple(BeatFiducials(**b) for b in truth["beats"]),
                true_intervals=truth["true_intervals"],
                diagnostic_labels=np.array(truth["diagnostic_labels"], dtype=np.int64),
            )
        )
    return SyntheticCohort(recordings=recordings, config=config)
