"""Classical ECG interval measurements - the "already known" feature set.

WHY THIS MODULE IS THE PIVOT OF THE WHOLE PROJECT
-------------------------------------------------
Phase 8 asks whether a neural network's ECG age gap encodes anything cardiology
does not already measure. Answering that requires a concrete, defensible
definition of "already measured", and this module is it: heart rate, PR
interval, QRS duration and QT interval, computed by classical signal
processing with no learning of any kind, exactly as they have been measured
from ECGs for decades.

The logic of the validation framework is that any component of the model's age
gap predictable from these numbers is *rediscovery*, and only the remainder is
even a candidate for discovery. That places an unusual burden on this file: it
is not merely a feature extractor, it is the yardstick the paper's central
claim is measured against.

THE ERROR THAT MATTERS, AND WHICH WAY IT POINTS
-----------------------------------------------
Measurement error here is not symmetric in its consequences. If these features
are noisier or more biased than they should be, they explain *less* of the age
gap than the underlying physiology would - which makes the unexplained residual
look larger and a discovery claim look stronger. **Sloppy measurement here
manufactures false discoveries.** Two decisions follow from that asymmetry:

1. Intervals are measured at 500 Hz even though the model is trained at 100 Hz
   (see ``DataConfig.interval_sampling_rate_hz``). One sample at 100 Hz is
   10 ms, comparable to the effects being resolved.
2. Per-recording values are aggregated across beats with the **median**, so a
   single mis-delineated beat cannot corrupt a recording's features. A mean
   would let one bad beat through.

WHAT IS MEASURED
----------------
``heart_rate_bpm``
    Beats per minute, from the spacing between consecutive R peaks.
``rr_interval_ms``, ``rr_sd_ms``
    Mean beat-to-beat interval and its variability. The variability term is
    reported because it is a genuinely distinct property of the rhythm.
``p_duration_ms``
    Width of the P wave: how long atrial depolarisation takes.
``pr_interval_ms``
    P onset to QRS onset: how long the impulse takes to travel from the atria
    to the ventricles. Prolonged in conduction disease.
``qrs_duration_ms``
    Width of the QRS complex: how long ventricular depolarisation takes.
    Prolonged in bundle branch block, and one of the quantities that changes
    with age.
``qt_interval_ms``, ``qtc_bazett_ms``, ``qtc_fridericia_ms``
    QRS onset to T offset - the total duration of ventricular depolarisation
    plus repolarisation - and two rate corrections of it. QT shortens as heart
    rate rises, so a raw QT confounds repolarisation with rate; the corrections
    divide it by the RR interval raised to a power (1/2 for Bazett, 1/3 for
    Fridericia). Both are reported because neither is uniformly preferred:
    Bazett is the clinical convention but over-corrects at high heart rates,
    which Fridericia handles better.

BEYOND TIMING: AMPLITUDE, AXIS AND MORPHOLOGY
---------------------------------------------
Intervals are only part of what a cardiologist reads off an ECG, and a
known-feature set containing only timing would understate what is already
known - which, by the asymmetry above, would inflate any discovery claim. The
following are therefore measured as well, all long-established clinical
quantities:

``r_amplitude_mv``, ``t_amplitude_mv``, ``p_amplitude_mv``
    Height of each wave above the isoelectric baseline. Amplitude carries
    information timing does not: voltage reflects muscle mass and the
    electrical distance from the electrode.
``st_deviation_mv``
    Displacement of the ST segment from baseline, measured 60 ms after the J
    point (the end of the QRS). The classical marker of ischaemia and injury.
``qrs_axis_deg``, ``t_axis_deg``
    The direction of the mean depolarisation and repolarisation vectors in the
    frontal plane, computed from the net deflections in leads I and aVF.
    Axis deviation is a routine clinical finding and shifts with age, chamber
    enlargement and conduction disease.
``sokolow_lyon_mv``
    S wave in V1 plus the taller R wave of V5 or V6 - the standard voltage
    criterion for left ventricular hypertrophy.
``r_progression_lead``
    The first precordial lead (V1-V6) in which the R wave exceeds the S wave.
    Normal transition is V3-V4; poor R-wave progression is a recognised
    abnormality.

These are computed across all twelve leads, unlike the intervals, which are
delineated on one. Adding them can only ever *increase* the share of the age
gap attributable to existing knowledge, which is the conservative direction for
this project's central claim.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ecg_discovery.config import SignalProcessingConfig
from ecg_discovery.signal_processing.atrial_features import (
    ATRIAL_FEATURE_NAMES, atrial_features,
)
from ecg_discovery.signal_processing.qrs_detection import detect_r_peaks
from ecg_discovery.signal_processing.wave_delineation import (
    BeatDelineation,
    delineate_beats,
)

__all__ = [
    "IntervalFeatures",
    "INTERVAL_FEATURE_NAMES",
    "compute_interval_features",
    "interval_features_table",
]

#: The measurable quantities, in reporting order. ``ValidationFrameworkConfig``
#: selects which of these count as "known" for the residual decomposition.
INTERVAL_FEATURE_NAMES: tuple[str, ...] = (
    # Timing
    "heart_rate_bpm",
    "rr_interval_ms",
    "rr_sd_ms",
    "p_duration_ms",
    "pr_interval_ms",
    "qrs_duration_ms",
    "qt_interval_ms",
    "qtc_bazett_ms",
    "qtc_fridericia_ms",
    # Amplitude, axis and morphology
    "r_amplitude_mv",
    "t_amplitude_mv",
    "p_amplitude_mv",
    "st_deviation_mv",
    "qrs_axis_deg",
    "t_axis_deg",
    "sokolow_lyon_mv",
    "r_progression_lead",
    # Extended atrial measurement. Separated because these exist to test our own
    # positive result against a fuller enumeration of what cardiology already
    # reads from the P wave; see ``atrial_features``.
    *ATRIAL_FEATURE_NAMES,
)


@dataclass(frozen=True)
class IntervalFeatures:
    """Classical interval measurements for one recording.

    Any field may be NaN when it could not be measured - too few beats, or no
    P wave found in any beat. NaN is used deliberately rather than a filled-in
    default: a fabricated interval would enter the known-feature set as if it
    were a measurement, and quietly distort the very comparison this project
    exists to make.

    Attributes
    ----------
    n_beats:
        Beats used for the measurement.
    p_detection_rate:
        Fraction of beats in which a P wave was found. A low value is
        meaningful rather than merely a quality warning - it is what atrial
        fibrillation looks like.
    """

    heart_rate_bpm: float
    rr_interval_ms: float
    rr_sd_ms: float
    p_duration_ms: float
    pr_interval_ms: float
    qrs_duration_ms: float
    qt_interval_ms: float
    qtc_bazett_ms: float
    qtc_fridericia_ms: float

    r_amplitude_mv: float
    t_amplitude_mv: float
    p_amplitude_mv: float
    st_deviation_mv: float
    qrs_axis_deg: float
    t_axis_deg: float
    sokolow_lyon_mv: float
    r_progression_lead: float

    p_terminal_force_v1_mv_ms: float
    p_area_ii_mv_ms: float
    p_notch_depth_mv: float
    p_dispersion_ms: float

    n_beats: int
    p_detection_rate: float
    t_detection_rate: float

    @property
    def is_measurable(self) -> bool:
        """Whether the core intervals were all successfully measured."""
        return bool(
            np.isfinite(
                [self.heart_rate_bpm, self.qrs_duration_ms, self.qt_interval_ms]
            ).all()
        )

    def as_dict(self) -> dict[str, Any]:
        """All fields as a plain dictionary."""
        return asdict(self)

    def feature_vector(self, names: Sequence[str]) -> np.ndarray:
        """Selected features as a float array, for the validation framework."""
        missing = [n for n in names if n not in {f.name for f in fields(self)}]
        if missing:
            raise KeyError(
                f"unknown interval feature(s): {missing}. Available: "
                f"{INTERVAL_FEATURE_NAMES}"
            )
        return np.array([getattr(self, name) for name in names], dtype=np.float64)

    @classmethod
    def empty(cls) -> "IntervalFeatures":
        """An all-NaN result, for recordings where nothing could be measured."""
        nan = float("nan")
        return cls(
            heart_rate_bpm=nan, rr_interval_ms=nan, rr_sd_ms=nan,
            p_duration_ms=nan, pr_interval_ms=nan, qrs_duration_ms=nan,
            qt_interval_ms=nan, qtc_bazett_ms=nan, qtc_fridericia_ms=nan,
            r_amplitude_mv=nan, t_amplitude_mv=nan, p_amplitude_mv=nan,
            st_deviation_mv=nan, qrs_axis_deg=nan, t_axis_deg=nan,
            sokolow_lyon_mv=nan, r_progression_lead=nan,
            p_terminal_force_v1_mv_ms=nan, p_area_ii_mv_ms=nan,
            p_notch_depth_mv=nan, p_dispersion_ms=nan,
            n_beats=0, p_detection_rate=0.0, t_detection_rate=0.0,
        )


def _aggregate(values: Sequence[float], how: str) -> float:
    """Combine per-beat measurements into one per-recording value.

    Median by default: a single mis-delineated beat then cannot move the
    recording's value, whereas a mean would let it through. That robustness is
    worth more here than the mean's efficiency, because the failure mode being
    guarded against - one beat delineated badly - is common and its effect on a
    mean is unbounded.
    """
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.median(finite) if how == "median" else np.mean(finite))


def _amplitude_features(
    signal_array: np.ndarray,
    beats: Sequence[BeatDelineation],
    lead_names: Sequence[str],
    sampling_rate_hz: float,
    config: SignalProcessingConfig,
) -> dict[str, float]:
    """Measure wave amplitudes, ST deviation, electrical axis and R progression.

    Unlike the intervals, these read all twelve leads: amplitude is a property
    of the direction the electrode looks from, so the same beat is tall in one
    lead and inverted in another. Each lead gets its own isoelectric reference,
    taken from its own PR segment, because baseline offset differs per lead.

    Returns NaN for any quantity whose required leads are absent, rather than
    substituting a value - a fabricated measurement would enter the
    known-feature set as though it had been observed.
    """
    nan = float("nan")
    empty = {
        "r_amplitude_mv": nan, "t_amplitude_mv": nan, "p_amplitude_mv": nan,
        "st_deviation_mv": nan, "qrs_axis_deg": nan, "t_axis_deg": nan,
        "sokolow_lyon_mv": nan, "r_progression_lead": nan,
    }
    array = np.asarray(signal_array, dtype=np.float64)
    if array.ndim != 2 or not beats:
        return empty

    lead_index = {name: i for i, name in enumerate(lead_names) if i < array.shape[0]}
    st_offset = int(round(0.060 * sampling_rate_hz))     # J point + 60 ms, standard
    baseline_window = max(
        int(round(config.baseline_window_ms / 1000.0 * sampling_rate_hz)), 1
    )

    def lead_baseline(lead: int, beat: BeatDelineation) -> float:
        """Isoelectric level for one lead, from the PR segment before the complex."""
        high = max(beat.qrs.onset - 1, 0)
        low = max(high - baseline_window, 0)
        return float(np.median(array[lead, low : high + 1])) if high > low else 0.0

    # Per-beat, per-lead measurements, aggregated with the median across beats.
    r_amp: list[float] = []
    t_amp: list[float] = []
    p_amp: list[float] = []
    st_dev: list[float] = []
    axis_qrs: list[float] = []
    axis_t: list[float] = []
    sokolow: list[float] = []
    progression: list[float] = []

    reference = lead_index.get("II", 0)
    for beat in beats:
        base_ref = lead_baseline(reference, beat)
        r_amp.append(abs(array[reference, beat.r_peak] - base_ref))
        if beat.t_wave is not None:
            t_amp.append(array[reference, beat.t_wave.peak] - base_ref)
        if beat.p_wave is not None:
            p_amp.append(array[reference, beat.p_wave.peak] - base_ref)

        # ST deviation: largest displacement across leads 60 ms after the J point.
        j_point = beat.qrs.offset + st_offset
        if j_point < array.shape[1]:
            deviations = [
                array[lead, j_point] - lead_baseline(lead, beat)
                for lead in lead_index.values()
            ]
            st_dev.append(max(deviations, key=abs))

        # Frontal-plane axes from the net deflection in leads I and aVF.
        if "I" in lead_index and "aVF" in lead_index:
            def net(lead: int, start: int, stop: int) -> float:
                base = lead_baseline(lead, beat)
                segment = array[lead, max(start, 0) : min(stop + 1, array.shape[1])]
                return float(np.sum(segment - base)) if segment.size else 0.0

            qrs_i = net(lead_index["I"], beat.qrs.onset, beat.qrs.offset)
            qrs_f = net(lead_index["aVF"], beat.qrs.onset, beat.qrs.offset)
            if abs(qrs_i) > 1e-9 or abs(qrs_f) > 1e-9:
                axis_qrs.append(math.degrees(math.atan2(qrs_f, qrs_i)))
            if beat.t_wave is not None:
                t_i = net(lead_index["I"], beat.t_wave.onset, beat.t_wave.offset)
                t_f = net(lead_index["aVF"], beat.t_wave.onset, beat.t_wave.offset)
                if abs(t_i) > 1e-9 or abs(t_f) > 1e-9:
                    axis_t.append(math.degrees(math.atan2(t_f, t_i)))

        # Sokolow-Lyon voltage: depth of S in V1 plus the taller R of V5/V6.
        if all(name in lead_index for name in ("V1", "V5", "V6")):
            window = slice(beat.qrs.onset, beat.qrs.offset + 1)
            s_v1 = abs(min(0.0, float(array[lead_index["V1"], window].min())
                           - lead_baseline(lead_index["V1"], beat)))
            r_v5 = max(0.0, float(array[lead_index["V5"], window].max())
                       - lead_baseline(lead_index["V5"], beat))
            r_v6 = max(0.0, float(array[lead_index["V6"], window].max())
                       - lead_baseline(lead_index["V6"], beat))
            sokolow.append(s_v1 + max(r_v5, r_v6))

        # R-wave progression: first precordial lead where R exceeds S.
        precordial = [f"V{i}" for i in range(1, 7)]
        if all(name in lead_index for name in precordial):
            window = slice(beat.qrs.onset, beat.qrs.offset + 1)
            transition = float("nan")
            for position, name in enumerate(precordial, start=1):
                lead = lead_index[name]
                base = lead_baseline(lead, beat)
                r_height = max(0.0, float(array[lead, window].max()) - base)
                s_depth = abs(min(0.0, float(array[lead, window].min()) - base))
                if r_height > s_depth:
                    transition = float(position)
                    break
            progression.append(transition)

    def median_of(values: Sequence[float]) -> float:
        finite = [v for v in values if v is not None and np.isfinite(v)]
        return float(np.median(finite)) if finite else nan

    return {
        "r_amplitude_mv": median_of(r_amp),
        "t_amplitude_mv": median_of(t_amp),
        "p_amplitude_mv": median_of(p_amp),
        "st_deviation_mv": median_of(st_dev),
        # Axes are directions, so the median is taken on the unwrapped angle to
        # avoid the wrap at +/-180 degrees producing a meaningless average.
        "qrs_axis_deg": (
            float(np.degrees(np.median(np.unwrap(np.radians(axis_qrs)))))
            if axis_qrs else nan
        ),
        "t_axis_deg": (
            float(np.degrees(np.median(np.unwrap(np.radians(axis_t)))))
            if axis_t else nan
        ),
        "sokolow_lyon_mv": median_of(sokolow),
        "r_progression_lead": median_of(progression),
    }


def compute_interval_features(
    signal_array: np.ndarray,
    sampling_rate_hz: float,
    config: SignalProcessingConfig | None = None,
    lead_names: Sequence[str] | None = None,
    beats: Sequence[BeatDelineation] | None = None,
    include_dispersion: bool = True,
) -> IntervalFeatures:
    """Measure the classical ECG intervals for one recording.

    Runs R-peak detection and wave delineation unless pre-computed ``beats`` are
    supplied, then aggregates per-beat measurements into per-recording values.

    Parameters
    ----------
    signal_array:
        ``(n_samples,)`` or ``(n_leads, n_samples)`` recording in millivolts.
    sampling_rate_hz:
        Sampling rate. Use 500 Hz where available: interval precision directly
        controls how much of the age gap the known features can explain.
    config:
        Signal-processing parameters; defaults are used if omitted.
    lead_names:
        Lead names of ``signal_array``; defaults to the standard 12-lead order.
    beats:
        Pre-computed delineation, to avoid repeating that work when the caller
        already has it.
    include_dispersion:
        Whether to measure P-wave dispersion, which costs twelve extra boundary
        searches per beat. Set False when the feature will not be used.

    Returns
    -------
    IntervalFeatures
        Measurements, with NaN for any quantity that could not be determined.
    """
    config = config or SignalProcessingConfig()

    if beats is None:
        detection = detect_r_peaks(signal_array, sampling_rate_hz, config, lead_names)
        beats = delineate_beats(
            signal_array, sampling_rate_hz, detection.r_peaks, config, lead_names
        )
    beats = list(beats)

    if len(beats) < config.min_beats_for_features:
        return IntervalFeatures.empty()

    samples_to_ms = 1000.0 / sampling_rate_hz

    # -- Rhythm ---------------------------------------------------------------
    r_peaks = np.array([beat.r_peak for beat in beats], dtype=np.float64)
    rr_ms = np.diff(r_peaks) * samples_to_ms
    if rr_ms.size == 0:
        return IntervalFeatures.empty()
    rr_interval = float(np.median(rr_ms) if config.aggregation == "median" else rr_ms.mean())
    heart_rate = 60_000.0 / rr_interval if rr_interval > 0 else float("nan")
    # Variability is a spread, so it is always a standard deviation regardless
    # of how central tendency is aggregated.
    rr_sd = float(np.std(rr_ms, ddof=1)) if rr_ms.size > 1 else 0.0

    # -- Per-beat intervals ---------------------------------------------------
    qrs_durations: list[float] = []
    pr_intervals: list[float] = []
    p_durations: list[float] = []
    qt_intervals: list[float] = []
    n_with_p = n_with_t = 0

    for beat in beats:
        qrs_durations.append(beat.qrs_duration_samples * samples_to_ms)
        if beat.p_wave is not None:
            n_with_p += 1
            p_durations.append(beat.p_wave.duration_samples * samples_to_ms)
            pr = beat.pr_interval_samples
            if pr is not None and pr > 0:
                pr_intervals.append(pr * samples_to_ms)
        if beat.t_wave is not None:
            n_with_t += 1
            qt = beat.qt_interval_samples
            if qt is not None and qt > 0:
                qt_intervals.append(qt * samples_to_ms)

    how = config.aggregation
    qrs_duration = _aggregate(qrs_durations, how)
    pr_interval = _aggregate(pr_intervals, how)
    p_duration = _aggregate(p_durations, how)
    qt_interval = _aggregate(qt_intervals, how)

    # -- Rate corrections -----------------------------------------------------
    # QT shortens as the heart speeds up, so a raw QT mixes repolarisation with
    # rate. Both standard corrections are reported: Bazett is the clinical
    # convention, Fridericia is better behaved at high rates.
    rr_seconds = rr_interval / 1000.0
    if np.isfinite(qt_interval) and rr_seconds > 0:
        qtc_bazett = qt_interval / math.sqrt(rr_seconds)
        qtc_fridericia = qt_interval / (rr_seconds ** (1.0 / 3.0))
    else:
        qtc_bazett = qtc_fridericia = float("nan")

    resolved_leads = lead_names or _default_lead_names(signal_array)
    amplitudes = _amplitude_features(
        signal_array, beats, resolved_leads, sampling_rate_hz, config,
    )
    atrial = atrial_features(
        signal_array, beats, resolved_leads, sampling_rate_hz, config,
        include_dispersion=include_dispersion,
    )

    return IntervalFeatures(
        heart_rate_bpm=heart_rate,
        rr_interval_ms=rr_interval,
        rr_sd_ms=rr_sd,
        p_duration_ms=p_duration,
        pr_interval_ms=pr_interval,
        qrs_duration_ms=qrs_duration,
        qt_interval_ms=qt_interval,
        qtc_bazett_ms=qtc_bazett,
        qtc_fridericia_ms=qtc_fridericia,
        **amplitudes,
        **atrial,
        n_beats=len(beats),
        p_detection_rate=n_with_p / len(beats),
        t_detection_rate=n_with_t / len(beats),
    )


def _default_lead_names(signal_array: np.ndarray) -> tuple[str, ...]:
    """Standard 12-lead order, truncated to whatever was supplied."""
    from ecg_discovery.data.synthetic_ecg import LEAD_NAMES

    array = np.asarray(signal_array)
    n_leads = array.shape[0] if array.ndim == 2 else 1
    return LEAD_NAMES[:n_leads]


def interval_features_table(
    signals: np.ndarray,
    sampling_rate_hz: float,
    config: SignalProcessingConfig | None = None,
    lead_names: Sequence[str] | None = None,
    record_ids: Sequence[str] | None = None,
    include_dispersion: bool = True,
) -> pd.DataFrame:
    """Measure intervals for a batch of recordings.

    Parameters
    ----------
    signals:
        ``(n_recordings, n_leads, n_samples)`` array.
    record_ids:
        Optional identifiers; a ``record_id`` column is added when supplied, so
        features can be joined to model predictions in Phase 8.

    Returns
    -------
    pandas.DataFrame
        One row per recording. Rows where measurement failed contain NaN rather
        than being dropped, so the table stays aligned with the input order.
    """
    signals = np.asarray(signals)
    if signals.ndim != 3:
        raise ValueError(
            f"expected (n_recordings, n_leads, n_samples), got shape {signals.shape}"
        )
    if record_ids is not None and len(record_ids) != signals.shape[0]:
        raise ValueError(
            f"record_ids has {len(record_ids)} entries but there are "
            f"{signals.shape[0]} recordings"
        )

    rows = [
        compute_interval_features(
            signals[i], sampling_rate_hz, config, lead_names,
            include_dispersion=include_dispersion,
        ).as_dict()
        for i in range(signals.shape[0])
    ]
    frame = pd.DataFrame(rows)
    if record_ids is not None:
        frame.insert(0, "record_id", list(record_ids))
    return frame
