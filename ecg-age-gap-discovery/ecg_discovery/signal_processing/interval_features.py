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
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ecg_discovery.config import SignalProcessingConfig
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
    "heart_rate_bpm",
    "rr_interval_ms",
    "rr_sd_ms",
    "p_duration_ms",
    "pr_interval_ms",
    "qrs_duration_ms",
    "qt_interval_ms",
    "qtc_bazett_ms",
    "qtc_fridericia_ms",
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


def compute_interval_features(
    signal_array: np.ndarray,
    sampling_rate_hz: float,
    config: SignalProcessingConfig | None = None,
    lead_names: Sequence[str] | None = None,
    beats: Sequence[BeatDelineation] | None = None,
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
        n_beats=len(beats),
        p_detection_rate=n_with_p / len(beats),
        t_detection_rate=n_with_t / len(beats),
    )


def interval_features_table(
    signals: np.ndarray,
    sampling_rate_hz: float,
    config: SignalProcessingConfig | None = None,
    lead_names: Sequence[str] | None = None,
    record_ids: Sequence[str] | None = None,
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
            signals[i], sampling_rate_hz, config, lead_names
        ).as_dict()
        for i in range(signals.shape[0])
    ]
    frame = pd.DataFrame(rows)
    if record_ids is not None:
        frame.insert(0, "record_id", list(record_ids))
    return frame
