"""Approximate P-wave and T-wave localisation relative to detected R peaks.

WHAT THIS FINDS
---------------
Given the R peaks from :mod:`ecg_discovery.signal_processing.qrs_detection`,
this module marks the boundaries of the three waves that make up a heartbeat:

- **P wave** - atrial depolarisation, the small bump before the spike.
- **QRS complex** - ventricular depolarisation, the spike itself.
- **T wave** - ventricular repolarisation, the broader bump after the spike.

Those boundaries define the intervals cardiologists have measured for a
century (PR, QRS duration, QT), which
:mod:`ecg_discovery.signal_processing.interval_features` turns into numbers and
Phase 8 treats as the body of "what is already known".

HOW IT WORKS
------------
*QRS boundaries.* Walking outward from the R peak along the raw signal does not
work: the waveform crosses the baseline between the Q, R and S deflections, so
any threshold on amplitude or slope would stop at the first internal crossing
and report a complex a third of its true width. Instead the search runs on a
smoothed envelope of the absolute derivative, which fills those internal gaps,
and a boundary is only accepted where the envelope stays quiet for a sustained
run rather than for a single sample.

*P wave.* The largest deflection from baseline inside a physiologically
plausible window before QRS onset (40-320 ms). Its edges are where the
deflection falls back to a fraction of its own peak.

*T wave.* The largest deflection in a window after QRS offset whose length
scales with the RR interval, because repolarisation genuinely takes longer at
slower heart rates. Its end is found with the **tangent method** (Lepeschkin):
a tangent is drawn at the steepest point of the T wave's downslope and followed
to where it crosses baseline. This is the conventional manual technique and is
more stable than an amplitude threshold, because the T wave approaches baseline
gradually and a threshold crossing is very sensitive to where the baseline is
assumed to sit.

THIS IS AN APPROXIMATION - WHAT THAT COSTS
------------------------------------------
Clinical-grade delineation is a harder problem than this project needs to
solve, and it is important not to present these heuristics as equivalent to it.
Concretely:

1. **Single-lead measurement understates global durations.** A clinical QRS
   duration is measured from the *earliest* onset in any lead to the *latest*
   offset in any lead, because depolarisation reaches different parts of the
   heart at different times. Measuring in one lead misses whatever begins or
   ends elsewhere, and so is biased short. This module can read a multi-lead
   vector-magnitude signal instead of a single lead (``qrs_boundary_source``),
   which recovers part of that, but it is not equivalent to true multi-lead
   delineation.
2. **The synthetic cohort cannot detect that particular error.** Its leads are
   exact linear projections of three shared components, so every lead has
   identical wave timing and inter-lead dispersion is zero by construction. Any
   validation here is therefore silent on the single-lead bias, which is a
   limitation of the test fixture and not evidence that the bias is absent. It
   has to be re-checked on PTB-XL.
3. **T-offset is the least reliable measurement**, in this implementation and in
   clinical practice alike. Where a T wave "ends" is genuinely ambiguous when it
   merges into the following P wave or into baseline drift.
4. **No pathology handling.** Absent P waves (atrial fibrillation), fused P and
   T waves at high rates, and bundle branch blocks are not specially handled.
   A P wave that cannot be found is reported as missing rather than guessed at.

Because Phase 8 measures discovery *against* these features, any measurement
error here inflates the apparent "unexplained" residual - the direction that
would flatter a discovery claim. That asymmetry is the reason these limitations
are stated this plainly, and it is discussed in docs/validation_methodology.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ecg_discovery.config import SignalProcessingConfig
from ecg_discovery.signal_processing.qrs_detection import bandpass_filter

__all__ = [
    "WaveBoundaries",
    "BeatDelineation",
    "delineate_beats",
    "delineation_signals",
]


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WaveBoundaries:
    """Onset, peak and offset sample indices of a single wave."""

    onset: int
    peak: int
    offset: int

    @property
    def duration_samples(self) -> int:
        """Width of the wave in samples."""
        return self.offset - self.onset

    def contains(self, index: int) -> bool:
        """Whether a sample index falls inside this wave."""
        return self.onset <= index <= self.offset


@dataclass(frozen=True)
class BeatDelineation:
    """Wave boundaries for one heartbeat.

    ``p_wave`` and ``t_wave`` are ``None`` when the wave could not be found -
    which is a real occurrence, not just a failure: atrial fibrillation has no
    organised P wave at all. Downstream code must handle absence rather than
    receive an invented value.

    Attributes
    ----------
    r_peak:
        The R-peak sample this beat was delineated around.
    baseline_mv:
        Isoelectric level estimated from the PR segment, in millivolts. Wave
        amplitudes are measured relative to it.
    """

    r_peak: int
    qrs: WaveBoundaries
    p_wave: WaveBoundaries | None
    t_wave: WaveBoundaries | None
    baseline_mv: float

    @property
    def qrs_duration_samples(self) -> int:
        """Width of the QRS complex in samples."""
        return self.qrs.duration_samples

    @property
    def pr_interval_samples(self) -> int | None:
        """P onset to QRS onset, or ``None`` if no P wave was found."""
        return None if self.p_wave is None else self.qrs.onset - self.p_wave.onset

    @property
    def qt_interval_samples(self) -> int | None:
        """QRS onset to T offset, or ``None`` if no T wave was found."""
        return None if self.t_wave is None else self.t_wave.offset - self.qrs.onset


# --------------------------------------------------------------------------- #
# Signal preparation
# --------------------------------------------------------------------------- #
def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average, used to smooth the derivative envelope."""
    window = max(int(window), 1)
    if window == 1:
        return x.copy()
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="same")[pad : pad + x.size]


def delineation_signals(
    signal_array: np.ndarray,
    sampling_rate_hz: float,
    config: SignalProcessingConfig,
    lead_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare the two signals delineation reads.

    Returns
    -------
    wave_signal:
        A single, morphology-preserving lead (0.5-40 Hz bandpass). Wave peaks
        and their polarity are read from this, because a P or T wave's direction
        carries meaning that a rectified signal would destroy.
    qrs_envelope:
        A smoothed envelope of absolute slope, used only to find QRS boundaries.
        When ``config.qrs_boundary_source`` is ``vector_magnitude`` and a
        multi-lead recording is supplied, this is built from the root-mean-square
        across all leads, which is both steadier against per-lead noise and
        closer to the clinical notion of a *global* QRS duration than any single
        lead can be.
    """
    array = np.asarray(signal_array, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(
            f"expected 1-D or (n_leads, n_samples) input, got shape {array.shape}"
        )

    if lead_names is None:
        from ecg_discovery.data.synthetic_ecg import LEAD_NAMES

        lead_names = LEAD_NAMES[: array.shape[0]]
    lead_names = list(lead_names)

    if array.shape[0] == 1:
        wave_lead = array[0]
    elif config.delineation_lead in lead_names:
        wave_lead = array[lead_names.index(config.delineation_lead)]
    else:
        raise KeyError(
            f"delineation_lead {config.delineation_lead!r} is not among the "
            f"available leads {tuple(lead_names)}"
        )

    wave_signal = bandpass_filter(
        wave_lead, sampling_rate_hz,
        config.refine_band_low_hz, config.refine_band_high_hz, config.filter_order,
    )

    if config.qrs_boundary_source == "vector_magnitude" and array.shape[0] > 1:
        filtered = bandpass_filter(
            array, sampling_rate_hz,
            config.refine_band_low_hz, config.refine_band_high_hz,
            config.filter_order,
        )
        # Root-mean-square across leads. Because the deflections of one beat are
        # simultaneous across leads, this concentrates the complex rather than
        # smearing it, while averaging down independent per-lead noise.
        source = np.sqrt(np.mean(filtered ** 2, axis=0))
    else:
        source = wave_signal

    slope = np.abs(np.gradient(source)) * sampling_rate_hz
    smooth_samples = max(
        int(round(config.qrs_envelope_smooth_ms / 1000.0 * sampling_rate_hz)), 1
    )
    qrs_envelope = _moving_average(slope, smooth_samples)
    return wave_signal, qrs_envelope


# --------------------------------------------------------------------------- #
# Boundary search
# --------------------------------------------------------------------------- #
def _sustained_boundary(
    envelope: np.ndarray,
    start: int,
    limit: int,
    threshold: float,
    quiet_samples: int,
    direction: int,
) -> int:
    """Find where the envelope goes and stays quiet, walking from ``start``.

    A single sample below threshold is not enough. The waveform dips to zero
    slope between the Q, R and S deflections of one complex, so a boundary
    accepted on one quiet sample would land inside the complex. Requiring the
    envelope to stay below threshold for ``quiet_samples`` consecutive samples
    distinguishes "the complex is over" from "the complex is momentarily flat".

    Falls back to ``limit`` - the edge of the physiologically plausible search
    window - if the envelope never settles, so a boundary is always returned.
    """
    quiet = 0
    index = start
    while (direction > 0 and index <= limit) or (direction < 0 and index >= limit):
        if envelope[index] < threshold:
            quiet += 1
            if quiet >= quiet_samples:
                # Step back to the first sample of the quiet run: that is where
                # the wave actually ended.
                return index - direction * (quiet_samples - 1)
        else:
            quiet = 0
        index += direction
    return limit


def _wave_extent(
    signal: np.ndarray,
    peak: int,
    baseline: float,
    lower: int,
    upper: int,
    threshold_fraction: float,
    quiet_samples: int = 1,
    max_half_width: int | None = None,
) -> tuple[int, int]:
    """Edges of a wave, where its deflection settles back towards baseline.

    Used for the P wave and as the fallback for T-wave boundaries. The edge is
    the point past which the deflection stays within ``threshold_fraction`` of
    the wave's own peak height for ``quiet_samples`` consecutive samples.

    The sustained requirement is not cosmetic. A P wave is only about 0.1 mV
    tall, so a threshold at 20% of its height is roughly 0.02 mV - comparable to
    the baseline noise of a real recording. Accepting the *first* sample below
    that threshold worked on clean synthetic data and failed badly on PTB-XL,
    where the walk simply never terminated: 22% of P-wave onsets ran all the way
    to the edge of the search window, producing PR intervals pinned at the
    window ceiling and P waves over 200 ms wide, which is physiologically
    impossible.

    ``max_half_width`` additionally caps how far the search may travel from the
    peak, so a wave can never be reported wider than physiology allows even if
    the signal never settles.
    """
    amplitude = signal[peak] - baseline
    if amplitude == 0:
        return peak, peak
    cutoff = abs(amplitude) * threshold_fraction

    if max_half_width is not None:
        lower = max(lower, peak - max_half_width)
        upper = min(upper, peak + max_half_width)

    quiet_samples = max(int(quiet_samples), 1)
    near_baseline = np.abs(signal - baseline) <= cutoff

    def settle(direction: int, limit: int) -> int:
        run = 0
        index = peak
        while (index - direction) >= lower and (index - direction) <= upper and index != limit:
            index += direction
            if index < 0 or index >= signal.size:
                break
            if near_baseline[index]:
                run += 1
                if run >= quiet_samples:
                    # The wave ended at the first sample of the quiet run, which
                    # is the one nearest the peak.
                    return index - direction * (quiet_samples - 1)
            else:
                run = 0
        return limit

    return settle(-1, lower), settle(+1, upper)


def _tangent_boundary(
    signal: np.ndarray,
    peak: int,
    limit: int,
    baseline: float,
    fallback: int,
    direction: int,
) -> int:
    """A wave boundary by the tangent (Lepeschkin) method.

    A tangent is taken at the steepest point of the wave's limb - descending
    for an offset, ascending for an onset - and extended to where it meets the
    isoelectric line; that intersection is the boundary.

    This is the conventional manual technique for T-wave offset, and it is
    preferred to a plain amplitude threshold because a T wave approaches
    baseline *gradually*. A fixed fraction-of-peak threshold therefore fires
    well inside the wave: for a raised-cosine T wave, a 15% threshold crosses
    about 13% of the way in, which at a 140 ms T wave is an 18 ms error before
    any noise is considered. It is also unstable, since a small error in the
    assumed baseline moves a shallow crossing a long way. The steepest-slope
    point suffers from neither problem.

    Applied symmetrically to both limbs here: the same argument holds for
    T-wave onset, which rises out of the ST segment just as gradually as the
    offset descends into baseline.

    Parameters
    ----------
    direction:
        ``+1`` to search forward from the peak (offset), ``-1`` backward
        (onset).
    fallback:
        Boundary to return when no usable tangent exists - a flat limb, or too
        few samples to differentiate.
    """
    if direction > 0:
        if limit <= peak + 2:
            return fallback
        segment = signal[peak : limit + 1]
        base_index = peak
    else:
        if peak - limit < 2:
            return fallback
        segment = signal[limit : peak + 1]
        base_index = limit

    derivative = np.gradient(segment)
    # The limb that carries the wave back towards baseline. Going forwards that
    # means falling towards it; going backwards it means rising away from it, so
    # the sign flips with direction.
    towards_baseline = -direction * np.sign(signal[peak] - baseline) * derivative
    steepest = int(np.argmax(towards_baseline))
    if towards_baseline[steepest] <= 0:
        return fallback

    index = base_index + steepest
    slope = derivative[steepest]
    if slope == 0:
        return fallback

    crossing = index + (baseline - signal[index]) / slope
    if not np.isfinite(crossing):
        return fallback

    # Clamp between the steepest-slope point and the edge of the search window.
    # When those bounds cross - the steepest point is already at the window edge,
    # leaving no room for a boundary beyond it - there is no usable tangent and
    # the caller's fallback is returned rather than a bound that would sit
    # outside the wave's own search range.
    low, high = (index + 1, limit) if direction > 0 else (limit, index - 1)
    if low > high:
        return fallback
    return int(min(max(round(crossing), low), high))


# --------------------------------------------------------------------------- #
# Delineation
# --------------------------------------------------------------------------- #
def delineate_beats(
    signal_array: np.ndarray,
    sampling_rate_hz: float,
    r_peaks: np.ndarray,
    config: SignalProcessingConfig | None = None,
    lead_names: Sequence[str] | None = None,
) -> list[BeatDelineation]:
    """Locate P, QRS and T wave boundaries for every detected beat.

    Parameters
    ----------
    signal_array:
        ``(n_samples,)`` or ``(n_leads, n_samples)`` recording in millivolts.
    sampling_rate_hz:
        Sampling rate. Delineation accuracy is limited by this: at 100 Hz one
        sample is 10 ms, which is why interval features are measured at 500 Hz.
    r_peaks:
        R-peak sample indices, from
        :func:`~ecg_discovery.signal_processing.qrs_detection.detect_r_peaks`.
    config:
        Delineation parameters; defaults are used if omitted.
    lead_names:
        Lead names of ``signal_array``; defaults to the standard 12-lead order.

    Returns
    -------
    list[BeatDelineation]
        One entry per R peak, in time order. A beat whose QRS boundaries cannot
        be established is skipped; a beat whose P or T wave cannot be found is
        returned with that wave set to ``None``.
    """
    config = config or SignalProcessingConfig()
    r_peaks = np.asarray(r_peaks, dtype=np.int64)
    if r_peaks.size == 0:
        return []

    wave_signal, qrs_envelope = delineation_signals(
        signal_array, sampling_rate_hz, config, lead_names
    )
    n_samples = wave_signal.size

    def to_samples(milliseconds: float) -> int:
        return int(round(milliseconds / 1000.0 * sampling_rate_hz))

    qrs_search = to_samples(config.qrs_search_ms)
    quiet_samples = max(to_samples(config.quiet_run_ms), 1)
    baseline_window = max(to_samples(config.baseline_window_ms), 1)
    p_search_min = to_samples(config.p_search_min_ms)
    p_search_max = to_samples(config.p_search_max_ms)
    t_search_min = to_samples(config.t_search_min_ms)
    t_search_max = to_samples(config.t_search_max_ms)

    # RR interval preceding each beat, used to scale the T-wave search window:
    # repolarisation genuinely takes longer when the heart beats more slowly.
    if r_peaks.size >= 2:
        intervals = np.diff(r_peaks)
        rr_per_beat = np.concatenate([[intervals[0]], intervals])
    else:
        rr_per_beat = np.array([int(sampling_rate_hz)])

    results: list[BeatDelineation] = []
    for position, r_peak in enumerate(r_peaks):
        r_peak = int(r_peak)
        if not (0 <= r_peak < n_samples):
            continue

        # -- QRS boundaries ---------------------------------------------------
        low = max(r_peak - qrs_search, 0)
        high = min(r_peak + qrs_search, n_samples - 1)
        if high - low < 3:
            continue
        peak_slope = float(qrs_envelope[low : high + 1].max())
        if peak_slope <= 0:
            continue
        threshold = config.qrs_boundary_threshold * peak_slope

        qrs_onset = _sustained_boundary(
            qrs_envelope, r_peak, low, threshold, quiet_samples, direction=-1
        )
        qrs_offset = _sustained_boundary(
            qrs_envelope, r_peak, high, threshold, quiet_samples, direction=+1
        )
        if qrs_offset <= qrs_onset:
            continue

        # -- Baseline from the PR segment -------------------------------------
        # The stretch just before the complex is electrically silent, so it is
        # the natural reference level for measuring wave amplitudes.
        base_hi = max(qrs_onset - 1, 0)
        base_lo = max(base_hi - baseline_window, 0)
        baseline = (
            float(np.median(wave_signal[base_lo : base_hi + 1]))
            if base_hi > base_lo
            else float(wave_signal[base_hi])
        )

        # -- P wave -----------------------------------------------------------
        p_wave = _find_p_wave(
            wave_signal, qrs_onset, baseline, p_search_min, p_search_max,
            previous_offset=(results[-1].t_wave.offset if results and results[-1].t_wave else 0),
            config=config, quiet_samples=quiet_samples,
            max_half_width=max(to_samples(config.p_max_duration_ms) // 2, 1),
        )

        # -- T wave -----------------------------------------------------------
        rr = int(rr_per_beat[position])
        t_window_end = min(
            qrs_offset + max(int(config.t_search_frac_rr * rr), t_search_min + 1),
            qrs_offset + t_search_max,
            n_samples - 1,
        )
        # Do not run into the next beat's complex.
        if position + 1 < r_peaks.size:
            t_window_end = min(t_window_end, int(r_peaks[position + 1]) - qrs_search // 2)
        t_wave = _find_t_wave(
            wave_signal, qrs_offset, t_window_end, baseline, t_search_min, config
        )

        results.append(
            BeatDelineation(
                r_peak=r_peak,
                qrs=WaveBoundaries(onset=qrs_onset, peak=r_peak, offset=qrs_offset),
                p_wave=p_wave,
                t_wave=t_wave,
                baseline_mv=baseline,
            )
        )
    return results


def _find_p_wave(
    signal: np.ndarray,
    qrs_onset: int,
    baseline: float,
    search_min: int,
    search_max: int,
    previous_offset: int,
    config: SignalProcessingConfig,
    quiet_samples: int = 1,
    max_half_width: int | None = None,
) -> WaveBoundaries | None:
    """Locate the P wave in the window before QRS onset.

    Returns ``None`` when nothing rises far enough above baseline to be a P
    wave. That is reported rather than guessed at: atrial fibrillation genuinely
    has no organised P wave, and inventing one would put a fictitious PR
    interval into the known-feature set.
    """
    upper = qrs_onset - search_min
    lower = max(qrs_onset - search_max, previous_offset, 0)
    if upper - lower < 3:
        return None

    window = signal[lower : upper + 1] - baseline
    peak = lower + int(np.argmax(np.abs(window)))
    if abs(signal[peak] - baseline) < config.p_min_amplitude_mv:
        return None

    onset, offset = _wave_extent(
        signal, peak, baseline, lower, upper, config.p_boundary_threshold,
        quiet_samples=quiet_samples, max_half_width=max_half_width,
    )
    if offset <= onset:
        return None
    return WaveBoundaries(onset=onset, peak=peak, offset=offset)


def _find_t_wave(
    signal: np.ndarray,
    qrs_offset: int,
    window_end: int,
    baseline: float,
    search_min: int,
    config: SignalProcessingConfig,
) -> WaveBoundaries | None:
    """Locate the T wave after QRS offset, ending it by the configured method."""
    lower = qrs_offset + search_min
    upper = window_end
    if upper - lower < 3:
        return None

    window = signal[lower : upper + 1] - baseline
    peak = lower + int(np.argmax(np.abs(window)))
    if abs(signal[peak] - baseline) < config.t_min_amplitude_mv:
        return None

    threshold_onset, threshold_offset = _wave_extent(
        signal, peak, baseline, qrs_offset, upper, config.t_boundary_threshold
    )
    if config.t_offset_method == "tangent":
        onset = _tangent_boundary(
            signal, peak, qrs_offset, baseline, threshold_onset, direction=-1
        )
        offset = _tangent_boundary(
            signal, peak, upper, baseline, threshold_offset, direction=+1
        )
    else:
        onset, offset = threshold_onset, threshold_offset

    if offset <= onset:
        return None
    return WaveBoundaries(onset=onset, peak=peak, offset=offset)
