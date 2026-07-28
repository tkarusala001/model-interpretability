"""From-scratch R-peak detection, in the style of Pan & Tompkins (1985).

WHAT AN R PEAK IS AND WHY IT MATTERS HERE
-----------------------------------------
The QRS complex is the sharp spike in an ECG: the moment the ventricles
depolarise and contract. Its tallest point is the R peak. Almost everything
downstream is measured relative to it - the P wave is the bump before it, the
T wave the bump after, heart rate is the reciprocal of the spacing between
consecutive R peaks. If R-peak localisation is wrong, every interval feature is
wrong, and the Phase 8 claim about how much of the age gap is "already known"
is measured against a corrupted yardstick. That is why this module is validated
against constructed ground truth before it is trusted anywhere.

THE ALGORITHM
-------------
Pan & Tompkins is the standard approach and works by progressively transforming
the signal until beats become unmissable bumps, then tracking an adaptive
threshold across them:

1. **Bandpass 5-15 Hz.** The QRS complex is the fastest normal feature of an
   ECG. Below this band lie P and T waves and baseline wander; above it lies
   muscle noise. Filtering to it leaves the QRS dominant.
2. **Differentiate.** The QRS is defined by its steep slopes, so the derivative
   emphasises it further and suppresses whatever slow content survived step 1.
3. **Square.** Makes everything positive (an R wave may deflect either way) and
   amplifies large deflections relative to small ones.
4. **Moving-window integration** over ~150 ms, roughly the width of the widest
   normal QRS. This merges the separate Q, R and S deflections of one beat into
   a single smooth energy bump, so each beat produces one candidate rather than
   three.
5. **Adaptive dual thresholds.** Running estimates of typical signal-peak and
   noise-peak amplitudes set a threshold that drifts with the recording, so a
   patient with small QRS amplitudes is handled as well as one with large.
   Two mechanisms guard the obvious failure modes: a refractory period plus a
   T-wave slope test to avoid counting tall T waves as beats, and a search-back
   pass that re-examines long gaps with a halved threshold to recover beats the
   threshold missed.

DELIBERATE DEVIATIONS FROM THE 1985 PAPER
-----------------------------------------
Both are consequences of processing complete recordings offline rather than a
live monitor stream, and both are stated here because they change the numbers:

- **Zero-phase filtering.** The original uses causal filters, which necessarily
  delay the signal, and then corrects for that delay. We process finished
  10-second recordings, so we can filter forwards and backwards
  (``scipy.signal.filtfilt``), which has no phase distortion at all. This
  removes group-delay error from R-peak positions - worth having, since
  localisation error propagates into every interval measurement.
- **Central-difference derivative.** The paper's five-point derivative
  coefficients are tuned for 200 Hz sampling. A plain central difference is
  sampling-rate agnostic, which matters because this project measures intervals
  at 500 Hz while training the model at 100 Hz.

``scipy.signal`` supplies the filtering and local-maximum primitives; the
detection logic - thresholds, refractory handling, T-wave rejection,
search-back - is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import signal as sp_signal

from ecg_discovery.config import SignalProcessingConfig

__all__ = [
    "QRSDetection",
    "DetectionScore",
    "bandpass_filter",
    "detect_r_peaks",
    "score_detection",
]


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QRSDetection:
    """R-peak positions plus the intermediate signals that produced them.

    The intermediate stages are retained because a detector that silently
    returns the wrong answer is far more dangerous here than one that fails
    loudly: being able to plot the integrated signal against the threshold is
    how a disagreement with ground truth gets diagnosed. See
    ``notebooks/00_signal_processing_sanity_checks.ipynb``.

    Attributes
    ----------
    r_peaks:
        Sample indices of detected R peaks, ascending.
    sampling_rate_hz:
        Sampling rate of the analysed signal.
    filtered:
        The 5-15 Hz bandpassed detection signal.
    integrated:
        The moving-window-integrated energy envelope thresholding runs on.
    thresholds:
        Value of the adaptive primary threshold at each candidate peak.
    candidate_peaks:
        Every local maximum considered, including those rejected.
    polarity:
        Which way the R wave was taken to deflect (+1 or -1).
    """

    r_peaks: np.ndarray
    sampling_rate_hz: float
    filtered: np.ndarray = field(repr=False)
    integrated: np.ndarray = field(repr=False)
    thresholds: np.ndarray = field(repr=False)
    candidate_peaks: np.ndarray = field(repr=False)
    polarity: int = 1

    @property
    def n_beats(self) -> int:
        """Number of detected beats."""
        return int(self.r_peaks.size)

    def rr_intervals_ms(self) -> np.ndarray:
        """Intervals between consecutive R peaks, in milliseconds."""
        if self.r_peaks.size < 2:
            return np.empty(0, dtype=np.float64)
        return np.diff(self.r_peaks) / self.sampling_rate_hz * 1000.0

    def heart_rate_bpm(self) -> float:
        """Mean heart rate, or NaN if fewer than two beats were detected."""
        rr = self.rr_intervals_ms()
        if rr.size == 0:
            return float("nan")
        return float(60_000.0 / np.mean(rr))


@dataclass(frozen=True)
class DetectionScore:
    """Agreement between detected R peaks and a reference set.

    ``sensitivity`` is the fraction of true beats that were found;
    ``ppv`` (positive predictive value) is the fraction of detections that were
    real beats. Both matter and they trade off: a detector can reach perfect
    sensitivity by firing constantly.

    ``mean_absolute_error_ms`` is the localisation accuracy over matched beats -
    the quantity that propagates into every interval measurement, and therefore
    the number this project actually cares most about.
    """

    n_reference: int
    n_detected: int
    n_matched: int
    sensitivity: float
    ppv: float
    f1: float
    mean_absolute_error_ms: float
    max_absolute_error_ms: float
    errors_ms: np.ndarray = field(repr=False)


# --------------------------------------------------------------------------- #
# Filtering primitives
# --------------------------------------------------------------------------- #
def bandpass_filter(
    x: np.ndarray,
    sampling_rate_hz: float,
    low_hz: float,
    high_hz: float,
    order: int = 2,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    Applied forwards and backwards (``filtfilt``), so the output has no phase
    distortion and features stay where they were in time. That property is the
    reason this is used instead of a causal filter: a shifted R peak would shift
    every interval measured from it.

    If the requested upper cutoff reaches the Nyquist frequency it is lowered to
    just below it and only the low cut is applied there, rather than raising -
    at 100 Hz the standard 40 Hz ECG passband is close to Nyquist and this keeps
    the same configuration usable at both of the project's sampling rates.
    """
    nyquist = sampling_rate_hz / 2.0
    if low_hz <= 0 or low_hz >= nyquist:
        raise ValueError(
            f"low_hz must lie in (0, {nyquist}) for a {sampling_rate_hz} Hz signal, "
            f"got {low_hz}"
        )
    high = min(high_hz, nyquist * 0.99)

    # filtfilt needs a signal comfortably longer than the filter's transient.
    padlen = 3 * (2 * order + 1)
    if x.shape[-1] <= padlen:
        raise ValueError(
            f"signal of {x.shape[-1]} samples is too short for an order-{order} "
            f"zero-phase filter (needs more than {padlen})"
        )

    if high <= low_hz:
        sos = sp_signal.butter(order, low_hz / nyquist, btype="highpass", output="sos")
    else:
        sos = sp_signal.butter(
            order, [low_hz / nyquist, high / nyquist], btype="bandpass", output="sos"
        )
    return sp_signal.sosfiltfilt(sos, x, axis=-1)


def _moving_window_integrate(x: np.ndarray, window_samples: int) -> np.ndarray:
    """Centred moving average - the energy envelope Pan & Tompkins thresholds.

    Kept centred (rather than trailing, as in the original real-time design) so
    the envelope's peak sits over the beat rather than after it, consistent with
    the zero-phase filtering used elsewhere in this module.
    """
    window_samples = max(int(window_samples), 1)
    kernel = np.ones(window_samples, dtype=np.float64) / window_samples
    pad = window_samples // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="same")[pad : pad + x.size]


def _select_lead(
    signal_array: np.ndarray,
    lead_names: Sequence[str] | None,
    detection_lead: str,
) -> np.ndarray:
    """Pick the single lead detection runs on, accepting 1-D or 12-lead input."""
    array = np.asarray(signal_array, dtype=np.float64)
    if array.ndim == 1:
        return array
    if array.ndim != 2:
        raise ValueError(
            f"expected a 1-D lead or a 2-D (n_leads, n_samples) array, got shape "
            f"{array.shape}"
        )
    if lead_names is None:
        from ecg_discovery.data.synthetic_ecg import LEAD_NAMES

        lead_names = LEAD_NAMES[: array.shape[0]]
    if detection_lead not in lead_names:
        raise KeyError(
            f"detection_lead {detection_lead!r} is not among the available leads "
            f"{tuple(lead_names)}"
        )
    return array[list(lead_names).index(detection_lead)]


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def detect_r_peaks(
    signal_array: np.ndarray,
    sampling_rate_hz: float,
    config: SignalProcessingConfig | None = None,
    lead_names: Sequence[str] | None = None,
) -> QRSDetection:
    """Detect R peaks in an ECG recording.

    Parameters
    ----------
    signal_array:
        Either a single lead of shape ``(n_samples,)`` or a multi-lead recording
        of shape ``(n_leads, n_samples)``, in millivolts. For multi-lead input,
        detection runs on ``config.detection_lead``.
    sampling_rate_hz:
        Sampling rate of the recording.
    config:
        Detection parameters; defaults are used if omitted.
    lead_names:
        Names of the leads in ``signal_array``. Defaults to the standard
        12-lead order.

    Returns
    -------
    QRSDetection
        Detected peak positions plus intermediate signals for inspection.

    Notes
    -----
    Returns an empty detection rather than raising when the recording contains
    no detectable beats (a flat trace, or one shorter than a single beat). A
    silent recording is a real thing that happens to real electrodes, and
    downstream code already has to handle a recording whose beats could not be
    measured.
    """
    config = config or SignalProcessingConfig()
    lead = _select_lead(signal_array, lead_names, config.detection_lead)
    n_samples = lead.size

    empty = QRSDetection(
        r_peaks=np.empty(0, dtype=np.int64),
        sampling_rate_hz=sampling_rate_hz,
        filtered=np.zeros(n_samples),
        integrated=np.zeros(n_samples),
        thresholds=np.empty(0),
        candidate_peaks=np.empty(0, dtype=np.int64),
    )

    min_samples = int(sampling_rate_hz * 0.5)
    if n_samples < max(min_samples, 3 * (2 * config.filter_order + 1) + 1):
        return empty
    if not np.isfinite(lead).all() or np.ptp(lead) <= 0:
        return empty

    # -- Stages 1-4: transform beats into isolated energy bumps ---------------
    filtered = bandpass_filter(
        lead, sampling_rate_hz,
        config.qrs_band_low_hz, config.qrs_band_high_hz, config.filter_order,
    )
    derivative = np.gradient(filtered) * sampling_rate_hz
    squared = derivative ** 2
    integration_samples = max(
        int(round(config.integration_window_ms / 1000.0 * sampling_rate_hz)), 1
    )
    integrated = _moving_window_integrate(squared, integration_samples)

    if not np.isfinite(integrated).all() or integrated.max() <= 0:
        return empty

    # -- Candidate beats: local maxima no closer than the refractory period ---
    refractory_samples = max(
        int(round(config.refractory_ms / 1000.0 * sampling_rate_hz)), 1
    )
    candidates, _ = sp_signal.find_peaks(integrated, distance=refractory_samples)
    if candidates.size == 0:
        return empty

    # -- Stage 5: adaptive thresholding --------------------------------------
    # The T-wave rule needs waveform steepness, which must be read from the
    # differentiated signal rather than from `integrated`: the integrator's
    # output is at a local maximum at every candidate, hence locally flat there,
    # so its slope carries no information about what kind of wave caused it.
    accepted, thresholds = _threshold_candidates(
        integrated, np.abs(derivative), candidates, sampling_rate_hz, config
    )
    if not accepted:
        return empty

    # -- Refine peak positions on a morphology-preserving signal --------------
    refined_signal = bandpass_filter(
        lead, sampling_rate_hz,
        config.refine_band_low_hz, config.refine_band_high_hz, config.filter_order,
    )
    polarity = _infer_polarity(refined_signal, accepted, sampling_rate_hz, config)
    r_peaks = _refine_peaks(
        refined_signal, accepted, sampling_rate_hz, config, polarity
    )

    return QRSDetection(
        r_peaks=r_peaks,
        sampling_rate_hz=sampling_rate_hz,
        filtered=filtered,
        integrated=integrated,
        thresholds=np.asarray(thresholds, dtype=np.float64),
        candidate_peaks=candidates.astype(np.int64),
        polarity=polarity,
    )


def _threshold_candidates(
    integrated: np.ndarray,
    steepness: np.ndarray,
    candidates: np.ndarray,
    sampling_rate_hz: float,
    config: SignalProcessingConfig,
) -> tuple[list[int], list[float]]:
    """Walk the candidate peaks in time, deciding which are beats.

    This is the adaptive core of Pan & Tompkins. Two running estimates are
    maintained - the typical amplitude of a signal peak and of a noise peak -
    and each is updated towards whichever category a candidate falls into::

        estimate <- adaptation * observed + (1 - adaptation) * estimate

    The acceptance threshold sits a fixed fraction of the way from the noise
    estimate up to the signal estimate, so it tracks the recording's own
    amplitude rather than any absolute millivolt value. This is what lets one
    parameter set work across patients with very different QRS amplitudes.

    Two safeguards run alongside it:

    *T-wave rejection.* A tall T wave can clear the threshold on amplitude
    alone. But a T wave is a slow repolarisation bump while a QRS is a fast
    depolarisation spike, so the two are separable by *steepness* even when
    their heights are comparable: a candidate arriving suspiciously soon after a
    beat and rising at less than half the previous beat's slope is rejected.
    Steepness is read from ``steepness`` - the magnitude of the differentiated
    signal - and not from ``integrated``, which is at a local maximum (and hence
    locally flat) at every candidate and so cannot discriminate anything.

    *Search-back.* If a gap much longer than the recent average RR interval
    opens up, the detector re-examines it against a halved threshold. This
    recovers a beat whose amplitude briefly dipped - a missed beat corrupts two
    RR intervals at once, so it is worth a second look.

    The original scans sample by sample; scanning pre-computed local maxima is
    equivalent, since only local maxima can ever be accepted, and is far easier
    to read.
    """
    learning_samples = max(int(config.learning_seconds * sampling_rate_hz), 1)
    learning_region = integrated[:learning_samples]
    if learning_region.size == 0:
        learning_region = integrated

    # Initialise the running estimates from the learning phase, as the paper does.
    signal_peak = float(np.max(learning_region)) * 0.25
    noise_peak = float(np.mean(learning_region)) * 0.5
    if signal_peak <= noise_peak:
        signal_peak = noise_peak + 1e-12

    def primary_threshold() -> float:
        return noise_peak + config.threshold_fraction * (signal_peak - noise_peak)

    refractory_samples = int(round(config.refractory_ms / 1000.0 * sampling_rate_hz))
    t_wave_samples = int(
        round(config.t_wave_discrimination_ms / 1000.0 * sampling_rate_hz)
    )
    slope_half_window = max(int(0.05 * sampling_rate_hz), 1)

    def local_slope(index: int) -> float:
        """Steepest rise near a candidate - a QRS's defining characteristic."""
        lo = max(index - slope_half_window, 0)
        hi = min(index + slope_half_window, steepness.size - 1)
        if hi <= lo:
            return 0.0
        return float(np.max(steepness[lo : hi + 1]))

    accepted: list[int] = []
    thresholds: list[float] = []
    rr_history: list[float] = []
    rejected_since_last: list[int] = []
    last_slope = 0.0

    for candidate in candidates:
        peak_value = float(integrated[candidate])
        threshold = primary_threshold()
        thresholds.append(threshold)

        # Search-back: has too long a gap opened since the last accepted beat?
        if accepted and rr_history:
            mean_rr = float(np.mean(rr_history[-config.rr_history :]))
            gap = candidate - accepted[-1]
            if gap > config.searchback_factor * mean_rr and rejected_since_last:
                secondary = threshold * 0.5
                revivable = [
                    index for index in rejected_since_last
                    if integrated[index] > secondary
                    and index - accepted[-1] > refractory_samples
                ]
                if revivable:
                    best = max(revivable, key=lambda i: integrated[i])
                    accepted.append(best)
                    rr_history.append(float(best - accepted[-2]))
                    # A search-back beat is real signal, but was found with a
                    # relaxed threshold, so it updates the estimate more gently.
                    signal_peak = 0.25 * integrated[best] + 0.75 * signal_peak
                    last_slope = local_slope(best)
                    rejected_since_last = [
                        i for i in rejected_since_last if i > best
                    ]

        is_beat = peak_value > threshold

        if is_beat and accepted:
            distance = candidate - accepted[-1]
            if distance < refractory_samples:
                is_beat = False                       # physiologically impossible
            elif distance < t_wave_samples:
                slope = local_slope(candidate)
                if slope < config.t_wave_slope_fraction * last_slope:
                    is_beat = False                   # too slow to be a QRS

        if is_beat:
            if accepted:
                rr_history.append(float(candidate - accepted[-1]))
            accepted.append(candidate)
            signal_peak = (
                config.threshold_adaptation * peak_value
                + (1.0 - config.threshold_adaptation) * signal_peak
            )
            last_slope = local_slope(candidate)
            rejected_since_last = []
        else:
            noise_peak = (
                config.threshold_adaptation * peak_value
                + (1.0 - config.threshold_adaptation) * noise_peak
            )
            rejected_since_last.append(int(candidate))

    accepted.sort()
    return accepted, thresholds


def _infer_polarity(
    refined: np.ndarray,
    accepted: Sequence[int],
    sampling_rate_hz: float,
    config: SignalProcessingConfig,
) -> int:
    """Decide whether R waves deflect up or down in this lead.

    The same beat is a tall positive spike in lead II and a deep negative one in
    aVR or V1, so peak refinement has to know which extremum to look for. In
    ``auto`` mode the decision is made once for the whole recording by comparing
    the largest positive and negative excursions near detected beats - a global
    choice, because polarity is a property of the lead's orientation relative to
    the heart, not of an individual beat.
    """
    if config.polarity == "positive":
        return 1
    if config.polarity == "negative":
        return -1

    half_window = max(int(config.refine_window_ms / 1000.0 * sampling_rate_hz / 2), 1)
    positive = 0.0
    negative = 0.0
    for index in accepted:
        lo = max(index - half_window, 0)
        hi = min(index + half_window, refined.size)
        segment = refined[lo:hi]
        if segment.size:
            positive = max(positive, float(segment.max()))
            negative = max(negative, float(-segment.min()))
    return 1 if positive >= negative else -1


def _refine_peaks(
    refined: np.ndarray,
    accepted: Sequence[int],
    sampling_rate_hz: float,
    config: SignalProcessingConfig,
    polarity: int,
) -> np.ndarray:
    """Move each detection to the true R-peak sample.

    Thresholding operates on an energy envelope that has been squared and
    smoothed over 150 ms, so its maximum indicates *that* a beat happened, not
    exactly when. The precise instant is recovered by taking the extremum of the
    0.5-40 Hz signal - which preserves QRS shape - within a short window around
    the detection.
    """
    half_window = max(int(config.refine_window_ms / 1000.0 * sampling_rate_hz / 2), 1)
    oriented = refined * polarity

    peaks: list[int] = []
    for index in accepted:
        lo = max(index - half_window, 0)
        hi = min(index + half_window + 1, oriented.size)
        if hi <= lo:
            peaks.append(int(index))
            continue
        peaks.append(int(lo + np.argmax(oriented[lo:hi])))

    # Refinement can collapse two nearby detections onto the same sample.
    unique = np.array(sorted(set(peaks)), dtype=np.int64)

    # Drop beats too close to either end to be complete. A recording starts and
    # stops mid-rhythm, so its edge complexes are typically cut in half; keeping
    # a truncated one would add a spuriously short RR interval and bias the
    # measured heart rate. See SignalProcessingConfig.edge_guard_ms.
    guard = int(round(config.edge_guard_ms / 1000.0 * sampling_rate_hz))
    if guard > 0:
        unique = unique[(unique >= guard) & (unique < refined.size - guard)]
    return unique


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_detection(
    detected: np.ndarray,
    reference: np.ndarray,
    sampling_rate_hz: float,
    tolerance_ms: float = 50.0,
) -> DetectionScore:
    """Score detected R peaks against a reference set.

    Matching is one-to-one and greedy in time: each reference beat claims the
    nearest unclaimed detection within ``tolerance_ms``. One-to-one matching
    matters because it stops a detector that fires twice per beat from scoring
    as if it were perfect - the second detection becomes a false positive and
    lowers the positive predictive value, which is exactly what should happen.

    Parameters
    ----------
    detected, reference:
        Sample indices, in any order (both are sorted internally).
    tolerance_ms:
        How far apart a detection and a reference beat may be and still count as
        the same beat.
    """
    detected = np.sort(np.asarray(detected, dtype=np.int64))
    reference = np.sort(np.asarray(reference, dtype=np.int64))
    tolerance_samples = tolerance_ms / 1000.0 * sampling_rate_hz

    errors: list[float] = []
    unmatched = list(detected)
    for ref in reference:
        if not unmatched:
            break
        distances = [abs(candidate - ref) for candidate in unmatched]
        best = int(np.argmin(distances))
        if distances[best] <= tolerance_samples:
            errors.append((unmatched[best] - ref) / sampling_rate_hz * 1000.0)
            unmatched.pop(best)

    n_matched = len(errors)
    n_reference = int(reference.size)
    n_detected = int(detected.size)
    sensitivity = n_matched / n_reference if n_reference else float("nan")
    ppv = n_matched / n_detected if n_detected else float("nan")
    f1 = (
        2 * sensitivity * ppv / (sensitivity + ppv)
        if n_matched and (sensitivity + ppv) > 0
        else 0.0
    )
    errors_array = np.array(errors, dtype=np.float64)

    return DetectionScore(
        n_reference=n_reference,
        n_detected=n_detected,
        n_matched=n_matched,
        sensitivity=sensitivity,
        ppv=ppv,
        f1=f1,
        mean_absolute_error_ms=(
            float(np.mean(np.abs(errors_array))) if errors else float("nan")
        ),
        max_absolute_error_ms=(
            float(np.max(np.abs(errors_array))) if errors else float("nan")
        ),
        errors_ms=errors_array,
    )
