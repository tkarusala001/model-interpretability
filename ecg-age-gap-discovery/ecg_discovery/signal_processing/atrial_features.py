"""Atrial (P-wave) measurements beyond duration, amplitude and PR interval.

WHY THIS MODULE EXISTS
----------------------
This project's central claim is that the apparent size of a discovery depends on
how thoroughly prior knowledge was enumerated. Applied honestly, that claim binds
our own positive result: reporting that a model depends on atrial information
"beyond classical P-wave measurement" is only as strong as the list of classical
measurements it was tested against. Three of them - duration, amplitude and PR
interval - is a short list. Cardiology routinely reads more from the P wave than
those three numbers.

This module adds the measures a cardiologist would reach for next:

``p_terminal_force_v1_mv_ms``
    The Morris index: the terminal negative deflection of the P wave in lead V1,
    quantified as its duration times its depth. The standard marker of left
    atrial abnormality, and the single most likely candidate to explain an
    age-related atrial signal.
``p_area_ii_mv_ms``
    Net area under the P wave in lead II. Duration and amplitude each capture one
    dimension of the wave; area is the product actually related to atrial mass,
    and is not recoverable from the other two unless the shape is fixed.
``p_notch_depth_mv``
    Depth of the notch between the two humps of a bifid P wave in lead II. Left
    and right atrial depolarisation are near-simultaneous in a healthy heart;
    when conduction between the atria slows they separate, and the P wave splits.
    Neither duration nor amplitude detects this directly.
``p_dispersion_ms``
    Maximum minus minimum P-wave duration across the twelve leads. See the
    reliability warning below - this one is not like the others.

THE FIRST THREE ARE CHEAP; THE FOURTH IS NOT
--------------------------------------------
Terminal force, area and notch depth all read a *different lead* inside the *same
time window* the existing delineation already produces. They need no new boundary
detection, so they inherit the accuracy of the P onset/offset already validated
against synthetic ground truth.

P-wave dispersion is a different kind of measurement. It requires P onset and
offset to be found *independently in each of twelve leads*, which is precisely
the operation `wave_delineation` documents as its least reliable
(see its module docstring, "single-lead measurement understates global
durations"). Worse, the synthetic cohort cannot validate it: every synthetic lead
is an exact linear projection of the same three components, so true inter-lead
dispersion is identically zero there by construction. Any dispersion measured on
synthetic data is therefore pure method noise - which makes it a direct estimate
of the measurement floor, and that is how :func:`dispersion_noise_floor` uses it.

We compute the feature anyway rather than declining to. A number with a measured
reliability attached is an argument; an omission is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import SignalProcessingConfig
from .wave_delineation import BeatDelineation, _wave_extent

__all__ = [
    "ATRIAL_FEATURE_NAMES",
    "DispersionReliability",
    "atrial_features",
    "per_lead_p_durations",
    "dispersion_noise_floor",
]

#: Extended atrial measurements, in reporting order.
ATRIAL_FEATURE_NAMES: tuple[str, ...] = (
    "p_terminal_force_v1_mv_ms",
    "p_area_ii_mv_ms",
    "p_notch_depth_mv",
    "p_dispersion_ms",
)

#: Local maxima shallower than this fraction of the P wave's own height are
#: treated as noise rather than as the second hump of a bifid P wave. A real P
#: wave is only ~0.1 mV tall, so an absolute threshold would be either useless on
#: small P waves or blind on large ones.
_NOTCH_PROMINENCE_FRACTION = 0.08

#: Smoothing applied before notch detection. Long enough to remove sample-level
#: noise, short enough to preserve a genuine inter-atrial notch, which is tens of
#: milliseconds wide.
_NOTCH_SMOOTH_MS = 8.0


@dataclass(frozen=True)
class DispersionReliability:
    """How much of a measured P-wave dispersion is real, and how much is noise.

    Attributes
    ----------
    noise_floor_ms:
        Dispersion measured on recordings whose true dispersion is known to be
        zero. Anything at or below this is indistinguishable from measurement
        error.
    observed_sd_ms:
        Between-recording standard deviation of measured dispersion on the real
        cohort. If this is not comfortably above ``noise_floor_ms``, the feature
        carries no usable between-subject signal.
    usable:
        Whether the observed spread exceeds the noise floor by enough to treat
        the feature as a measurement rather than as noise.
    """

    noise_floor_ms: float
    observed_sd_ms: float
    n_synthetic: int
    n_real: int

    @property
    def usable(self) -> bool:
        """Whether between-subject spread clears the measurement floor."""
        if not np.isfinite([self.noise_floor_ms, self.observed_sd_ms]).all():
            return False
        return self.observed_sd_ms > 2.0 * self.noise_floor_ms

    def summary_text(self) -> str:
        """One-paragraph verdict, for reports and for the paper."""
        verdict = (
            "usable: between-subject spread clears the measurement floor"
            if self.usable else
            "NOT USABLE: the spread between recordings is not distinguishable "
            "from the error of the measurement itself"
        )
        return (
            f"P-wave dispersion reliability\n"
            f"  noise floor (zero-dispersion cohort): {self.noise_floor_ms:6.2f} ms "
            f"(n={self.n_synthetic})\n"
            f"  observed between-recording SD:        {self.observed_sd_ms:6.2f} ms "
            f"(n={self.n_real})\n"
            f"  {verdict}"
        )


def _lead_baseline(
    array: np.ndarray, lead: int, beat: BeatDelineation, baseline_window: int
) -> float:
    """Isoelectric level for one lead, from its own PR segment.

    Each lead needs its own reference: baseline offset is a property of the
    electrode, so borrowing lead II's baseline for lead V1 would bias every
    amplitude measured there.
    """
    high = max(beat.qrs.onset - 1, 0)
    low = max(high - baseline_window, 0)
    return float(np.median(array[lead, low : high + 1])) if high > low else 0.0


def _terminal_force(trace: np.ndarray, samples_to_ms: float) -> float:
    """Morris index for one beat: duration times depth of the terminal negativity.

    ``trace`` is the P wave in lead V1, baseline-subtracted. The P wave in V1 is
    normally biphasic - right atrial depolarisation moves towards the electrode
    and left atrial depolarisation away from it - so the terminal portion is
    negative. A deeper, longer terminal negativity means a larger or slower left
    atrium.

    Returned negative by convention, so that more abnormal is more negative.
    Returns 0.0 when the P wave has no terminal negative component at all, which
    is a genuine observation and not a failure to measure.
    """
    if trace.size == 0:
        return float("nan")
    negative = trace < 0.0
    if not negative[-1]:
        # The wave ends above baseline: no terminal negative deflection exists.
        return 0.0
    # Walk back from the end while the trace stays below baseline.
    start = trace.size
    while start > 0 and negative[start - 1]:
        start -= 1
    run = trace[start:]
    duration_ms = run.size * samples_to_ms
    depth_mv = float(-run.min())
    return -(duration_ms * depth_mv)


def _notch_depth(
    trace: np.ndarray, smooth_samples: int
) -> float:
    """Depth of the notch between two humps of a bifid P wave, in millivolts.

    Returns 0.0 for a single-humped P wave. That is a measurement, not a missing
    value: most P waves are not notched, and coding them NaN would drop every
    normal recording from the analysis.

    Noise is the enemy here. An unsmoothed P wave has many local maxima, so a
    naive peak count reports every recording as notched. We smooth first, then
    require the second hump to rise a minimum fraction of the wave's own height
    above the trough between them.
    """
    if trace.size < 5:
        return float("nan")

    height = float(np.abs(trace).max())
    if height <= 0.0:
        return 0.0

    # P waves are predominantly positive in lead II; measure the notch on the
    # dominant polarity so an inverted P wave is handled the same way.
    oriented = trace if trace[np.argmax(np.abs(trace))] >= 0 else -trace

    if smooth_samples > 1:
        kernel = np.ones(smooth_samples) / smooth_samples
        smoothed = np.convolve(oriented, kernel, mode="same")
    else:
        smoothed = oriented

    interior = smoothed[1:-1]
    is_max = (interior > smoothed[:-2]) & (interior >= smoothed[2:])
    peaks = np.flatnonzero(is_max) + 1
    if peaks.size < 2:
        return 0.0

    # The two tallest peaks define the candidate notch; the trough is the
    # minimum between them.
    tallest = peaks[np.argsort(smoothed[peaks])[-2:]]
    left, right = int(min(tallest)), int(max(tallest))
    if right - left < 2:
        return 0.0
    trough = float(smoothed[left:right + 1].min())
    depth = float(min(smoothed[left], smoothed[right]) - trough)

    if depth < _NOTCH_PROMINENCE_FRACTION * height:
        return 0.0
    return depth


def per_lead_p_durations(
    signal_array: np.ndarray,
    beats: Sequence[BeatDelineation],
    sampling_rate_hz: float,
    config: SignalProcessingConfig,
) -> np.ndarray:
    """P-wave duration measured independently in every lead, in milliseconds.

    Returns an ``(n_beats, n_leads)`` array with NaN where no P wave cleared the
    amplitude threshold in that lead. This is the operation P-wave dispersion
    requires, and the one this project's delineation is least able to do well:
    the search window comes from the reference-lead delineation, but the peak and
    both boundaries are found in each lead's own signal, where the P wave may be
    a few hundredths of a millivolt tall.
    """
    array = np.asarray(signal_array, dtype=np.float64)
    if array.ndim != 2 or not beats:
        return np.empty((0, 0))

    samples_to_ms = 1000.0 / sampling_rate_hz
    baseline_window = max(int(round(config.baseline_window_ms / 1000.0 * sampling_rate_hz)), 1)
    quiet_samples = max(int(round(config.quiet_run_ms / 1000.0 * sampling_rate_hz)), 1)
    max_half_width = max(int(round(config.p_max_duration_ms / 1000.0 * sampling_rate_hz)) // 2, 1)

    durations = np.full((len(beats), array.shape[0]), np.nan)
    for b, beat in enumerate(beats):
        if beat.p_wave is None:
            continue
        # Search inside the reference delineation's P window, widened by half the
        # permitted width so a lead whose P starts earlier is not truncated by
        # the reference lead's own boundary.
        pad = max_half_width
        lower = max(beat.p_wave.onset - pad, 0)
        upper = min(beat.p_wave.offset + pad, array.shape[1] - 1)
        if upper - lower < 3:
            continue

        for lead in range(array.shape[0]):
            baseline = _lead_baseline(array, lead, beat, baseline_window)
            window = array[lead, lower : upper + 1] - baseline
            peak = lower + int(np.argmax(np.abs(window)))
            if abs(array[lead, peak] - baseline) < config.p_min_amplitude_mv:
                continue
            onset, offset = _wave_extent(
                array[lead], peak, baseline, lower, upper,
                config.p_boundary_threshold,
                quiet_samples=quiet_samples, max_half_width=max_half_width,
            )
            if offset > onset:
                durations[b, lead] = (offset - onset) * samples_to_ms
    return durations


def dispersion_noise_floor(
    per_lead_durations: Sequence[np.ndarray],
) -> float:
    """Median dispersion measured where the true dispersion is known to be zero.

    Pass per-lead duration arrays from a cohort whose leads are exact linear
    projections of shared components - the synthetic generator's output. Every
    lead there has identical wave timing, so true dispersion is zero and whatever
    is measured is the method's own error.
    """
    values: list[float] = []
    for durations in per_lead_durations:
        spread = _dispersion_from_durations(durations)
        if np.isfinite(spread):
            values.append(spread)
    return float(np.median(values)) if values else float("nan")


def _dispersion_from_durations(durations: np.ndarray) -> float:
    """Max minus min P duration across leads, aggregated over beats."""
    if durations.size == 0:
        return float("nan")
    per_beat: list[float] = []
    for row in np.atleast_2d(durations):
        finite = row[np.isfinite(row)]
        # Dispersion across two leads is not a meaningful spread; require a
        # majority of the twelve to have yielded a measurement.
        if finite.size >= max(3, row.size // 2):
            per_beat.append(float(finite.max() - finite.min()))
    return float(np.median(per_beat)) if per_beat else float("nan")


def atrial_features(
    signal_array: np.ndarray,
    beats: Sequence[BeatDelineation],
    lead_names: Sequence[str],
    sampling_rate_hz: float,
    config: SignalProcessingConfig,
    include_dispersion: bool = True,
) -> dict[str, float]:
    """Extended P-wave measurements for one recording.

    Returns NaN for any quantity whose required lead is absent or whose P wave
    was never found, rather than substituting a value: a fabricated measurement
    would enter the known-feature set as though it had been observed, and this
    project exists to compare against what was genuinely known.

    ``include_dispersion`` exists because per-lead delineation costs twelve
    boundary searches per beat, and callers that will not use the feature should
    not pay for it.
    """
    nan = float("nan")
    empty = dict.fromkeys(ATRIAL_FEATURE_NAMES, nan)
    array = np.asarray(signal_array, dtype=np.float64)
    if array.ndim != 2 or not beats:
        return empty

    lead_index = {name: i for i, name in enumerate(lead_names) if i < array.shape[0]}
    samples_to_ms = 1000.0 / sampling_rate_hz
    baseline_window = max(int(round(config.baseline_window_ms / 1000.0 * sampling_rate_hz)), 1)
    smooth_samples = max(int(round(_NOTCH_SMOOTH_MS / 1000.0 * sampling_rate_hz)), 1)

    terminal_force: list[float] = []
    area: list[float] = []
    notch: list[float] = []

    for beat in beats:
        if beat.p_wave is None:
            continue
        window = slice(beat.p_wave.onset, beat.p_wave.offset + 1)

        if "V1" in lead_index:
            lead = lead_index["V1"]
            trace = array[lead, window] - _lead_baseline(array, lead, beat, baseline_window)
            terminal_force.append(_terminal_force(trace, samples_to_ms))

        if "II" in lead_index:
            lead = lead_index["II"]
            trace = array[lead, window] - _lead_baseline(array, lead, beat, baseline_window)
            area.append(float(np.sum(trace)) * samples_to_ms)
            notch.append(_notch_depth(trace, smooth_samples))

    def median_of(values: Sequence[float]) -> float:
        finite = [v for v in values if v is not None and np.isfinite(v)]
        return float(np.median(finite)) if finite else nan

    result = {
        "p_terminal_force_v1_mv_ms": median_of(terminal_force),
        "p_area_ii_mv_ms": median_of(area),
        "p_notch_depth_mv": median_of(notch),
        "p_dispersion_ms": nan,
    }
    if include_dispersion:
        result["p_dispersion_ms"] = _dispersion_from_durations(
            per_lead_p_durations(array, beats, sampling_rate_hz, config)
        )
    return result
