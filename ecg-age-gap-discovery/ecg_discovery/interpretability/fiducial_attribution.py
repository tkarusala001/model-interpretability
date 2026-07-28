"""Aggregating per-sample attribution into cardiologist-legible ECG segments.

THE PROBLEM THIS SOLVES
-----------------------
Raw attribution over a 12-lead ECG is 12,000 numbers for a ten-second recording
at 100 Hz. Rendered as a heatmap it is unreadable, and worse, it is not
*checkable*: a cardiologist cannot agree or disagree with a coloured strip.
Interpretability output that a domain expert cannot argue with is not evidence
of anything.

An ECG is not an undifferentiated time series. It is a repeating structure of
named parts - the P wave, the QRS complex, the T wave - each corresponding to a
distinct physiological event, each with an established clinical literature. This
module projects attribution onto that structure, so the output becomes a
statement of the form *"38% of this prediction's attribution falls in the QRS
complex, concentrated in leads V1 to V3"* - a claim a clinician can evaluate,
compare against what is known about ageing hearts, and reject.

This is an **adaptation of existing attribution methods to the structure of a
modality**, not a new attribution algorithm. The underlying numbers come from
standard Integrated Gradients; what is contributed here is the aggregation and
the care taken to make it not misleading.

THE CONFOUND THAT MAKES NAIVE AGGREGATION WRONG
-----------------------------------------------
Summing attribution within each segment and comparing the totals is the obvious
approach and it is misleading, because **the segments have very different
widths**. A T wave spans roughly 150 ms and a QRS complex roughly 90 ms, so at
equal per-sample importance the T wave collects about 70% more attribution
purely by being wider. A naive summary would report the T wave as the dominant
structure in essentially every recording, and that finding would be an artefact
of arithmetic rather than a property of the model.

Two complementary quantities are therefore reported for every segment, and
neither is sufficient alone:

``share``
    Fraction of the recording's total absolute attribution falling in that
    segment. This answers *"where does the model's evidence live?"* and is the
    right quantity for describing a prediction as a whole - a wide segment
    genuinely does carry more of the total.
``density``
    Mean absolute attribution **per sample**. This answers *"where is the model
    looking hardest?"* and is comparable across segments of different widths -
    the right quantity for asking which structure the model finds most
    informative.

A segment with a high share but ordinary density is merely wide. A segment with
high density is where the model is actually concentrating. Reporting only the
first would systematically overstate the T wave; reporting only the second would
understate how much of the prediction rests on it.

DIRECTION IS ALSO REPORTED
--------------------------
Attribution is signed: a sample can push the predicted age up or down. Magnitude
answers "how much did this matter", but sign answers "which way", and for an
age-gap analysis the direction is often the interesting part - whether a
structure made the heart look older or younger. ``signed_share`` keeps it.

KNOWN LIMITATIONS
-----------------
- Segment boundaries come from a single delineation applied to all twelve leads
  (see :mod:`ecg_discovery.signal_processing.wave_delineation`). Real ECGs have
  inter-lead timing dispersion, so a lead whose QRS begins slightly earlier will
  have a few of its QRS samples counted as isoelectric.
- Delineation error propagates directly. T-wave onset in particular is measured
  late by roughly 24 ms on synthetic data, which moves some early T-wave
  attribution into the ``other`` bucket. The effect is a modest, known
  underestimate of the T wave's share.
- A high share is not evidence that a segment carries real information. It says
  where the model looked. Whether that corresponds to anything independently
  verifiable is the question Phases 8 and 9 exist to answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ecg_discovery.signal_processing.wave_delineation import BeatDelineation

__all__ = [
    "SEGMENT_NAMES",
    "PRECORDIAL_LEADS",
    "LIMB_LEADS",
    "FiducialAttribution",
    "segment_masks",
    "aggregate_by_fiducial_segment",
    "summarise_cohort",
]

#: Segments in reporting order. ``other`` covers everything outside a detected
#: wave - the isoelectric baseline, the ST segment, and any beat whose waves
#: could not be delineated.
SEGMENT_NAMES: tuple[str, ...] = ("P", "QRS", "T", "other")

#: Standard lead groupings, for summaries such as "concentrated in V1-V3".
PRECORDIAL_LEADS: tuple[str, ...] = ("V1", "V2", "V3", "V4", "V5", "V6")
LIMB_LEADS: tuple[str, ...] = ("I", "II", "III", "aVR", "aVL", "aVF")


@dataclass(frozen=True)
class FiducialAttribution:
    """Attribution for one recording, organised by lead and cardiac segment.

    Attributes
    ----------
    share:
        ``(n_leads, n_segments)`` fraction of the recording's **total** absolute
        attribution. Sums to 1 across the whole array.
    density:
        ``(n_leads, n_segments)`` mean absolute attribution per sample,
        comparable across segments of unequal width.
    signed_share:
        ``(n_leads, n_segments)`` signed attribution as a fraction of total
        absolute attribution, so the direction of each segment's contribution is
        preserved.
    segment_samples:
        ``(n_segments,)`` number of samples assigned to each segment. Reported
        because the width difference between segments is exactly what makes
        ``share`` alone misleading.
    n_beats:
        Beats contributing to the aggregation.
    prediction, baseline_prediction:
        Model outputs, carried through so a summary is self-contained.
    """

    record_id: str | None
    lead_names: tuple[str, ...]
    segment_names: tuple[str, ...]
    share: np.ndarray
    density: np.ndarray
    signed_share: np.ndarray
    segment_samples: np.ndarray
    total_absolute: float
    n_beats: int
    prediction: float
    baseline_prediction: float

    @property
    def age_gap_direction(self) -> float:
        """How far the model moved from its baseline output, in years."""
        return float(self.prediction - self.baseline_prediction)

    def segment_share(self, segment: str, leads: Sequence[str] | None = None) -> float:
        """Fraction of total attribution in one segment, optionally within leads.

        Parameters
        ----------
        segment:
            One of :data:`SEGMENT_NAMES`.
        leads:
            Restrict to these leads, e.g. ``PRECORDIAL_LEADS``. All leads if
            omitted.
        """
        segment_index = self._segment_index(segment)
        lead_indices = self._lead_indices(leads)
        return float(self.share[lead_indices, segment_index].sum())

    def segment_density(self, segment: str, leads: Sequence[str] | None = None) -> float:
        """Mean per-sample attribution in one segment, optionally within leads."""
        segment_index = self._segment_index(segment)
        lead_indices = self._lead_indices(leads)
        return float(self.density[lead_indices, segment_index].mean())

    def dominant_segment(self, by: str = "density") -> str:
        """The segment the model weights most heavily.

        Defaults to ``density`` because that is the width-corrected measure;
        ranking by ``share`` mostly ranks segments by how wide they are.
        """
        values = self.density if by == "density" else self.share
        # Exclude 'other': it spans most of the recording and is not a structure.
        candidates = [
            (name, values[:, index].mean() if by == "density" else values[:, index].sum())
            for index, name in enumerate(self.segment_names)
            if name != "other"
        ]
        return max(candidates, key=lambda item: item[1])[0]

    def to_frame(self) -> pd.DataFrame:
        """Long-format table with one row per lead and segment."""
        rows = []
        for lead_index, lead in enumerate(self.lead_names):
            for segment_index, segment in enumerate(self.segment_names):
                rows.append({
                    "record_id": self.record_id,
                    "lead": lead,
                    "segment": segment,
                    "share": self.share[lead_index, segment_index],
                    "density": self.density[lead_index, segment_index],
                    "signed_share": self.signed_share[lead_index, segment_index],
                })
        return pd.DataFrame(rows)

    def summary_text(self) -> str:
        """A short, clinician-readable description of where attribution fell.

        Deliberately reports share and density together: share alone would make
        the widest segment look dominant in every recording.
        """
        lines = [
            f"Predicted age moved {self.age_gap_direction:+.1f} years from baseline; "
            f"{self.n_beats} beats delineated."
        ]
        for segment in self.segment_names:
            if segment == "other":
                continue
            share = self.segment_share(segment)
            density = self.segment_density(segment)
            best_lead = self.lead_names[
                int(np.argmax(self.share[:, self._segment_index(segment)]))
            ]
            lines.append(
                f"  {segment:>4}: {share:5.1%} of total attribution "
                f"(density {density:.3g} per sample), strongest in lead {best_lead}"
            )
        dominant = self.dominant_segment()
        lines.append(
            f"  Most concentrated attention: {dominant} "
            "(by per-sample density, which corrects for segment width)."
        )
        return "\n".join(lines)

    def _segment_index(self, segment: str) -> int:
        if segment not in self.segment_names:
            raise KeyError(
                f"unknown segment {segment!r}; expected one of {self.segment_names}"
            )
        return self.segment_names.index(segment)

    def _lead_indices(self, leads: Sequence[str] | None) -> list[int]:
        if leads is None:
            return list(range(len(self.lead_names)))
        missing = [lead for lead in leads if lead not in self.lead_names]
        if missing:
            raise KeyError(f"unknown lead(s) {missing}; available: {self.lead_names}")
        return [self.lead_names.index(lead) for lead in leads]


def segment_masks(
    n_samples: int, beats: Sequence[BeatDelineation]
) -> dict[str, np.ndarray]:
    """Boolean masks marking which samples belong to which cardiac segment.

    Every sample is assigned to exactly one segment. Where delineated waves
    would overlap - which the delineator's ordering guarantees should prevent,
    but which floating boundaries can still produce at beat edges - the QRS
    complex takes precedence, then the P wave, then the T wave. The QRS is
    ranked first because it is the shortest, sharpest and most reliably
    delineated of the three, so a contested sample is least likely to be
    misassigned by giving it to the QRS.

    Parameters
    ----------
    n_samples:
        Length of the recording.
    beats:
        Delineated beats, from
        :func:`~ecg_discovery.signal_processing.wave_delineation.delineate_beats`.

    Returns
    -------
    dict[str, numpy.ndarray]
        One boolean mask per entry of :data:`SEGMENT_NAMES`. The masks are
        disjoint and together cover every sample.
    """
    qrs = np.zeros(n_samples, dtype=bool)
    p_wave = np.zeros(n_samples, dtype=bool)
    t_wave = np.zeros(n_samples, dtype=bool)

    for beat in beats:
        qrs[max(beat.qrs.onset, 0) : beat.qrs.offset + 1] = True
        if beat.p_wave is not None:
            p_wave[max(beat.p_wave.onset, 0) : beat.p_wave.offset + 1] = True
        if beat.t_wave is not None:
            t_wave[max(beat.t_wave.onset, 0) : beat.t_wave.offset + 1] = True

    # Resolve overlaps by the precedence described above.
    p_wave &= ~qrs
    t_wave &= ~(qrs | p_wave)
    other = ~(qrs | p_wave | t_wave)

    return {"P": p_wave, "QRS": qrs, "T": t_wave, "other": other}


def aggregate_by_fiducial_segment(
    attributions: np.ndarray,
    beats: Sequence[BeatDelineation],
    lead_names: Sequence[str],
    prediction: float = float("nan"),
    baseline_prediction: float = float("nan"),
    record_id: str | None = None,
) -> FiducialAttribution:
    """Project per-sample attribution onto P, QRS, T and isoelectric segments.

    Parameters
    ----------
    attributions:
        ``(n_leads, n_samples)`` per-sample attribution for one recording, from
        :mod:`ecg_discovery.interpretability.attribution`.
    beats:
        Delineated beats for the same recording, at the same sampling rate.
    lead_names:
        Names of the leads, in the order they appear in ``attributions``.
    prediction, baseline_prediction:
        Model outputs, carried into the result so a summary stands alone.
    record_id:
        Optional identifier.

    Returns
    -------
    FiducialAttribution
        Share, density and signed share for every lead and segment.

    Notes
    -----
    Both ``share`` and ``density`` are returned because neither answers the
    question alone: share is dominated by how wide a segment is, density is
    width-corrected but says nothing about how much of the prediction rests on
    the segment overall. See the module docstring.
    """
    attributions = np.asarray(attributions, dtype=np.float64)
    if attributions.ndim != 2:
        raise ValueError(
            f"expected (n_leads, n_samples) attributions, got {attributions.shape}"
        )
    n_leads, n_samples = attributions.shape
    if len(lead_names) != n_leads:
        raise ValueError(
            f"lead_names has {len(lead_names)} entries but attributions have "
            f"{n_leads} leads"
        )

    masks = segment_masks(n_samples, beats)
    absolute = np.abs(attributions)
    total_absolute = float(absolute.sum())

    share = np.zeros((n_leads, len(SEGMENT_NAMES)))
    density = np.zeros((n_leads, len(SEGMENT_NAMES)))
    signed_share = np.zeros((n_leads, len(SEGMENT_NAMES)))
    segment_samples = np.zeros(len(SEGMENT_NAMES), dtype=np.int64)

    for segment_index, segment in enumerate(SEGMENT_NAMES):
        mask = masks[segment]
        segment_samples[segment_index] = int(mask.sum())
        if not mask.any():
            continue
        segment_absolute = absolute[:, mask]
        if total_absolute > 0:
            share[:, segment_index] = segment_absolute.sum(axis=1) / total_absolute
            signed_share[:, segment_index] = (
                attributions[:, mask].sum(axis=1) / total_absolute
            )
        density[:, segment_index] = segment_absolute.mean(axis=1)

    return FiducialAttribution(
        record_id=record_id,
        lead_names=tuple(lead_names),
        segment_names=SEGMENT_NAMES,
        share=share,
        density=density,
        signed_share=signed_share,
        segment_samples=segment_samples,
        total_absolute=total_absolute,
        n_beats=len(beats),
        prediction=float(prediction),
        baseline_prediction=float(baseline_prediction),
    )


def summarise_cohort(results: Sequence[FiducialAttribution]) -> pd.DataFrame:
    """Average share and density across recordings, per lead and segment.

    Returns
    -------
    pandas.DataFrame
        One row per lead and segment, with the mean and standard deviation of
        share and density across the cohort. The standard deviation is included
        because a segment that dominates in every recording and one that
        dominates on average while varying wildly are very different findings.
    """
    if not results:
        raise ValueError("cannot summarise an empty sequence of results")
    frame = pd.concat([result.to_frame() for result in results], ignore_index=True)
    summary = (
        frame.groupby(["lead", "segment"], sort=False)
        .agg(
            share_mean=("share", "mean"),
            share_sd=("share", "std"),
            density_mean=("density", "mean"),
            density_sd=("density", "std"),
            signed_share_mean=("signed_share", "mean"),
        )
        .reset_index()
    )
    return summary
