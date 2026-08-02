"""Null controls for segment attribution: is this the model, or just the signal?

THE PROBLEM
-----------
A segment attribution profile is easy to over-read. The QRS complex is by far
the largest deflection on an ECG, so gradients there are large whatever the
model has learned. "The model attends to the QRS complex" may be a fact about
the arithmetic of the signal rather than a fact about the model.

This is not hypothetical. On the synthetic cohort where the age signal was
carried **entirely by T-wave morphology**, and where we therefore know the
answer, the measured profile was::

    share    P 0.019  QRS 0.340  T 0.464  other 0.178   <- correctly favours T
    density  P 0.0008 QRS 0.0136 T 0.0128 other 0.0014  <- still favours QRS

Attribution *share* found the right structure. Attribution *density* pointed at
the QRS even though the model's information was entirely in the T wave. A
density-based claim on real data would have been an amplitude artifact.

THE CONTROLS
------------
Three null profiles, each isolating a different way a result could be spurious:

``amplitude``
    Share of total signal energy per segment. This is what attribution would
    look like if it were simply proportional to how large the deflection is.
    The cheapest and most direct test of the confound above.

``untrained``
    The same architecture at random initialisation. It has never seen data, so
    any structure in its attribution profile comes from the architecture and the
    signal, not from anything learned. Gradients are non-trivial, which makes
    this a stricter control than a degenerate model.

``shuffled``
    Trained on permuted age labels. It undergoes real training dynamics but
    cannot learn a genuine age relationship. The strongest control, and the most
    expensive - it needs a full training run.

INTERPRETATION
--------------
A trained-model profile that matches all three nulls is **not a finding**: it
says the attribution reflects the signal's own structure. A profile that
departs from all three is evidence the model is attending to something specific.

Note that :meth:`ControlComparison.summary_text` reports the weaker disjunctive
form as well - which controls the profile departs from, if any - because a
partial result is worth seeing. The conjunctive reading above is the one that
supports a claim; a departure from one control out of three does not.

The comparison is paired per recording - the same recordings, the same
delineation, differing only in what produced the attribution - so the interval
on the difference reflects genuine variation rather than between-cohort noise.
Intervals are Bonferroni-corrected across every segment-by-control test in the
table, because the verdict is read off the whole table rather than off a cell
nominated in advance.

**Statistical departure is a weak criterion here, by construction.** The
interval is on a mean over recordings, so at PTB-XL scale its width shrinks to
a few thousandths of a share and nearly any systematic difference clears it.
That is why this module reports effect sizes and the amplitude correlation
beside the verdict, and why a departure found here is treated as a pointer to
be tested causally - by occlusion, which asks what the model *needs* rather
than what it merely attends to - and never as a finding on its own.

This module is offered as a contribution in its own right. The question *"does
this attribution reflect the model or the amplitude structure of the input?"*
applies to any attribution over a physiological signal, and it is not routinely
asked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from ecg_discovery.interpretability.attribution import integrated_gradients
from ecg_discovery.interpretability.fiducial_attribution import (
    SEGMENT_NAMES,
    aggregate_by_fiducial_segment,
    segment_masks,
)
from ecg_discovery.signal_processing.wave_delineation import BeatDelineation

__all__ = ["ControlProfile", "ControlComparison", "amplitude_profile",
           "model_profile", "compare_against_controls"]


@dataclass
class ControlProfile:
    """Per-recording segment shares produced by one method."""

    name: str
    share: np.ndarray            # (n_recordings, n_segments)
    density: np.ndarray

    @property
    def mean_share(self) -> np.ndarray:
        """Mean share per segment across recordings."""
        return np.nanmean(self.share, axis=0)

    @property
    def mean_density(self) -> np.ndarray:
        return np.nanmean(self.density, axis=0)


@dataclass
class ControlComparison:
    """A trained model's attribution profile set against its null controls."""

    trained: ControlProfile
    controls: dict[str, ControlProfile]
    segment_names: tuple[str, ...] = SEGMENT_NAMES
    confidence_level: float = 0.95

    @property
    def n_comparisons(self) -> int:
        """How many segment-by-control tests the verdict is drawn from.

        Every segment is compared against every control, and the family that
        matters for multiplicity is all of them together - a claim of the form
        "the trained profile departs from its nulls" is read off the whole
        table, not off one cell chosen in advance.
        """
        return max(len(self.segment_names) * len(self.controls), 1)

    def _critical_value(self) -> float:
        """Normal quantile for the Bonferroni-corrected two-sided interval.

        The uncorrected 1.96 that stood here tested four segments against three
        controls - twelve tests - at a nominal 5% each. Sibling modules in this
        repository correct for multiplicity carefully, and a headline drawn from
        this table should be held to the same standard.

        The correction is sized for how :meth:`summary_text` actually reports:
        it names *any* control the profile departs from on *any* segment, so the
        family is the whole table and all twelve cells are live. Note that this
        is deliberately stricter than the reading in the module docstring, where
        the evidence is departure from **every** control. That conjunctive form
        is an intersection-union test - requiring all three to clear the bar is
        already conservative and needs no correction across controls at all - so
        a profile that departs from all three under this critical value has
        cleared a higher bar than the interpretation section demands. The
        stricter of the two is used because it is the one that protects the
        weaker claim, and the difference costs only interval width.
        """
        tail = (1.0 - self.confidence_level) / self.n_comparisons / 2.0
        return float(norm.ppf(1.0 - tail))

    def difference(self, control: str) -> pd.DataFrame:
        """Paired per-segment difference between trained and one control.

        Rows with an interval excluding zero are where the trained model's
        attention genuinely departs from that null. The interval is on the
        *mean* difference, via the standard error, because the question is
        whether the average profile differs rather than whether every recording
        does - and its width is Bonferroni-corrected across every segment and
        control in the comparison.

        A caution that no interval can express: with several thousand
        recordings the standard error is small enough that almost any
        systematic difference clears the threshold, correction or not. Effect
        size is the thing to read here, and
        :meth:`amplitude_correlation` is the more informative check of whether
        the profile says anything about the model at all.
        """
        null = self.controls[control]
        critical = self._critical_value()
        rows = []
        for index, segment in enumerate(self.segment_names):
            trained_share = self.trained.share[:, index]
            null_share = null.share[:, index]
            # One mask drives all three columns. Delineation can fail on a
            # recording, and averaging each profile over every recording it
            # happens to be defined on - while differencing only the ones where
            # both are - lets `delta` disagree with `trained - control` in its
            # own row, which reads as an arithmetic error in the output table.
            # `n` is reported because it varies by segment.
            paired = np.isfinite(trained_share) & np.isfinite(null_share)
            if paired.sum() < 2:
                continue
            delta = trained_share[paired] - null_share[paired]
            stderr = float(delta.std(ddof=1) / np.sqrt(delta.size))
            mean = float(delta.mean())
            rows.append({
                "segment": segment,
                "n": int(paired.sum()),
                "trained": float(trained_share[paired].mean()),
                control: float(null_share[paired].mean()),
                "delta": mean,
                "ci_low": mean - critical * stderr,
                "ci_high": mean + critical * stderr,
                "differs": bool(abs(mean) > critical * stderr),
            })
        return pd.DataFrame(rows)

    def amplitude_correlation(self) -> float:
        """Correlation between the trained profile and the amplitude null.

        Computed across recordings and segments. A value near 1 means the
        attribution is essentially a restatement of where the signal is large,
        and no claim about model attention survives.
        """
        if "amplitude" not in self.controls:
            return float("nan")
        trained = self.trained.share.ravel()
        amplitude = self.controls["amplitude"].share.ravel()
        ok = np.isfinite(trained) & np.isfinite(amplitude)
        return float(np.corrcoef(trained[ok], amplitude[ok])[0, 1])

    def summary_text(self) -> str:
        """Report whether the trained profile survives its controls."""
        lines = [
            f"Segment attribution against null controls "
            f"({self.trained.share.shape[0]} recordings)",
            "",
            f"{'segment':<10}" + "".join(f"{name:>13}" for name in
                                         ["trained", *self.controls]),
        ]
        for index, segment in enumerate(self.segment_names):
            row = f"{segment:<10}{self.trained.mean_share[index]:>13.3f}"
            for control in self.controls.values():
                row += f"{control.mean_share[index]:>13.3f}"
            lines.append(row)

        correlation = self.amplitude_correlation()
        lines += [
            "",
            f"correlation with the amplitude null: r = {correlation:.3f}",
            f"intervals: Bonferroni-corrected across {self.n_comparisons} "
            f"segment-by-control tests "
            f"(z = {self._critical_value():.2f}, {len(self.segment_names)} segments "
            f"x {len(self.controls)} controls)",
        ]

        survived = []
        for name in self.controls:
            frame = self.difference(name)
            differing = frame[frame["differs"]]["segment"].tolist()
            if differing:
                survived.append(f"{name} (differs on {', '.join(differing)})")
        lines.append("")
        if survived:
            lines.append(
                "The trained profile departs from: " + "; ".join(survived) + "."
            )
            lines.append(
                f"NOTE: with {self.trained.share.shape[0]} recordings the standard "
                "error on a mean share is small, so 'differs' is easy to clear and "
                "says little on its own. Read the effect size in the delta column, "
                "and treat a departure as a pointer to be tested causally - by "
                "occlusion - rather than as a finding in itself."
            )
        else:
            lines.append(
                "The trained profile does NOT depart from any control. The "
                "attribution reflects the signal's own structure, not learned "
                "attention - no claim about what the model attends to is "
                "supported."
            )
        if correlation > 0.9:
            lines.append(
                f"CAUTION: r = {correlation:.3f} against the amplitude null means "
                "this profile largely restates where the ECG is biggest. Segment "
                "rankings should not be interpreted as model attention."
            )
        return "\n".join(lines)


def amplitude_profile(
    signals: np.ndarray,
    beats_per_recording: Sequence[Sequence[BeatDelineation]],
    lead_names: Sequence[str],
) -> ControlProfile:
    """Segment share of raw signal energy - the amplitude null.

    Answers: what would the attribution profile look like if attribution were
    simply proportional to how large the signal is?
    """
    shares, densities = [], []
    for signal, beats in zip(signals, beats_per_recording):
        magnitude = np.abs(np.asarray(signal, dtype=np.float64))
        # Reuse the attribution aggregator with signal magnitude in place of
        # attribution, so the two profiles are computed identically.
        result = aggregate_by_fiducial_segment(magnitude, beats, lead_names)
        shares.append(result.share.sum(axis=0))
        densities.append(result.density.mean(axis=0))
    return ControlProfile("amplitude", np.array(shares), np.array(densities))


def model_profile(
    model,
    normalised_signals: np.ndarray,
    sexes: np.ndarray,
    beats_per_recording: Sequence[Sequence[BeatDelineation]],
    lead_names: Sequence[str],
    name: str,
    n_steps: int = 32,
    batch_size: int = 32,
) -> ControlProfile:
    """Segment attribution profile for one model over a set of recordings."""
    shares, densities = [], []
    for start in range(0, len(normalised_signals), batch_size):
        stop = min(start + batch_size, len(normalised_signals))
        attribution = integrated_gradients(
            model,
            torch.tensor(normalised_signals[start:stop]),
            torch.tensor(sexes[start:stop], dtype=torch.float32),
            n_steps=n_steps,
            # An untrained or shuffled model can sit almost flat, making the
            # relative completeness check meaningless; the profile is still
            # well defined, so the warning is suppressed for controls.
            max_relative_error=None if name != "trained" else 0.05,
        )
        for offset in range(stop - start):
            result = aggregate_by_fiducial_segment(
                attribution.attributions[offset],
                beats_per_recording[start + offset],
                lead_names,
            )
            shares.append(result.share.sum(axis=0))
            densities.append(result.density.mean(axis=0))
    return ControlProfile(name, np.array(shares), np.array(densities))


def compare_against_controls(
    trained_model,
    normalised_signals: np.ndarray,
    raw_signals: np.ndarray,
    sexes: np.ndarray,
    beats_per_recording: Sequence[Sequence[BeatDelineation]],
    lead_names: Sequence[str],
    untrained_model=None,
    shuffled_model=None,
    n_steps: int = 32,
) -> ControlComparison:
    """Compute the trained profile and every available null control.

    Parameters
    ----------
    trained_model:
        The model whose attention is in question.
    normalised_signals:
        Model-space input, as the model was trained on.
    raw_signals:
        Millivolt signals, for the amplitude null. Kept separate because
        normalisation changes relative segment energy.
    untrained_model, shuffled_model:
        Optional controls. ``untrained_model`` is cheap - the same architecture
        at random initialisation. ``shuffled_model`` requires a full training
        run on permuted labels and is the strongest control.
    """
    trained = model_profile(
        trained_model, normalised_signals, sexes, beats_per_recording,
        lead_names, "trained", n_steps,
    )
    controls: dict[str, ControlProfile] = {
        "amplitude": amplitude_profile(raw_signals, beats_per_recording, lead_names)
    }
    if untrained_model is not None:
        controls["untrained"] = model_profile(
            untrained_model, normalised_signals, sexes, beats_per_recording,
            lead_names, "untrained", n_steps,
        )
    if shuffled_model is not None:
        controls["shuffled"] = model_profile(
            shuffled_model, normalised_signals, sexes, beats_per_recording,
            lead_names, "shuffled", n_steps,
        )
    return ControlComparison(trained=trained, controls=controls)
