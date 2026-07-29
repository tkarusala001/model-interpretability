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

The comparison is paired per recording - the same recordings, the same
delineation, differing only in what produced the attribution - so the interval
on the difference reflects genuine variation rather than between-cohort noise.

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

    def difference(self, control: str) -> pd.DataFrame:
        """Paired per-segment difference between trained and one control.

        Rows with an interval excluding zero are where the trained model's
        attention genuinely departs from that null.
        """
        null = self.controls[control]
        tail = (1.0 - self.confidence_level) / 2.0
        rows = []
        for index, segment in enumerate(self.segment_names):
            delta = self.trained.share[:, index] - null.share[:, index]
            delta = delta[np.isfinite(delta)]
            if delta.size == 0:
                continue
            low, high = np.percentile(delta, [100 * tail, 100 * (1 - tail)])
            # Interval on the mean, via the standard error, rather than on the
            # spread of individual recordings - the question is whether the
            # average profile differs, not whether every recording does.
            stderr = delta.std(ddof=1) / np.sqrt(delta.size)
            rows.append({
                "segment": segment,
                "trained": float(self.trained.share[:, index].mean()),
                control: float(null.share[:, index].mean()),
                "delta": float(delta.mean()),
                "ci_low": float(delta.mean() - 1.96 * stderr),
                "ci_high": float(delta.mean() + 1.96 * stderr),
                "differs": bool(abs(delta.mean()) > 1.96 * stderr),
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
        lines += ["", f"correlation with the amplitude null: r = {correlation:.3f}"]

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
