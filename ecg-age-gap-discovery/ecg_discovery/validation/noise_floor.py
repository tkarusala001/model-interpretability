"""How much of an age gap is training noise rather than signal.

WHY THIS EXISTS
---------------
The unexplained share of an age gap is reported as the ceiling on a discovery
claim. But part of that share is not eligible to be a discovery at all: it is
the model's own estimation error. Train the same architecture on the same data
with a different seed and the per-recording gaps differ, and no ECG measurement
can explain a difference that would not survive retraining.

Writing the gap as a reproducible component plus estimation noise,
``g = s + eps`` with ``Cov(s, eps) = 0``, the noise share is

    nu = Var(eps) / Var(g)

and for two independently seeded models with the same variance decomposition,
``Corr(g1, g2) = Var(s)/Var(g) = 1 - nu``. So the correlation between the age
gaps of two seeds estimates the noise floor directly, with no ground truth
required.

WHAT IT UNDERSTATES, AND WHY THAT IS THE SAFE DIRECTION
--------------------------------------------------------
Seeds differ in initialisation and batch order, but share the training set.
Estimation error driven by the *data* rather than the seed is therefore
reproducible across seeds, counts as ``s`` here, and is not captured. This
estimate is consequently a **lower bound** on the true noise share.

That is the conservative direction, and it matters that it is. The bound the
paper reports is ``N <= U - nu``: a smaller ``nu`` makes the bound looser, so
understating the noise floor cannot make a discovery claim look better than it
is. Overstating it could, which is why no attempt is made to inflate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

__all__ = ["NoiseFloor", "estimate_noise_floor"]


@dataclass(frozen=True)
class NoiseFloor:
    """The share of age-gap variance attributable to the training seed."""

    nu: float
    nu_ci: tuple[float, float]
    mean_correlation: float
    pairwise: dict[tuple[str, str], float]
    n_recordings: int
    n_seeds: int

    def corrected_unexplained(self, unexplained: float) -> float:
        """The unexplained share with the noise floor removed.

        This is the quantity a discovery claim is actually about: what is left
        after subtracting both existing knowledge and the model's own error.
        Floored at zero, since a negative eligible share is not meaningful.
        """
        return max(unexplained - self.nu, 0.0)

    def summary_text(self) -> str:
        low, high = self.nu_ci
        lines = [
            f"Noise floor over {self.n_recordings} recordings shared by "
            f"{self.n_seeds} independently seeded models.",
            f"  mean cross-seed correlation of age gaps : {self.mean_correlation:.3f}",
            f"  nu (share that is training noise)       : {self.nu:.1%} "
            f"[{low:.1%}, {high:.1%}]",
            "",
            "Pairwise correlations:",
        ]
        for (a, b), r in sorted(self.pairwise.items()):
            lines.append(f"  {a} vs {b}: r = {r:.3f}")
        lines += [
            "",
            "This is a LOWER bound on the noise share: seeds share a training set, "
            "so error driven by the data rather than the seed is reproducible "
            "across seeds and is not counted here. That is the safe direction - "
            "understating nu loosens the bound N <= U - nu rather than tightening "
            "it, so it cannot make a discovery claim look stronger than it is.",
        ]
        return "\n".join(lines)


def estimate_noise_floor(
    gaps_by_seed: Mapping[str, Sequence[float]],
    bootstrap_iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> NoiseFloor:
    """Estimate the seed-noise share from age gaps of independently trained models.

    Parameters
    ----------
    gaps_by_seed:
        Per-recording age gaps, keyed by a label for each training run. Every
        array must be aligned to the same recordings in the same order; the
        caller is responsible for that alignment, because silently intersecting
        differently-ordered runs is an easy way to compute a correlation
        between unrelated patients.
    bootstrap_iterations, confidence_level, seed:
        Percentile bootstrap over recordings for the interval on ``nu``.

    Returns
    -------
    NoiseFloor
    """
    labels = list(gaps_by_seed)
    if len(labels) < 2:
        raise ValueError(
            f"need at least 2 training runs to separate signal from seed noise, "
            f"got {len(labels)}"
        )
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in gaps_by_seed.items()}
    lengths = {k: a.size for k, a in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"age-gap arrays have differing lengths: {lengths}")
    n = next(iter(lengths.values()))
    if n < 3:
        raise ValueError(f"need at least 3 recordings, got {n}")

    stacked = np.column_stack([arrays[k] for k in labels])
    if not np.isfinite(stacked).all():
        raise ValueError("age gaps contain non-finite values")

    def mean_corr(matrix: np.ndarray) -> float:
        values = []
        for i, j in combinations(range(matrix.shape[1]), 2):
            a, b = matrix[:, i], matrix[:, j]
            if a.std() == 0 or b.std() == 0:
                continue
            values.append(float(np.corrcoef(a, b)[0, 1]))
        return float(np.mean(values)) if values else float("nan")

    pairwise = {
        (labels[i], labels[j]): float(np.corrcoef(stacked[:, i], stacked[:, j])[0, 1])
        for i, j in combinations(range(len(labels)), 2)
    }
    correlation = mean_corr(stacked)

    rng = np.random.default_rng(seed)
    draws = np.empty(bootstrap_iterations)
    for i in range(bootstrap_iterations):
        rows = rng.integers(0, n, n)
        draws[i] = 1.0 - mean_corr(stacked[rows])
    tail = (1.0 - confidence_level) / 2.0

    return NoiseFloor(
        # A correlation above 1 is impossible and below 0 would mean the seeds
        # disagree systematically, which is not a noise share; clip so the
        # reported figure is always a fraction of variance.
        nu=float(np.clip(1.0 - correlation, 0.0, 1.0)),
        nu_ci=(
            float(np.clip(np.nanpercentile(draws, 100 * tail), 0.0, 1.0)),
            float(np.clip(np.nanpercentile(draws, 100 * (1 - tail)), 0.0, 1.0)),
        ),
        mean_correlation=correlation,
        pairwise=pairwise,
        n_recordings=int(n),
        n_seeds=len(labels),
    )
