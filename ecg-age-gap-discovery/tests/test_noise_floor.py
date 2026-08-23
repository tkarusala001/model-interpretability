"""Tests for the cross-seed noise-floor estimate.

The quantity under test enters the paper's reported bound directly, so it is
checked against constructed data where the true noise share is known by
construction, in both directions: gaps that are pure signal must give nu near
zero, and gaps that are pure noise must give nu near one.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecg_discovery.validation.noise_floor import NoiseFloor, estimate_noise_floor


def _seeded_gaps(n=800, nu=0.3, n_seeds=3, seed=0):
    """Gaps sharing one signal component, with independent per-seed noise."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(scale=np.sqrt(1.0 - nu), size=n)
    return {
        f"seed{i}": signal + rng.normal(scale=np.sqrt(nu), size=n)
        for i in range(n_seeds)
    }


def test_recovers_a_known_noise_share():
    """The headline requirement: nu must track the constructed value."""
    for true_nu in (0.1, 0.3, 0.6):
        result = estimate_noise_floor(_seeded_gaps(nu=true_nu), bootstrap_iterations=200)
        assert result.nu == pytest.approx(true_nu, abs=0.05), true_nu


def test_identical_models_have_no_noise_floor():
    """Perfectly reproducible gaps are all signal."""
    rng = np.random.default_rng(1)
    gap = rng.normal(size=500)
    result = estimate_noise_floor(
        {"a": gap, "b": gap.copy()}, bootstrap_iterations=200
    )
    assert result.nu == pytest.approx(0.0, abs=1e-6)
    assert result.mean_correlation == pytest.approx(1.0, abs=1e-6)


def test_unrelated_models_are_all_noise():
    """Gaps that share nothing carry no reproducible signal at all."""
    rng = np.random.default_rng(2)
    result = estimate_noise_floor(
        {"a": rng.normal(size=2000), "b": rng.normal(size=2000)},
        bootstrap_iterations=200,
    )
    assert result.nu == pytest.approx(1.0, abs=0.1)


def test_nu_is_never_reported_outside_zero_and_one():
    """A share of variance that is negative or above one is not a share.

    Anticorrelated runs would give 1 - r > 1, which would read as "more than
    all of the variance is noise" and, worse, would subtract a nonsense
    quantity from the reported unexplained share.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(size=600)
    result = estimate_noise_floor({"a": base, "b": -base}, bootstrap_iterations=200)
    assert 0.0 <= result.nu <= 1.0
    assert 0.0 <= result.nu_ci[0] <= result.nu_ci[1] <= 1.0


def test_more_seeds_average_the_pairwise_correlations():
    """With three runs, all three pairs are reported and averaged, not just one."""
    result = estimate_noise_floor(_seeded_gaps(n_seeds=3), bootstrap_iterations=200)
    assert len(result.pairwise) == 3
    assert result.n_seeds == 3
    assert result.mean_correlation == pytest.approx(
        float(np.mean(list(result.pairwise.values()))), abs=1e-9
    )


def test_interval_brackets_the_estimate():
    result = estimate_noise_floor(_seeded_gaps(nu=0.35), bootstrap_iterations=400)
    low, high = result.nu_ci
    assert low <= result.nu <= high
    assert high - low < 0.2, "interval implausibly wide at n=800"


def test_corrected_unexplained_subtracts_the_floor_and_stops_at_zero():
    """The eligible share is what the paper actually reports as a ceiling."""
    result = estimate_noise_floor(_seeded_gaps(nu=0.3), bootstrap_iterations=200)
    assert result.corrected_unexplained(0.65) == pytest.approx(0.65 - result.nu, abs=1e-9)
    assert result.corrected_unexplained(0.05) == 0.0


def test_a_single_run_cannot_estimate_a_noise_floor():
    """One model gives no way to separate its error from its signal."""
    with pytest.raises(ValueError, match="at least 2 training runs"):
        estimate_noise_floor({"only": np.zeros(100)})


def test_misaligned_runs_are_refused():
    """Silently intersecting differing lengths would correlate unrelated patients."""
    with pytest.raises(ValueError, match="differing lengths"):
        estimate_noise_floor({"a": np.zeros(100), "b": np.zeros(90)})


def test_non_finite_gaps_are_refused():
    gaps = _seeded_gaps(n=200)
    gaps["seed0"] = np.asarray(gaps["seed0"]).copy()
    gaps["seed0"][3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        estimate_noise_floor(gaps)


def test_summary_states_the_lower_bound_caveat():
    """The estimate understates the true noise share and must say so."""
    text = estimate_noise_floor(_seeded_gaps(), bootstrap_iterations=200).summary_text()
    assert "LOWER bound" in text
    assert "safe direction" in text
