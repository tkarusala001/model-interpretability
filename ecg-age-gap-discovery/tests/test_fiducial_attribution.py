"""Tests for fiducial-segment attribution - the project's methodological claim.

Two things need establishing, and they are different in kind.

**That the aggregation is not misleading.** ECG segments have very different
widths, so summed attribution is confounded by width alone.
``test_density_corrects_for_segment_width_but_share_does_not`` demonstrates the
confound on constructed input where the right answer is known exactly, and shows
that reporting density alongside share resolves it. If this project claims that
careful aggregation is a contribution, this is the test that substantiates it.

**That it finds a known injected effect.** Two cohorts are built whose age
signal lives entirely in the QRS complex and entirely in the T wave
respectively; a model is trained on each and attribution must shift accordingly.
This is the ground-truth check the build plan requires before the method is
trusted on real ECGs.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from ecg_discovery.config import (
    BackboneConfig,
    SignalProcessingConfig,
    SyntheticConfig,
    TrainingConfig,
)
from ecg_discovery.data.preprocessing import resample_signals
from ecg_discovery.data.synthetic_ecg import LEAD_NAMES, generate_cohort
from ecg_discovery.interpretability.attribution import integrated_gradients
from ecg_discovery.interpretability.fiducial_attribution import (
    PRECORDIAL_LEADS,
    SEGMENT_NAMES,
    aggregate_by_fiducial_segment,
    segment_masks,
    summarise_cohort,
)
from ecg_discovery.signal_processing.qrs_detection import detect_r_peaks
from ecg_discovery.signal_processing.wave_delineation import (
    BeatDelineation,
    WaveBoundaries,
    delineate_beats,
)
from ecg_discovery.training.train import train_age_regressor

SP = SignalProcessingConfig()


def _beat(offset: int = 0) -> BeatDelineation:
    """A hand-built beat with deliberately unequal segment widths.

    P spans 20 samples, QRS 30, T 60 - roughly the real proportions, and the
    reason a summed comparison across segments is confounded.
    """
    return BeatDelineation(
        r_peak=offset + 55,
        qrs=WaveBoundaries(onset=offset + 40, peak=offset + 55, offset=offset + 70),
        p_wave=WaveBoundaries(onset=offset + 10, peak=offset + 20, offset=offset + 30),
        t_wave=WaveBoundaries(onset=offset + 90, peak=offset + 120, offset=offset + 150),
        baseline_mv=0.0,
    )


# --------------------------------------------------------------------------- #
# Segment masks
# --------------------------------------------------------------------------- #
def test_masks_are_disjoint_and_cover_every_sample():
    masks = segment_masks(400, [_beat(0), _beat(180)])
    stacked = np.stack([masks[name] for name in SEGMENT_NAMES])
    np.testing.assert_array_equal(stacked.sum(axis=0), np.ones(400, dtype=int))


def test_masks_land_on_the_constructed_boundaries():
    masks = segment_masks(300, [_beat(0)])
    assert masks["P"][10] and masks["P"][30] and not masks["P"][9]
    assert masks["QRS"][40] and masks["QRS"][70] and not masks["QRS"][71]
    assert masks["T"][90] and masks["T"][150] and not masks["T"][151]
    assert masks["other"][0] and masks["other"][35] and masks["other"][80]


def test_overlapping_waves_resolve_with_qrs_taking_precedence():
    """Contested samples go to the QRS: shortest, sharpest, best delineated."""
    overlapping = BeatDelineation(
        r_peak=50,
        qrs=WaveBoundaries(onset=40, peak=50, offset=70),
        p_wave=WaveBoundaries(onset=20, peak=35, offset=45),   # overlaps QRS onset
        t_wave=WaveBoundaries(onset=65, peak=90, offset=120),  # overlaps QRS offset
        baseline_mv=0.0,
    )
    masks = segment_masks(200, [overlapping])
    assert masks["QRS"][40:71].all()
    assert not masks["P"][40:46].any()
    assert not masks["T"][65:71].any()
    stacked = np.stack([masks[name] for name in SEGMENT_NAMES])
    np.testing.assert_array_equal(stacked.sum(axis=0), np.ones(200, dtype=int))


def test_beats_without_p_or_t_contribute_only_a_qrs():
    beat = BeatDelineation(
        r_peak=50, qrs=WaveBoundaries(40, 50, 70), p_wave=None, t_wave=None,
        baseline_mv=0.0,
    )
    masks = segment_masks(200, [beat])
    assert masks["QRS"].sum() == 31
    assert masks["P"].sum() == 0 and masks["T"].sum() == 0


def test_no_beats_puts_everything_in_other():
    masks = segment_masks(100, [])
    assert masks["other"].all()


# --------------------------------------------------------------------------- #
# The width confound - the core methodological point
# --------------------------------------------------------------------------- #
def test_density_corrects_for_segment_width_but_share_does_not():
    """The reason both quantities are reported, shown on a known answer.

    Attribution is set to exactly the same magnitude at every sample, so by
    construction no segment is more important than any other. A summed
    comparison nonetheless ranks the segments purely by width - the T wave gets
    twice the P wave's share simply because it is twice as wide. Per-sample
    density recovers the truth that they are equally important.

    Reporting share alone would therefore make the T wave appear dominant in
    essentially every recording, and that finding would be arithmetic rather
    than physiology.
    """
    beats = [_beat(0), _beat(180)]
    attributions = np.ones((12, 400))          # uniform importance everywhere

    result = aggregate_by_fiducial_segment(attributions, beats, LEAD_NAMES)

    widths = dict(zip(SEGMENT_NAMES, result.segment_samples))
    assert widths["T"] > widths["QRS"] > widths["P"], widths

    # Share tracks width, exactly proportionally.
    share_ratio = result.segment_share("T") / result.segment_share("P")
    width_ratio = widths["T"] / widths["P"]
    assert share_ratio == pytest.approx(width_ratio, rel=1e-6)

    # Density does not: every segment is correctly reported as equally intense.
    for segment in ("P", "QRS", "T"):
        assert result.segment_density(segment) == pytest.approx(1.0)


def test_shares_sum_to_one_across_leads_and_segments():
    rng = np.random.default_rng(0)
    result = aggregate_by_fiducial_segment(
        rng.standard_normal((12, 400)), [_beat(0), _beat(180)], LEAD_NAMES
    )
    assert result.share.sum() == pytest.approx(1.0)


def test_dominant_segment_uses_density_not_share():
    """Ranking by share would just rank segments by width."""
    beats = [_beat(0)]
    attributions = np.zeros((12, 300))
    # Make the QRS intense but narrow, and the T wave weak but wide.
    attributions[:, 40:71] = 10.0
    attributions[:, 90:151] = 1.0

    result = aggregate_by_fiducial_segment(attributions, beats, LEAD_NAMES)
    assert result.dominant_segment(by="density") == "QRS"
    assert result.segment_share("QRS") > result.segment_share("T")
    # Density separates them far more decisively than share does.
    assert result.segment_density("QRS") == pytest.approx(10.0)
    assert result.segment_density("T") == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Direction and reporting
# --------------------------------------------------------------------------- #
def test_signed_share_preserves_direction():
    """Whether a structure made the heart look older or younger is the point."""
    beats = [_beat(0)]
    attributions = np.zeros((12, 300))
    attributions[:, 40:71] = 2.0       # QRS pushes the prediction up
    attributions[:, 90:151] = -1.0     # T wave pushes it down

    result = aggregate_by_fiducial_segment(attributions, beats, LEAD_NAMES)
    qrs = result.signed_share[:, SEGMENT_NAMES.index("QRS")].sum()
    t_wave = result.signed_share[:, SEGMENT_NAMES.index("T")].sum()
    assert qrs > 0 and t_wave < 0
    # Magnitudes are unaffected by sign.
    assert result.segment_share("T") > 0


def test_lead_subsets_can_be_queried():
    """Supports summaries of the form "concentrated in leads V1-V3"."""
    beats = [_beat(0)]
    attributions = np.zeros((12, 300))
    v1 = LEAD_NAMES.index("V1")
    attributions[v1, 40:71] = 5.0

    result = aggregate_by_fiducial_segment(attributions, beats, LEAD_NAMES)
    assert result.segment_share("QRS", leads=["V1"]) == pytest.approx(1.0)
    assert result.segment_share("QRS", leads=PRECORDIAL_LEADS) == pytest.approx(1.0)
    assert result.segment_share("QRS", leads=["II"]) == pytest.approx(0.0)


def test_unknown_segment_or_lead_raises():
    result = aggregate_by_fiducial_segment(np.ones((12, 300)), [_beat(0)], LEAD_NAMES)
    with pytest.raises(KeyError, match="unknown segment"):
        result.segment_share("ST")
    with pytest.raises(KeyError, match="unknown lead"):
        result.segment_share("QRS", leads=["V9"])


def test_summary_text_is_readable_and_mentions_each_wave():
    result = aggregate_by_fiducial_segment(
        np.random.default_rng(1).standard_normal((12, 400)),
        [_beat(0), _beat(180)], LEAD_NAMES,
        prediction=71.0, baseline_prediction=60.0,
    )
    text = result.summary_text()
    for segment in ("P", "QRS", "T"):
        assert segment in text
    assert "+11.0 years" in text
    assert "density" in text


def test_to_frame_has_one_row_per_lead_and_segment():
    result = aggregate_by_fiducial_segment(np.ones((12, 400)), [_beat(0)], LEAD_NAMES)
    frame = result.to_frame()
    assert len(frame) == 12 * len(SEGMENT_NAMES)
    assert set(frame["segment"]) == set(SEGMENT_NAMES)


def test_summarise_cohort_reports_spread_as_well_as_mean():
    rng = np.random.default_rng(2)
    results = [
        aggregate_by_fiducial_segment(
            rng.standard_normal((12, 400)), [_beat(0), _beat(180)], LEAD_NAMES
        )
        for _ in range(5)
    ]
    summary = summarise_cohort(results)
    assert len(summary) == 12 * len(SEGMENT_NAMES)
    assert {"share_mean", "share_sd", "density_mean"} <= set(summary.columns)
    with pytest.raises(ValueError, match="empty"):
        summarise_cohort([])


def test_bad_attribution_shapes_are_rejected():
    with pytest.raises(ValueError, match="n_leads, n_samples"):
        aggregate_by_fiducial_segment(np.ones(400), [_beat(0)], LEAD_NAMES)
    with pytest.raises(ValueError, match="lead_names has"):
        aggregate_by_fiducial_segment(np.ones((12, 400)), [_beat(0)], ["I", "II"])


def test_zero_attribution_does_not_divide_by_zero():
    result = aggregate_by_fiducial_segment(np.zeros((12, 400)), [_beat(0)], LEAD_NAMES)
    assert np.isfinite(result.share).all()
    assert result.share.sum() == 0.0


# --------------------------------------------------------------------------- #
# Ground truth: does attribution find a known injected effect?
# --------------------------------------------------------------------------- #
def _train_and_attribute(**overrides) -> dict[str, float]:
    """Train on a cohort with a specific injected effect and measure attribution."""
    config = dataclasses.replace(SyntheticConfig(sampling_rate_hz=500), **overrides)
    cohort = generate_cohort(config, n_recordings=400)
    signals = resample_signals(cohort.signals, 500, 100)

    result = train_age_regressor(
        signals=signals,
        ages=cohort.ages,
        sexes=cohort.sexes,
        patient_ids=cohort.patient_ids,
        backbone_config=BackboneConfig(
            stem_channels=16, stage_channels=(16, 24, 32), stride_per_stage=(2, 2, 2)
        ),
        training_config=TrainingConfig(
            epochs=25, batch_size=32, learning_rate=3e-3, warmup_epochs=1,
            early_stopping_patience=30, seed=0,
        ),
    )

    normalised = result.normalizer.transform(signals)
    indices = result.splits.test[:30]
    attribution = integrated_gradients(
        result.model,
        torch.tensor(normalised[indices]),
        torch.tensor(cohort.sexes[indices], dtype=torch.float32),
        n_steps=64,
    )

    densities = {segment: [] for segment in ("P", "QRS", "T")}
    shares = {segment: [] for segment in ("P", "QRS", "T")}
    for position, index in enumerate(indices):
        detection = detect_r_peaks(signals[index], 100.0, SP, LEAD_NAMES)
        beats = delineate_beats(signals[index], 100.0, detection.r_peaks, SP, LEAD_NAMES)
        summary = aggregate_by_fiducial_segment(
            attribution.attributions[position], beats, LEAD_NAMES
        )
        for segment in densities:
            densities[segment].append(summary.segment_density(segment))
            shares[segment].append(summary.segment_share(segment))

    output = {f"density_{k}": float(np.mean(v)) for k, v in densities.items()}
    output.update({f"share_{k}": float(np.mean(v)) for k, v in shares.items()})
    output["mae"] = result.test_metrics["mae"]
    return output


@pytest.fixture(scope="module")
def qrs_effect_cohort() -> dict[str, float]:
    """Age signal carried entirely by QRS widening."""
    return _train_and_attribute(
        qrs_widening_ms_per_decade=8.0,
        t_wave_skew_per_decade=0.0,
        hr_change_bpm_per_decade=0.0,
        unexplained_age_offset_sd_years=0.0,
    )


@pytest.fixture(scope="module")
def t_effect_cohort() -> dict[str, float]:
    """Age signal carried entirely by T-wave skew."""
    return _train_and_attribute(
        qrs_widening_ms_per_decade=0.0,
        t_wave_skew_per_decade=0.30,
        hr_change_bpm_per_decade=0.0,
        known_age_offset_sd_years=0.0,
    )


def test_attribution_localises_a_qrs_only_effect(qrs_effect_cohort):
    """When only the QRS carries age information, attention must go there."""
    result = qrs_effect_cohort
    assert result["mae"] < 12.0, "model did not learn; attribution test is meaningless"
    assert result["density_QRS"] > 5 * result["density_T"]
    assert result["density_QRS"] > 5 * result["density_P"]
    assert result["share_QRS"] > 0.5


def test_attribution_shifts_to_the_t_wave_when_the_effect_moves_there(
    qrs_effect_cohort, t_effect_cohort
):
    """The load-bearing ground-truth check, framed as a contrast.

    Attribution must move to the T wave when the injected age signal moves
    there. The comparison is made *between cohorts* rather than by demanding
    that the T wave dominate outright, because the QRS complex retains
    substantial attribution even when it carries no age information at all -
    it is the largest deflection in the signal and the model needs it to locate
    the beat before it can read anything else. Demanding absolute T-wave
    dominance would therefore be testing a claim that is not true, and not the
    one the method actually makes.

    What the method does claim is that attribution *tracks where the
    information lives*, and the ratio below moves by more than an order of
    magnitude between the two cohorts.
    """
    qrs_ratio = qrs_effect_cohort["density_T"] / qrs_effect_cohort["density_QRS"]
    t_ratio = t_effect_cohort["density_T"] / t_effect_cohort["density_QRS"]

    assert t_effect_cohort["mae"] < 12.0
    assert t_ratio > 10 * qrs_ratio, (
        f"T-to-QRS attribution ratio barely moved: {qrs_ratio:.4f} when the "
        f"effect is in the QRS, {t_ratio:.4f} when it is in the T wave"
    )
    # And by total share, the T wave does overtake the QRS.
    assert t_effect_cohort["share_T"] > t_effect_cohort["share_QRS"]
    assert t_effect_cohort["share_T"] > 3 * qrs_effect_cohort["share_T"]
