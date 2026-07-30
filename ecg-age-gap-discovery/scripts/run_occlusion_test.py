#!/usr/bin/env python3
"""Causal test: which cardiac segment does age prediction actually need?

Attribution says where a model looks. It does not say the model *needs* what it
is looking at - a gradient can be large on a feature the prediction would
survive losing. This script asks the causal question directly: replace one
cardiac segment with the isoelectric baseline, and measure how much age
prediction degrades.

TWO CONFOUNDS THIS CONTROLS FOR
-------------------------------
**Segment width.** A T wave spans more samples than a P wave, so occluding it
removes more signal. Degradation is therefore reported both in absolute terms
and *per 100 samples occluded*, and the second is the comparable figure.

**Removing signal at all.** Blanking any part of an ECG perturbs the input, and
a network may degrade simply because its input became unfamiliar. Each segment
is therefore paired against an **isoelectric control**: a randomly placed window
in the electrically silent stretches, occluded to exactly the same total number
of samples. The difference between segment occlusion and its width-matched
isoelectric control is the quantity that carries meaning.

WHAT WOULD SUPPORT THE P-WAVE FINDING
-------------------------------------
Phase 7 found the model places more attribution on the P wave than signal
amplitude, an untrained model, or a shuffled-label model would predict,
replicated across three independently trained models and concentrated in the
limb leads and V1. If that reflects genuine reliance, occluding the P wave
should degrade age prediction by more than occluding an equal number of
isoelectric samples. If P-wave occlusion is indistinguishable from its control,
the attribution was showing where the model looked without showing what it
needed - which is worth reporting just as plainly.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_discovery.config import (  # noqa: E402
    BackboneConfig, DataConfig, SignalProcessingConfig, TrainingConfig, load_config,
)
from ecg_discovery.data.ptbxl_dataset import load_ptbxl, official_split  # noqa: E402
from ecg_discovery.data.synthetic_ecg import LEAD_NAMES  # noqa: E402
from ecg_discovery.interpretability.fiducial_attribution import (  # noqa: E402
    SEGMENT_NAMES, segment_masks,
)
from ecg_discovery.runtime import RunContext, set_global_seed  # noqa: E402
from ecg_discovery.signal_processing.qrs_detection import detect_r_peaks  # noqa: E402
from ecg_discovery.signal_processing.wave_delineation import delineate_beats  # noqa: E402
from ecg_discovery.training.train import train_age_regressor  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs"
TESTED_SEGMENTS = ("P", "QRS", "T")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptbxl-root", type=Path, default=Path("data/ptbxl"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n-recordings", type=int, default=1200,
                        help="test recordings to occlude")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    return parser.parse_args(argv)


def _occlude(
    signal: np.ndarray, mask: np.ndarray, baseline: np.ndarray
) -> np.ndarray:
    """Replace masked samples with each lead's isoelectric level."""
    out = signal.copy()
    out[:, mask] = baseline[:, None]
    return out


def _isoelectric_control_mask(
    other_mask: np.ndarray, n_samples_to_occlude: int, rng: np.random.Generator
) -> np.ndarray:
    """A width-matched occlusion inside the electrically silent stretches.

    Controls for the fact that blanking *any* stretch of an ECG perturbs the
    input. Sampled from the isoelectric region so it removes no cardiac wave.
    """
    available = np.flatnonzero(other_mask)
    control = np.zeros_like(other_mask)
    if available.size == 0:
        return control
    take = min(n_samples_to_occlude, available.size)
    control[rng.choice(available, size=take, replace=False)] = True
    return control


@torch.no_grad()
def _predict(model, signals: np.ndarray, sexes: np.ndarray, batch: int = 64) -> np.ndarray:
    """Predict ages, on CPU.

    Training may have run on Apple MPS, and passing CPU tensors to an MPS model
    fails with an opaque error from inside a convolution. The model is moved to
    CPU here for the same reason
    :mod:`ecg_discovery.interpretability.attribution` avoids MPS: this project
    has observed that backend returning silently wrong numbers. These are
    forward passes only and the model is small, so CPU costs little.
    """
    model = model.to("cpu").eval()
    out = []
    for start in range(0, len(signals), batch):
        stop = min(start + batch, len(signals))
        out.append(model(
            torch.tensor(signals[start:stop], dtype=torch.float32, device="cpu"),
            torch.tensor(sexes[start:stop], dtype=torch.float32, device="cpu"),
        ).cpu().numpy())
    return np.concatenate(out)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_config = load_config(DataConfig, CONFIG_DIR / "data.yaml")
    backbone_config = load_config(BackboneConfig, CONFIG_DIR / "backbone.yaml")
    signal_config = load_config(SignalProcessingConfig, CONFIG_DIR / "signal_processing.yaml")

    print("loading PTB-XL ...")
    data = load_ptbxl(args.ptbxl_root, data_config,
                      sampling_rate_hz=data_config.sampling_rate_hz)
    splits = official_split(data.strat_folds, data.patient_ids)
    sexes = data.sexes.astype(np.float64)

    with RunContext("occlusion_test", args.runs_dir, seed=args.seeds[0],
                    configs={"data": data_config, "backbone": backbone_config}) as run:
        print(f"run directory: {run.dir}\n")

        rows: list[dict] = []
        for seed in args.seeds:
            training_config = dataclasses.replace(
                load_config(TrainingConfig, CONFIG_DIR / "training.yaml"),
                epochs=args.epochs, seed=seed, experiment_name="occlusion_test",
            )
            result = train_age_regressor(
                signals=data.signals, ages=data.ages, sexes=sexes,
                patient_ids=data.patient_ids, record_ids=data.record_ids,
                data_config=data_config, backbone_config=backbone_config,
                training_config=training_config, splits=splits,
            )
            test = result.predictions["test"]
            subset = test.indices[: args.n_recordings]
            normalised = result.normalizer.transform(data.signals)
            true_age = data.ages[subset]
            subset_sexes = sexes[subset]

            print(f"seed {seed}: baseline test MAE {result.test_metrics['mae']:.3f}; "
                  f"occluding {len(subset)} recordings ...")

            # Build occluded copies of the whole subset, one variant per segment
            # plus one width-matched isoelectric control per segment.
            variants: dict[str, np.ndarray] = {
                name: normalised[subset].copy()
                for name in list(TESTED_SEGMENTS) + [f"{s}_control" for s in TESTED_SEGMENTS]
            }
            occluded_samples = {name: 0 for name in TESTED_SEGMENTS}
            rng = np.random.default_rng(seed)

            for position, row in enumerate(subset):
                detection = detect_r_peaks(
                    data.signals[row], 100.0, signal_config, LEAD_NAMES
                )
                beats = delineate_beats(
                    data.signals[row], 100.0, detection.r_peaks,
                    signal_config, LEAD_NAMES,
                )
                masks = segment_masks(normalised.shape[2], beats)
                # Isoelectric level per lead, matching the attribution baseline.
                baseline = np.median(normalised[row], axis=1)

                for segment in TESTED_SEGMENTS:
                    mask = masks[segment]
                    occluded_samples[segment] += int(mask.sum())
                    variants[segment][position] = _occlude(
                        normalised[row], mask, baseline
                    )
                    control = _isoelectric_control_mask(
                        masks["other"], int(mask.sum()), rng
                    )
                    variants[f"{segment}_control"][position] = _occlude(
                        normalised[row], control, baseline
                    )

            intact = _predict(result.model, normalised[subset], subset_sexes)
            mae_intact = float(np.mean(np.abs(intact - true_age)))

            for segment in TESTED_SEGMENTS:
                predicted = _predict(result.model, variants[segment], subset_sexes)
                predicted_control = _predict(
                    result.model, variants[f"{segment}_control"], subset_sexes
                )
                error_segment = np.abs(predicted - true_age)
                error_control = np.abs(predicted_control - true_age)
                error_intact = np.abs(intact - true_age)

                mean_samples = occluded_samples[segment] / len(subset)
                # Paired difference: how much worse than the width-matched
                # isoelectric control, per recording.
                paired = error_segment - error_control
                stderr = paired.std(ddof=1) / np.sqrt(paired.size)
                rows.append({
                    "seed": seed,
                    "segment": segment,
                    "mae_intact": mae_intact,
                    "mae_occluded": float(error_segment.mean()),
                    "mae_control": float(error_control.mean()),
                    "delta_vs_intact": float(error_segment.mean() - error_intact.mean()),
                    "delta_vs_control": float(paired.mean()),
                    "ci_low": float(paired.mean() - 1.96 * stderr),
                    "ci_high": float(paired.mean() + 1.96 * stderr),
                    "samples_occluded": mean_samples,
                    "delta_per_100_samples": float(
                        paired.mean() / mean_samples * 100 if mean_samples else np.nan
                    ),
                })
                print(f"    {segment:<4} MAE {error_segment.mean():6.3f} vs control "
                      f"{error_control.mean():6.3f} | delta {paired.mean():+6.3f} "
                      f"[{rows[-1]['ci_low']:+.3f}, {rows[-1]['ci_high']:+.3f}] | "
                      f"{mean_samples:5.0f} samples | per-100 "
                      f"{rows[-1]['delta_per_100_samples']:+.3f}")

        frame = pd.DataFrame(rows)
        print()
        print("=" * 78)
        print("SUMMARY: degradation beyond a width-matched isoelectric occlusion")
        print("=" * 78)
        summary = frame.groupby("segment", sort=False).agg(
            mean_delta=("delta_vs_control", "mean"),
            min_delta=("delta_vs_control", "min"),
            max_delta=("delta_vs_control", "max"),
            mean_per_100=("delta_per_100_samples", "mean"),
            samples=("samples_occluded", "mean"),
            n_seeds_positive=("ci_low", lambda s: int((s > 0).sum())),
        ).reset_index()
        print(summary.to_string(index=False))

        print()
        for _, entry in summary.iterrows():
            verdict = (
                f"replicates in {entry.n_seeds_positive}/{len(args.seeds)} seeds"
                if entry.n_seeds_positive else "no reliable effect"
            )
            print(f"  {entry.segment:<4} {entry.mean_delta:+.3f} years "
                  f"({entry.mean_per_100:+.3f} per 100 samples) - {verdict}")

        run.save_json("occlusion.json", {
            "per_seed": frame.to_dict(orient="records"),
            "summary": summary.to_dict(orient="records"),
        })
        print(f"\nartifacts: {run.artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
