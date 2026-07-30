#!/usr/bin/env python3
"""Interrogate the elevated-P-wave attribution finding until it breaks or holds.

Phase 7 found that the trained model places more attribution on the P wave than
signal amplitude, an untrained model, or a shuffled-label model would predict.
That is a candidate finding. This script tries to break it, and - if it
survives - tries to make it specific.

Four probes, in order of how badly they could kill it:

1. **Replication across seeds.** The single most important test. A previous
   apparently-significant result in this project (the unexplained residual
   improving MI detection) evaporated when the data split changed. Any finding
   that has not been re-derived from an independently trained model is not yet a
   finding.

2. **Integration convergence.** The Phase 7 run used 32 Integrated Gradients
   steps and the completeness check failed on roughly 1 recording in 32 - those
   with very large output excursions. Rerunning at higher resolution establishes
   whether the effect is an artefact of under-integration.

3. **Where in the P wave, and in which leads.** A finding localised to the leads
   where the P wave is actually diagnostic (II, V1) is far more credible than
   one smeared uniformly, which would suggest an artefact.

4. **Is it beyond classical P-wave measurement?** The sharpest question, and the
   one that connects Phase 7 to Phase 8. If the model attends to the P wave and
   its age gap is *not* explained by P duration and P amplitude, then it is
   reading something in the atrial complex that classical P-wave measurement
   does not capture. If the age gap *is* explained by them, the attention is
   rediscovery.
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
    BackboneConfig, DataConfig, SignalProcessingConfig, TrainingConfig,
    ValidationFrameworkConfig, load_config,
)
from ecg_discovery.data.ptbxl_dataset import (  # noqa: E402
    load_ptbxl, load_waveform_subset, official_split,
)
from ecg_discovery.data.synthetic_ecg import LEAD_NAMES  # noqa: E402
from ecg_discovery.interpretability.attribution import integrated_gradients  # noqa: E402
from ecg_discovery.interpretability.attribution_controls import (  # noqa: E402
    amplitude_profile, compare_against_controls,
)
from ecg_discovery.interpretability.fiducial_attribution import (  # noqa: E402
    SEGMENT_NAMES, aggregate_by_fiducial_segment,
)
from ecg_discovery.models.ecg_age_regressor import ECGAgeRegressor  # noqa: E402
from ecg_discovery.runtime import RunContext  # noqa: E402
from ecg_discovery.signal_processing.interval_features import (  # noqa: E402
    interval_features_table,
)
from ecg_discovery.signal_processing.qrs_detection import detect_r_peaks  # noqa: E402
from ecg_discovery.signal_processing.wave_delineation import delineate_beats  # noqa: E402
from ecg_discovery.training.train import train_age_regressor  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs"
P_INDEX = SEGMENT_NAMES.index("P")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptbxl-root", type=Path, default=Path("data/ptbxl"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n-attribution", type=int, default=400)
    parser.add_argument("--ig-steps", type=int, default=128)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_config = load_config(DataConfig, CONFIG_DIR / "data.yaml")
    backbone_config = load_config(BackboneConfig, CONFIG_DIR / "backbone.yaml")
    signal_config = load_config(SignalProcessingConfig, CONFIG_DIR / "signal_processing.yaml")
    validation_config = load_config(
        ValidationFrameworkConfig, CONFIG_DIR / "validation_framework.yaml"
    )

    print("loading PTB-XL ...")
    data = load_ptbxl(args.ptbxl_root, data_config,
                      sampling_rate_hz=data_config.sampling_rate_hz)
    splits = official_split(data.strat_folds, data.patient_ids)
    sexes = data.sexes.astype(np.float64)

    with RunContext("p_wave_probe", args.runs_dir, seed=args.seeds[0],
                    configs={"data": data_config, "backbone": backbone_config,
                             "validation": validation_config}) as run:
        print(f"run directory: {run.dir}\n")

        # ------------------------------------------------------------------ #
        # Probe 1 + 2: replication across seeds, at higher integration resolution
        # ------------------------------------------------------------------ #
        print("=" * 72)
        print("PROBE 1+2: does the P-wave elevation replicate across independently")
        print(f"           trained models, at {args.ig_steps} integration steps?")
        print("=" * 72)

        per_seed = []
        beats_cache = None
        subset = None

        for seed in args.seeds:
            training_config = dataclasses.replace(
                load_config(TrainingConfig, CONFIG_DIR / "training.yaml"),
                epochs=args.epochs, seed=seed, experiment_name="p_wave_probe",
            )
            result = train_age_regressor(
                signals=data.signals, ages=data.ages, sexes=sexes,
                patient_ids=data.patient_ids, record_ids=data.record_ids,
                data_config=data_config, backbone_config=backbone_config,
                training_config=training_config, splits=splits,
            )
            test = result.predictions["test"]
            normalised = result.normalizer.transform(data.signals)

            if subset is None:
                subset = test.indices[: args.n_attribution]
                beats_cache = []
                for row in subset:
                    detection = detect_r_peaks(
                        data.signals[row], 100.0, signal_config, LEAD_NAMES
                    )
                    beats_cache.append(delineate_beats(
                        data.signals[row], 100.0, detection.r_peaks,
                        signal_config, LEAD_NAMES,
                    ))

            comparison = compare_against_controls(
                result.model, normalised[subset], data.signals[subset],
                sexes[subset].astype(np.float32), beats_cache, LEAD_NAMES,
                untrained_model=ECGAgeRegressor(backbone_config).eval(),
                n_steps=args.ig_steps,
            )
            delta_vs_amplitude = comparison.difference("amplitude")
            p_row = delta_vs_amplitude[delta_vs_amplitude["segment"] == "P"].iloc[0]
            per_seed.append({
                "seed": seed,
                "test_mae": result.test_metrics["mae"],
                "p_share_trained": float(comparison.trained.mean_share[P_INDEX]),
                "p_share_amplitude": float(
                    comparison.controls["amplitude"].mean_share[P_INDEX]
                ),
                "p_delta": float(p_row["delta"]),
                "ci_low": float(p_row["ci_low"]),
                "ci_high": float(p_row["ci_high"]),
                "amplitude_r": comparison.amplitude_correlation(),
            })
            print(f"  seed {seed}: MAE {result.test_metrics['mae']:.2f} | "
                  f"P share {per_seed[-1]['p_share_trained']:.3f} vs amplitude "
                  f"{per_seed[-1]['p_share_amplitude']:.3f} | "
                  f"delta {p_row['delta']:+.4f} "
                  f"[{p_row['ci_low']:+.4f}, {p_row['ci_high']:+.4f}]")

            if seed == args.seeds[0]:
                first_model, first_result = result.model, result

        frame = pd.DataFrame(per_seed)
        replicates = bool((frame["ci_low"] > 0).all())
        print()
        print(f"REPLICATION: P-wave elevation positive with CI excluding zero in "
              f"{int((frame['ci_low'] > 0).sum())}/{len(frame)} independently "
              f"trained models -> {'HOLDS' if replicates else 'DOES NOT HOLD'}")
        print(f"  delta across seeds: mean {frame['p_delta'].mean():+.4f}, "
              f"range [{frame['p_delta'].min():+.4f}, {frame['p_delta'].max():+.4f}]")
        run.save_json("replication.json", frame.to_dict(orient="records"))

        if not replicates:
            print("\nThe finding does not replicate. Stopping - the remaining probes "
                  "would be characterising noise.")
            return 0

        # ------------------------------------------------------------------ #
        # Probe 3: which leads carry it?
        # ------------------------------------------------------------------ #
        print()
        print("=" * 72)
        print("PROBE 3: which leads carry the P-wave elevation?")
        print("=" * 72)
        normalised = first_result.normalizer.transform(data.signals)
        per_lead_trained = np.zeros((len(subset), 12))
        per_lead_amplitude = np.zeros((len(subset), 12))

        for start in range(0, len(subset), 32):
            rows = subset[start : start + 32]
            attribution = integrated_gradients(
                first_model, torch.tensor(normalised[rows]),
                torch.tensor(sexes[rows], dtype=torch.float32),
                n_steps=args.ig_steps, max_relative_error=None,
            )
            for offset, row in enumerate(rows):
                beats = beats_cache[start + offset]
                trained = aggregate_by_fiducial_segment(
                    attribution.attributions[offset], beats, LEAD_NAMES
                )
                magnitude = aggregate_by_fiducial_segment(
                    np.abs(data.signals[row].astype(np.float64)), beats, LEAD_NAMES
                )
                per_lead_trained[start + offset] = trained.share[:, P_INDEX]
                per_lead_amplitude[start + offset] = magnitude.share[:, P_INDEX]

        lead_frame = pd.DataFrame({
            "lead": LEAD_NAMES,
            "p_share_trained": per_lead_trained.mean(axis=0),
            "p_share_amplitude": per_lead_amplitude.mean(axis=0),
        })
        lead_frame["delta"] = lead_frame.p_share_trained - lead_frame.p_share_amplitude
        stderr = (per_lead_trained - per_lead_amplitude).std(axis=0, ddof=1) / np.sqrt(len(subset))
        lead_frame["ci_low"] = lead_frame.delta - 1.96 * stderr
        lead_frame["ci_high"] = lead_frame.delta + 1.96 * stderr
        print(lead_frame.sort_values("delta", ascending=False).to_string(index=False))
        run.save_json("per_lead.json", lead_frame.to_dict(orient="records"))

        # ------------------------------------------------------------------ #
        # Probe 4: is the P-wave attention beyond classical P measurement?
        # ------------------------------------------------------------------ #
        print()
        print("=" * 72)
        print("PROBE 4: does classical P-wave measurement explain the age gap?")
        print("=" * 72)
        print("measuring intervals at 500 Hz on the attributed subset ...")
        signals_500 = load_waveform_subset(
            args.ptbxl_root, data.metadata, subset,
            data_config.interval_sampling_rate_hz,
        )
        features = interval_features_table(
            signals_500, float(data_config.interval_sampling_rate_hz),
            signal_config, LEAD_NAMES,
        )
        test = first_result.predictions["test"]
        position = {int(row): i for i, row in enumerate(test.indices)}
        gap = np.array([test.age_gap[position[int(row)]] for row in subset])
        true_age = np.array([test.true_age[position[int(row)]] for row in subset])

        from ecg_discovery.validation.residual_decomposition import decompose_age_gap

        p_only = dataclasses.replace(
            validation_config,
            known_features=("p_duration_ms", "p_amplitude_mv", "pr_interval_ms"),
            explainer_models=("gradient_boosting",), cv_folds=5,
        )
        p_decomposition = decompose_age_gap(
            gap, features, p_only, ages=true_age, sexes=sexes[subset]
        )
        p_explainer = p_decomposition.most_explanatory
        print(f"  classical P-wave features (P duration, P amplitude, PR) explain "
              f"{p_explainer.r2_incremental:.1%} of the age gap beyond demographics")
        for feature, score in sorted(p_explainer.univariate_r2.items(),
                                     key=lambda item: -item[1]):
            print(f"    {feature:<20} {score:+.4f}")

        # Does per-recording P attention track the measured P features at all?
        p_attention = per_lead_trained.sum(axis=1)
        print()
        print("  correlation of this recording's P-wave ATTENTION with its measured "
              "P-wave properties:")
        for column in ("p_duration_ms", "p_amplitude_mv", "pr_interval_ms"):
            values = features[column].to_numpy()
            ok = np.isfinite(values) & np.isfinite(p_attention)
            r = float(np.corrcoef(p_attention[ok], values[ok])[0, 1])
            print(f"    {column:<20} r = {r:+.3f}")

        run.save_json("p_wave_decomposition.json", {
            "r2_incremental": p_explainer.r2_incremental,
            "univariate_r2": p_explainer.univariate_r2,
            "summary": p_decomposition.summary_text(),
        })
        print(f"\nartifacts: {run.artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
