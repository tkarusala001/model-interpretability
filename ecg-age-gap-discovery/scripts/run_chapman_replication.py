#!/usr/bin/env python3
"""External replication of the atrial finding on a second cohort.

Every result in this project so far comes from PTB-XL: one country, one set of
institutions, one generation of recording equipment. This script asks whether
the P-wave finding survives a genuinely independent cohort - the
Chapman-Shaoxing-Ningbo database, 42,997 usable adult recordings from three
Chinese hospitals.

TWO ARMS, ANSWERING DIFFERENT QUESTIONS
---------------------------------------
**Fresh** - train a new model on Chapman, then repeat the whole attribution and
occlusion analysis. Answers: *does a model trained on a different population, at
different institutions, also depend on the atria?* This is external replication
in the sense a reviewer means it, and it is the stronger claim.

**Transfer** - apply the PTB-XL-trained model to Chapman recordings unchanged.
Answers: *does this particular model behave consistently on new data?* Weaker on
its own, because a domain shift would look like a failed replication even if the
finding were real - but informative alongside the fresh arm.

Run together, the two arms separate "the finding is a property of ECGs" from
"the finding is a property of this model". If fresh replicates and transfer does
not, the effect is real but the model does not generalise; if neither
replicates, the PTB-XL result was cohort-specific.

WHAT IS NOT REPLICATED HERE
---------------------------
The diagnostic-link experiment. Chapman labels diagnoses with SNOMED-CT codes
that do not map cleanly onto PTB-XL's five superclasses, and inventing a mapping
would put a fabricated label set into a validation framework whose entire
purpose is to avoid fabricated comparisons. This cohort tests the decomposition
and the attribution finding only.
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
from ecg_discovery.data.chapman_dataset import (  # noqa: E402
    load_chapman, load_chapman_waveform_subset, summarise_chapman,
)
from ecg_discovery.data.preprocessing import LeadNormalizer, patient_level_split  # noqa: E402
from ecg_discovery.data.synthetic_ecg import LEAD_NAMES  # noqa: E402
from ecg_discovery.interpretability.attribution_controls import (  # noqa: E402
    compare_against_controls,
)
from ecg_discovery.interpretability.fiducial_attribution import SEGMENT_NAMES  # noqa: E402
from ecg_discovery.models.ecg_age_regressor import ECGAgeRegressor  # noqa: E402
from ecg_discovery.runtime import RunContext  # noqa: E402
from ecg_discovery.signal_processing.interval_features import (  # noqa: E402
    interval_features_table,
)
from ecg_discovery.signal_processing.qrs_detection import detect_r_peaks  # noqa: E402
from ecg_discovery.signal_processing.wave_delineation import delineate_beats  # noqa: E402
from ecg_discovery.training.train import _regression_metrics, train_age_regressor  # noqa: E402
from ecg_discovery.validation.residual_decomposition import decompose_age_gap  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs"
P_INDEX = SEGMENT_NAMES.index("P")
TESTED_SEGMENTS = ("P", "QRS", "T")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapman-root", type=Path, default=Path("data/chapman"))
    parser.add_argument("--ptbxl-checkpoint", type=Path, default=None,
                        help="model.pt from a PTB-XL run, for the transfer arm")
    parser.add_argument("--limit", type=int, default=24000,
                        help="Chapman recordings to load (memory-bounded)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n-attribution", type=int, default=400)
    parser.add_argument("--n-occlusion", type=int, default=1200)
    parser.add_argument("--ig-steps", type=int, default=128)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    return parser.parse_args(argv)


@torch.no_grad()
def _predict(model, signals, sexes, batch=64):
    """Predict on CPU - MPS has produced silently wrong numbers in this project."""
    model = model.to("cpu").eval()
    out = []
    for start in range(0, len(signals), batch):
        stop = min(start + batch, len(signals))
        out.append(model(
            torch.tensor(signals[start:stop], dtype=torch.float32, device="cpu"),
            torch.tensor(sexes[start:stop], dtype=torch.float32, device="cpu"),
        ).cpu().numpy())
    return np.concatenate(out)


def _delineate_all(signals, rows, signal_config):
    beats = []
    for row in rows:
        detection = detect_r_peaks(signals[row], 100.0, signal_config, LEAD_NAMES)
        beats.append(delineate_beats(
            signals[row], 100.0, detection.r_peaks, signal_config, LEAD_NAMES
        ))
    return beats


def _occlude_segments(normalised, rows, beats, model, sexes, true_age, seed):
    """Width-matched occlusion, identical protocol to run_occlusion_test.py."""
    from ecg_discovery.interpretability.fiducial_attribution import segment_masks

    rng = np.random.default_rng(seed)
    variants = {name: normalised[rows].copy() for name in
                list(TESTED_SEGMENTS) + [f"{s}_ctl" for s in TESTED_SEGMENTS]}
    counts = {name: 0 for name in TESTED_SEGMENTS}

    for position, row in enumerate(rows):
        masks = segment_masks(normalised.shape[2], beats[position])
        baseline = np.median(normalised[row], axis=1)
        for segment in TESTED_SEGMENTS:
            mask = masks[segment]
            counts[segment] += int(mask.sum())
            variants[segment][position][:, mask] = baseline[:, None]
            available = np.flatnonzero(masks["other"])
            if available.size:
                take = min(int(mask.sum()), available.size)
                control = rng.choice(available, size=take, replace=False)
                variants[f"{segment}_ctl"][position][:, control] = baseline[:, None]

    intact = _predict(model, normalised[rows], sexes[rows])
    results = {}
    for segment in TESTED_SEGMENTS:
        occluded = np.abs(_predict(model, variants[segment], sexes[rows]) - true_age)
        control = np.abs(_predict(model, variants[f"{segment}_ctl"], sexes[rows]) - true_age)
        paired = occluded - control
        stderr = paired.std(ddof=1) / np.sqrt(paired.size)
        results[segment] = {
            "delta_vs_control": float(paired.mean()),
            "ci_low": float(paired.mean() - 1.96 * stderr),
            "ci_high": float(paired.mean() + 1.96 * stderr),
            "per_100_samples": float(paired.mean() / (counts[segment] / len(rows)) * 100),
        }
    results["_mae_intact"] = float(np.abs(intact - true_age).mean())
    return results


def main(argv=None) -> int:
    args = parse_args(argv)
    data_config = load_config(DataConfig, CONFIG_DIR / "data.yaml")
    backbone_config = load_config(BackboneConfig, CONFIG_DIR / "backbone.yaml")
    signal_config = load_config(SignalProcessingConfig, CONFIG_DIR / "signal_processing.yaml")
    validation_config = load_config(
        ValidationFrameworkConfig, CONFIG_DIR / "validation_framework.yaml"
    )

    print(summarise_chapman(args.chapman_root, data_config))
    print(f"\nloading {args.limit} recordings at {data_config.sampling_rate_hz} Hz "
          "(resampled during load) ...")
    data = load_chapman(args.chapman_root, data_config, limit=args.limit, progress=True)
    sexes = data.sexes.astype(np.float64)
    print(f"loaded {len(data)} recordings\n")

    splits = patient_level_split(data.patient_ids, data_config)

    with RunContext("chapman_replication", args.runs_dir, seed=args.seeds[0],
                    configs={"data": data_config, "backbone": backbone_config,
                             "validation": validation_config},
                    extra={"cohort": "chapman", "n_loaded": len(data)}) as run:
        print(f"run directory: {run.dir}\n")
        summary_rows = []

        # ---------------- ARM A: fresh models trained on Chapman -------------
        print("=" * 78)
        print("ARM A - FRESH: models trained on Chapman")
        print("=" * 78)
        beats_cache = None
        attribution_rows = None
        first = None

        for seed in args.seeds:
            training_config = dataclasses.replace(
                load_config(TrainingConfig, CONFIG_DIR / "training.yaml"),
                epochs=args.epochs, seed=seed, experiment_name="chapman_replication",
            )
            result = train_age_regressor(
                signals=data.signals, ages=data.ages, sexes=sexes,
                patient_ids=data.patient_ids, record_ids=data.record_ids,
                data_config=data_config, backbone_config=backbone_config,
                training_config=training_config, splits=splits,
            )
            test = result.predictions["test"]
            normalised = result.normalizer.transform(data.signals)

            if attribution_rows is None:
                attribution_rows = test.indices[: args.n_attribution]
                beats_cache = _delineate_all(data.signals, attribution_rows, signal_config)
                first = result

            comparison = compare_against_controls(
                result.model, normalised[attribution_rows],
                data.signals[attribution_rows],
                sexes[attribution_rows].astype(np.float32),
                beats_cache, LEAD_NAMES,
                untrained_model=ECGAgeRegressor(backbone_config).eval(),
                n_steps=args.ig_steps,
            )
            p_row = comparison.difference("amplitude")
            p_row = p_row[p_row["segment"] == "P"].iloc[0]

            occlusion_rows = test.indices[: args.n_occlusion]
            occ_beats = _delineate_all(data.signals, occlusion_rows, signal_config)
            occlusion = _occlude_segments(
                normalised, occlusion_rows, occ_beats, result.model, sexes,
                data.ages[occlusion_rows], seed,
            )

            summary_rows.append({
                "arm": "fresh", "seed": seed,
                "test_mae": result.test_metrics["mae"],
                "p_attr_delta": float(p_row["delta"]),
                "p_attr_ci_low": float(p_row["ci_low"]),
                "p_occ_delta": occlusion["P"]["delta_vs_control"],
                "p_occ_ci_low": occlusion["P"]["ci_low"],
                "qrs_occ_delta": occlusion["QRS"]["delta_vs_control"],
                "t_occ_delta": occlusion["T"]["delta_vs_control"],
                "amplitude_r": comparison.amplitude_correlation(),
            })
            print(f"  seed {seed}: MAE {result.test_metrics['mae']:.2f} | "
                  f"P attribution {p_row['delta']:+.4f} "
                  f"[{p_row['ci_low']:+.4f}, {p_row['ci_high']:+.4f}] | "
                  f"P occlusion {occlusion['P']['delta_vs_control']:+.3f} "
                  f"[{occlusion['P']['ci_low']:+.3f}, {occlusion['P']['ci_high']:+.3f}]")

        # ---------------- ARM B: transfer the PTB-XL model -------------------
        if args.ptbxl_checkpoint and args.ptbxl_checkpoint.is_file():
            print()
            print("=" * 78)
            print("ARM B - TRANSFER: PTB-XL model applied to Chapman unchanged")
            print("=" * 78)
            payload = torch.load(args.ptbxl_checkpoint, weights_only=False)
            transfer = ECGAgeRegressor(BackboneConfig(**payload["backbone_config"]))
            transfer.load_state_dict(payload["state_dict"])
            transfer.eval()
            # The PTB-XL normaliser travels with the model: applying a different
            # transformation than the model trained under would change every
            # prediction for reasons unrelated to the cohort.
            ptbxl_normalizer = LeadNormalizer.from_dict(payload["normalizer"])
            normalised = ptbxl_normalizer.transform(data.signals)

            predicted = _predict(transfer, normalised[attribution_rows],
                                 sexes[attribution_rows])
            metrics = _regression_metrics(data.ages[attribution_rows], predicted)
            print(f"  transferred MAE on Chapman: {metrics['mae']:.2f} years "
                  f"(R^2 {metrics['r2']:.3f})")

            comparison = compare_against_controls(
                transfer, normalised[attribution_rows], data.signals[attribution_rows],
                sexes[attribution_rows].astype(np.float32), beats_cache, LEAD_NAMES,
                untrained_model=ECGAgeRegressor(backbone_config).eval(),
                n_steps=args.ig_steps,
            )
            p_row = comparison.difference("amplitude")
            p_row = p_row[p_row["segment"] == "P"].iloc[0]

            occlusion_rows = attribution_rows[: args.n_occlusion]
            occ_beats = beats_cache[: len(occlusion_rows)]
            occlusion = _occlude_segments(
                normalised, occlusion_rows, occ_beats, transfer, sexes,
                data.ages[occlusion_rows], 0,
            )
            summary_rows.append({
                "arm": "transfer", "seed": -1, "test_mae": metrics["mae"],
                "p_attr_delta": float(p_row["delta"]),
                "p_attr_ci_low": float(p_row["ci_low"]),
                "p_occ_delta": occlusion["P"]["delta_vs_control"],
                "p_occ_ci_low": occlusion["P"]["ci_low"],
                "qrs_occ_delta": occlusion["QRS"]["delta_vs_control"],
                "t_occ_delta": occlusion["T"]["delta_vs_control"],
                "amplitude_r": comparison.amplitude_correlation(),
            })
            print(f"  P attribution {p_row['delta']:+.4f} "
                  f"[{p_row['ci_low']:+.4f}, {p_row['ci_high']:+.4f}] | "
                  f"P occlusion {occlusion['P']['delta_vs_control']:+.3f}")

        # ---------------- decomposition on Chapman ---------------------------
        print()
        print("=" * 78)
        print("DECOMPOSITION on Chapman (500 Hz interval measurement)")
        print("=" * 78)
        test = first.predictions["test"]
        rows = test.indices[: args.n_occlusion]
        signals_500 = load_chapman_waveform_subset(data.metadata, rows)
        features = interval_features_table(
            signals_500, float(data_config.interval_sampling_rate_hz),
            signal_config, LEAD_NAMES,
        )
        position = {int(r): i for i, r in enumerate(test.indices)}
        gap = np.array([test.age_gap[position[int(r)]] for r in rows])
        true_age = np.array([test.true_age[position[int(r)]] for r in rows])
        decomposition = decompose_age_gap(
            gap, features, validation_config, ages=true_age, sexes=sexes[rows]
        )
        print(decomposition.summary_text())

        # ---------------- verdict --------------------------------------------
        frame = pd.DataFrame(summary_rows)
        print()
        print("=" * 78)
        print("REPLICATION VERDICT")
        print("=" * 78)
        print(frame.to_string(index=False))
        fresh = frame[frame.arm == "fresh"]
        attr_ok = int((fresh.p_attr_ci_low > 0).sum())
        occ_ok = int((fresh.p_occ_ci_low > 0).sum())
        print()
        print(f"  P-wave attribution above amplitude null: {attr_ok}/{len(fresh)} "
              f"fresh seeds, mean {fresh.p_attr_delta.mean():+.4f}")
        print(f"  P-wave causal occlusion:                 {occ_ok}/{len(fresh)} "
              f"fresh seeds, mean {fresh.p_occ_delta.mean():+.3f} years")
        replicates = attr_ok == len(fresh) and occ_ok == len(fresh)
        print()
        print("  VERDICT: the atrial finding "
              + ("REPLICATES on an independent cohort."
                 if replicates else "DOES NOT replicate on an independent cohort."))

        run.save_json("chapman_replication.json", {
            "summary": frame.to_dict(orient="records"),
            "replicates": replicates,
            "decomposition": decomposition.summary_text(),
        })
        print(f"\nartifacts: {run.artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
