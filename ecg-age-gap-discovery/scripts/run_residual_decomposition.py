#!/usr/bin/env python3
"""Decompose a model's age-gap residual into rediscovery and discovery candidate.

Trains the age regressor, measures classical ECG intervals independently at
500 Hz, and regresses the age gap on those intervals to report how much of the
model's "extra" signal is a repackaging of measurements cardiology already
takes - and how much is left over.

The unexplained share is the *ceiling* on any discovery claim, not evidence for
one. Phase 9 asks whether it predicts anything independently verifiable.

Examples
--------
Synthetic, quick::

    python scripts/run_residual_decomposition.py --n-recordings 900 --epochs 30

Real data, once downloaded::

    python scripts/run_residual_decomposition.py --data ptbxl
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_discovery.config import (  # noqa: E402
    BackboneConfig,
    DataConfig,
    SignalProcessingConfig,
    SyntheticConfig,
    TrainingConfig,
    ValidationFrameworkConfig,
    load_config,
)
from ecg_discovery.data.preprocessing import resample_signals  # noqa: E402
from ecg_discovery.data.synthetic_ecg import LEAD_NAMES, generate_cohort  # noqa: E402
from ecg_discovery.runtime import RunContext  # noqa: E402
from ecg_discovery.signal_processing.interval_features import (  # noqa: E402
    interval_features_table,
)
from ecg_discovery.training.train import train_age_regressor  # noqa: E402
from ecg_discovery.validation.residual_decomposition import decompose_age_gap  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", choices=["synthetic", "ptbxl"], default="synthetic")
    parser.add_argument("--ptbxl-root", type=Path, default=Path("data/ptbxl"))
    parser.add_argument("--n-recordings", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    data_config = load_config(DataConfig, CONFIG_DIR / "data.yaml")
    backbone_config = load_config(BackboneConfig, CONFIG_DIR / "backbone.yaml")
    signal_config = load_config(SignalProcessingConfig, CONFIG_DIR / "signal_processing.yaml")
    validation_config = load_config(
        ValidationFrameworkConfig, CONFIG_DIR / "validation_framework.yaml"
    )
    training_config = dataclasses.replace(
        load_config(TrainingConfig, CONFIG_DIR / "training.yaml"),
        epochs=args.epochs, seed=args.seed, experiment_name="residual_decomposition",
    )

    # -- Data. Two views of the same recordings: the model's and the clinician's.
    if args.data == "ptbxl":
        from ecg_discovery.data.ptbxl_dataset import load_ptbxl, summarise_ptbxl

        print(summarise_ptbxl(args.ptbxl_root, data_config))
        model_view = load_ptbxl(args.ptbxl_root, data_config,
                                sampling_rate_hz=data_config.sampling_rate_hz,
                                progress=True)
        interval_view = load_ptbxl(args.ptbxl_root, data_config,
                                   sampling_rate_hz=data_config.interval_sampling_rate_hz,
                                   progress=True)
        signals_model = model_view.signals
        signals_intervals = interval_view.signals
        ages, sexes = model_view.ages, model_view.sexes.astype(np.float64)
        patient_ids, record_ids = model_view.patient_ids, model_view.record_ids
    else:
        synthetic = dataclasses.replace(
            load_config(SyntheticConfig, CONFIG_DIR / "synthetic.yaml"),
            n_recordings=args.n_recordings,
        )
        print(f"generating {synthetic.n_recordings} synthetic recordings at "
              f"{synthetic.sampling_rate_hz} Hz ...")
        cohort = generate_cohort(synthetic)
        signals_intervals = cohort.signals                     # 500 Hz
        signals_model = resample_signals(                      # 100 Hz
            cohort.signals, synthetic.sampling_rate_hz, data_config.sampling_rate_hz
        )
        ages, sexes = cohort.ages, cohort.sexes.astype(np.float64)
        patient_ids = cohort.patient_ids
        record_ids = [r.record_id for r in cohort]

    runs_dir = args.runs_dir or Path(training_config.runs_dir)
    with RunContext(
        training_config.experiment_name, runs_dir, seed=args.seed,
        configs={"data": data_config, "backbone": backbone_config,
                 "training": training_config, "signal_processing": signal_config,
                 "validation": validation_config},
        extra={"source": args.data},
    ) as run:
        print(f"run directory: {run.dir}\n")

        result = train_age_regressor(
            signals=signals_model, ages=ages, sexes=sexes, patient_ids=patient_ids,
            record_ids=record_ids, data_config=data_config,
            backbone_config=backbone_config, training_config=training_config,
            run=run, progress=True,
        )
        metrics = result.test_metrics
        print(f"\ntest MAE {metrics['mae']:.2f} years | R^2 {metrics['r2']:.3f}")

        test = result.predictions["test"]

        # -- Classical intervals, measured independently at the higher rate ----
        print(
            f"\nmeasuring classical intervals on {len(test.indices)} test recordings "
            f"at {data_config.interval_sampling_rate_hz} Hz ..."
        )
        features = interval_features_table(
            signals_intervals[test.indices],
            float(data_config.interval_sampling_rate_hz),
            signal_config, LEAD_NAMES,
            record_ids=[record_ids[i] for i in test.indices],
        )
        run.save_json("interval_features.json", features.to_dict(orient="records"))

        # -- The decomposition -------------------------------------------------
        decomposition = decompose_age_gap(
            test.age_gap, features, validation_config,
            ages=test.true_age, sexes=sexes[test.indices],
            patient_ids=patient_ids[test.indices],
        )

        print()
        print(decomposition.summary_text())

        best = decomposition.most_explanatory
        print("\nper-feature contribution above demographics (out-of-fold R^2):")
        for feature, score in sorted(
            best.univariate_r2.items(), key=lambda item: -item[1]
        ):
            print(f"    {feature:<22} {score:+.3f}")

        run.save_json("decomposition.json", {
            "known_features": list(decomposition.known_features),
            "covariates": list(decomposition.covariates),
            "n_recordings": decomposition.n_recordings,
            "n_dropped": decomposition.n_dropped,
            "age_gap_sd": decomposition.age_gap_sd,
            "explainers": {
                name: {
                    "r2_baseline": explainer.r2_baseline,
                    "r2_full": explainer.r2_full,
                    "r2_incremental": explainer.r2_incremental,
                    "unexplained_fraction": explainer.unexplained_fraction,
                    "r2_full_ci": list(explainer.r2_full_ci),
                    "feature_effects": explainer.feature_effects,
                    "univariate_r2": explainer.univariate_r2,
                }
                for name, explainer in decomposition.explainers.items()
            },
            "summary_text": decomposition.summary_text(),
        })
        # The unexplained residual is what Phase 9 tests for independent signal.
        run.save_json("unexplained_residual.json", {
            "record_id": [record_ids[i] for i in test.indices],
            "index": test.indices.tolist(),
            "age_gap": test.age_gap.tolist(),
            "unexplained_residual": best.unexplained_residual.tolist(),
            "explainer": best.model_name,
        })

        print(f"\nartifacts written to {run.artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
