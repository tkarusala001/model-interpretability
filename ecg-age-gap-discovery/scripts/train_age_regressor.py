#!/usr/bin/env python3
"""Train the from-scratch ECG age regressor.

Runs against the synthetic cohort by default, so the whole pipeline is
exercisable without PTB-XL present. Pass ``--data ptbxl`` once PTB-XL has been
downloaded per ``scripts/download_data.sh``.

Examples
--------
Train on synthetic data, small and fast::

    python scripts/train_age_regressor.py --n-recordings 800 --epochs 30

Full configuration from YAML::

    python scripts/train_age_regressor.py \\
        --data-config configs/data.yaml \\
        --backbone-config configs/backbone.yaml \\
        --training-config configs/training.yaml
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
    SyntheticConfig,
    TrainingConfig,
    load_config,
)
from ecg_discovery.data.preprocessing import resample_signals  # noqa: E402
from ecg_discovery.data.synthetic_ecg import generate_cohort  # noqa: E402
from ecg_discovery.runtime import RunContext  # noqa: E402
from ecg_discovery.training.train import train_age_regressor  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=["synthetic", "ptbxl"], default="synthetic")
    parser.add_argument("--data-config", type=Path, default=CONFIG_DIR / "data.yaml")
    parser.add_argument("--backbone-config", type=Path, default=CONFIG_DIR / "backbone.yaml")
    parser.add_argument("--training-config", type=Path, default=CONFIG_DIR / "training.yaml")
    parser.add_argument("--synthetic-config", type=Path, default=CONFIG_DIR / "synthetic.yaml")
    parser.add_argument("--n-recordings", type=int, default=None, help="override cohort size")
    parser.add_argument("--epochs", type=int, default=None, help="override training epochs")
    parser.add_argument("--seed", type=int, default=None, help="override training seed")
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    data_config = load_config(DataConfig, args.data_config)
    backbone_config = load_config(BackboneConfig, args.backbone_config)
    training_config = load_config(TrainingConfig, args.training_config)
    if args.epochs is not None:
        training_config = dataclasses.replace(training_config, epochs=args.epochs)
    if args.seed is not None:
        training_config = dataclasses.replace(training_config, seed=args.seed)

    if args.data == "ptbxl":
        raise SystemExit(
            "PTB-XL loading is not implemented yet (Phase 7). Run with "
            "--data synthetic, or see scripts/download_data.sh."
        )

    synthetic_config = load_config(SyntheticConfig, args.synthetic_config)
    if args.n_recordings is not None:
        synthetic_config = dataclasses.replace(
            synthetic_config, n_recordings=args.n_recordings
        )

    print(
        f"generating {synthetic_config.n_recordings} synthetic recordings at "
        f"{synthetic_config.sampling_rate_hz} Hz ..."
    )
    cohort = generate_cohort(synthetic_config)

    # The model consumes the 100 Hz view; interval measurement (Phase 8) uses
    # the 500 Hz view of the very same recordings.
    signals = resample_signals(
        cohort.signals, synthetic_config.sampling_rate_hz, data_config.sampling_rate_hz
    )

    runs_dir = args.runs_dir or Path(training_config.runs_dir)
    with RunContext(
        training_config.experiment_name,
        runs_dir,
        seed=training_config.seed,
        configs={
            "data": data_config,
            "backbone": backbone_config,
            "training": training_config,
            "synthetic": synthetic_config,
        },
    ) as run:
        print(f"run directory: {run.dir}")
        result = train_age_regressor(
            signals=signals,
            ages=cohort.ages,
            sexes=cohort.sexes,
            patient_ids=cohort.patient_ids,
            record_ids=[r.record_id for r in cohort],
            data_config=data_config,
            backbone_config=backbone_config,
            training_config=training_config,
            run=run,
            progress=not args.quiet,
        )

    sizes = result.splits.sizes
    print(
        f"\nsplit sizes: train {sizes['train']}, val {sizes['val']}, "
        f"test {sizes['test']} recordings "
        f"({len(np.unique(cohort.patient_ids))} patients, split by patient)"
    )
    print(f"best epoch: {result.best_epoch} (val MAE {result.best_val_mae:.2f} years)")
    metrics = result.test_metrics
    print(
        f"test MAE {metrics['mae']:.2f} years | RMSE {metrics['rmse']:.2f} | "
        f"R^2 {metrics['r2']:.3f}"
    )
    baseline = float(
        np.mean(np.abs(result.predictions["test"].true_age - cohort.ages[result.splits.train].mean()))
    )
    print(f"constant-prediction baseline MAE: {baseline:.2f} years")
    print(f"artifacts: {result.checkpoint_path.parent if result.checkpoint_path else 'n/a'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
