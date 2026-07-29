#!/usr/bin/env python3
"""The full discovery pipeline: train, decompose, then test for real signal.

Chains every stage that a claim of discovery depends on:

1. Train the age regressor and take its age gap.
2. Measure classical ECG intervals independently at 500 Hz.
3. Decompose the gap into what those intervals explain and what they do not.
4. Ask whether the *unexplained* part predicts cardiologist-assigned diagnostic
   superclasses better than the intervals alone.

Step 4 reports whichever answer it gets. A null result is a genuine finding for
this project: it would show that attribution plus an unexplained residual is not
evidence of a discovered biomarker.

Examples
--------
    python scripts/run_discovery_experiment.py --n-recordings 1500 --epochs 40
    python scripts/run_discovery_experiment.py --data ptbxl
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

from ecg_discovery.analysis.visualization import (  # noqa: E402
    plot_age_gap_scatter,
    plot_diagnostic_link,
    plot_residual_decomposition,
)
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
from ecg_discovery.data.synthetic_ecg import (  # noqa: E402
    DIAGNOSTIC_CLASSES,
    LEAD_NAMES,
    generate_cohort,
)
from ecg_discovery.discovery_experiments.unexplained_residual_diagnostic_link import (  # noqa: E402
    evaluate_diagnostic_link,
)
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
    parser.add_argument(
        "--official-split", action="store_true",
        help="use PTB-XL's own strat_fold split (1-8 train, 9 val, 10 test) "
             "instead of a fresh patient-level split. Folds 9 and 10 received "
             "human over-reading, and published work uses them, so this makes "
             "results comparable with the literature.",
    )
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
        epochs=args.epochs, seed=args.seed, experiment_name="discovery_experiment",
    )

    # The 500 Hz view is loaded later, for the test split only. Loading all of
    # PTB-XL at 500 Hz costs ~5 GB as float32 against ~0.5 GB for the test
    # split, and interval measurement never needs the rest.
    load_intervals_later = None
    if args.data == "ptbxl":
        from ecg_discovery.data.ptbxl_dataset import (
            DIAGNOSTIC_SUPERCLASSES,
            load_ptbxl,
            load_waveform_subset,
            summarise_ptbxl,
        )

        print(summarise_ptbxl(args.ptbxl_root, data_config))
        model_view = load_ptbxl(args.ptbxl_root, data_config,
                                sampling_rate_hz=data_config.sampling_rate_hz, progress=True)
        signals_model = model_view.signals
        signals_intervals = None
        ages, sexes = model_view.ages, model_view.sexes.astype(np.float64)
        patient_ids, record_ids = model_view.patient_ids, model_view.record_ids
        diagnostic_labels = model_view.diagnostic_labels
        superclass_names = DIAGNOSTIC_SUPERCLASSES

        def load_intervals_later(indices):
            return load_waveform_subset(
                args.ptbxl_root, model_view.metadata, indices,
                data_config.interval_sampling_rate_hz,
            )
    else:
        synthetic = dataclasses.replace(
            load_config(SyntheticConfig, CONFIG_DIR / "synthetic.yaml"),
            n_recordings=args.n_recordings,
        )
        print(f"generating {synthetic.n_recordings} synthetic recordings ...")
        cohort = generate_cohort(synthetic)
        signals_intervals = cohort.signals
        signals_model = resample_signals(
            cohort.signals, synthetic.sampling_rate_hz, data_config.sampling_rate_hz
        )
        ages, sexes = cohort.ages, cohort.sexes.astype(np.float64)
        patient_ids = cohort.patient_ids
        record_ids = [r.record_id for r in cohort]
        diagnostic_labels = cohort.diagnostic_labels
        superclass_names = DIAGNOSTIC_CLASSES
        print(
            "  NOTE: synthetic diagnostic labels are linked to the "
            f"'{synthetic.diagnostic_link_source}' channel by construction. This "
            "run validates the machinery; it says nothing about real ECGs."
        )

    runs_dir = args.runs_dir or Path(training_config.runs_dir)
    with RunContext(
        training_config.experiment_name, runs_dir, seed=args.seed,
        configs={"data": data_config, "backbone": backbone_config,
                 "training": training_config, "signal_processing": signal_config,
                 "validation": validation_config},
        extra={"source": args.data},
    ) as run:
        print(f"run directory: {run.dir}\n")

        # -- 1. Train ---------------------------------------------------------
        chosen_split = None
        if args.official_split:
            if args.data != "ptbxl":
                raise SystemExit("--official-split requires --data ptbxl")
            from ecg_discovery.data.ptbxl_dataset import official_split

            chosen_split = official_split(model_view.strat_folds, patient_ids)
            print(f"using PTB-XL official folds: {chosen_split.sizes}")

        result = train_age_regressor(
            signals=signals_model, ages=ages, sexes=sexes, patient_ids=patient_ids,
            record_ids=record_ids, data_config=data_config,
            backbone_config=backbone_config, training_config=training_config,
            run=run, progress=True, splits=chosen_split,
        )
        test = result.predictions["test"]
        print(f"\ntest MAE {result.test_metrics['mae']:.2f} years | "
              f"R^2 {result.test_metrics['r2']:.3f}")

        # -- 2. Measure classical intervals -----------------------------------
        print(f"\nmeasuring intervals on {len(test.indices)} test recordings at "
              f"{data_config.interval_sampling_rate_hz} Hz ...")
        test_signals_500 = (
            load_intervals_later(test.indices) if load_intervals_later is not None
            else signals_intervals[test.indices]
        )
        features = interval_features_table(
            test_signals_500,
            float(data_config.interval_sampling_rate_hz),
            signal_config, LEAD_NAMES,
        )
        del test_signals_500

        # -- 3. Decompose -----------------------------------------------------
        decomposition = decompose_age_gap(
            test.age_gap, features, validation_config,
            ages=test.true_age, sexes=sexes[test.indices],
            patient_ids=patient_ids[test.indices],
        )
        print()
        print(decomposition.summary_text())
        best = decomposition.most_explanatory

        # -- 4. Test the unexplained residual for independent signal ----------
        # The residual is defined only on the rows that survived interval
        # measurement, so labels and covariates are aligned to the same subset.
        known = features[list(validation_config.known_features)].to_numpy(dtype=np.float64)
        usable = np.isfinite(known).all(axis=1) & np.isfinite(test.age_gap)
        rows = test.indices[usable]

        print("\n" + "=" * 72)
        report = evaluate_diagnostic_link(
            best.unexplained_residual,
            features[usable].reset_index(drop=True),
            diagnostic_labels[rows],
            superclass_names,
            validation_config,
            ages=test.true_age[usable],
            sexes=sexes[rows],
            patient_ids=patient_ids[rows],
        )
        print(report.summary_text())
        print("=" * 72)

        # -- Artifacts --------------------------------------------------------
        run.save_json("decomposition.json", {
            "summary_text": decomposition.summary_text(),
            "explainers": {
                name: {
                    "r2_baseline": e.r2_baseline, "r2_full": e.r2_full,
                    "r2_incremental": e.r2_incremental,
                    "unexplained_fraction": e.unexplained_fraction,
                    "univariate_r2": e.univariate_r2,
                }
                for name, e in decomposition.explainers.items()
            },
        })
        run.save_json("diagnostic_link.json", {
            "summary_text": report.summary_text(),
            "any_improvement": report.any_improvement,
            "correction": report.correction,
            "n_recordings": report.n_recordings,
            "skipped": report.skipped,
            "links": report.to_frame().to_dict(orient="records"),
        })
        plot_residual_decomposition(
            decomposition,
            title=f"Age-gap residual decomposition ({args.data})",
            path=run.artifact_path("residual_decomposition.png"),
        )
        plot_diagnostic_link(
            report,
            title=f"Does the unexplained residual predict diagnosis? ({args.data})",
            path=run.artifact_path("diagnostic_link.png"),
        )
        plot_age_gap_scatter(
            test.true_age, test.predicted_age,
            title=f"ECG age gap ({args.data} test set)",
            path=run.artifact_path("age_gap_scatter.png"),
        )
        print(f"\nartifacts written to {run.artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
