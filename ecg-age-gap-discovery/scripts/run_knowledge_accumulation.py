#!/usr/bin/env python3
"""How much of the unexplained residual is an artefact of a short known-feature list.

The residual decomposition reports one attributable share against one list of
known measurements. That number is only interpretable relative to the list, and
the list is configuration rather than physics - on PTB-XL, growing it from five
timing intervals to fifteen measurements quadrupled the attributable share with
no change to the model at all.

Two points on a monotone quantity is an anecdote. This script measures the
whole function: it sweeps the *size* of the known-feature set, drawing random
subsets at each size, and reports the attributable share as a curve. The
right-hand end is the diagnostic:

    still climbing  ->  enumeration is unfinished, and the unexplained residual
                        is not yet admissible as a discovery candidate, because
                        the next measurement would have eaten some of it
    flat            ->  the vocabulary is exhausted and the residual survives
                        the hardest test this feature set can mount

The spread at fixed size is a second result, and arguably the sharper one: if
five features chosen one way explain a few percent and five chosen another way
explain several times that, then "we adjusted for N known measurements" does
not specify how hard a discovery claim was tested.

Examples
--------
Synthetic, quick::

    python scripts/run_knowledge_accumulation.py --n-recordings 900 --epochs 30

PTB-XL on the official folds::

    python scripts/run_knowledge_accumulation.py --data ptbxl --official-split \
        --epochs 60

Chapman, for cross-cohort comparison::

    python scripts/run_knowledge_accumulation.py --data chapman --limit 24000 \
        --epochs 60

A curve of the same shape in two cohorts on different continents makes the
finding a property of enumeration rather than of one dataset.
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
from ecg_discovery.validation.knowledge_accumulation import (  # noqa: E402
    sweep_known_features,
)

CONFIG_DIR = REPO_ROOT / "configs"

#: Feature sets worth placing on the curve by name rather than leaving to the
#: random draw. The five timing intervals are the project's original known-set,
#: and the paper's headline collapse is the distance between this point and the
#: full set - so it belongs on the same axes as the curve that explains it.
NAMED_SUBSETS: dict[str, tuple[str, ...]] = {
    "original_five_timing": (
        "heart_rate_bpm", "rr_sd_ms", "qrs_duration_ms", "pr_interval_ms",
        "qt_interval_ms",
    ),
    "timing_only": (
        "heart_rate_bpm", "rr_sd_ms", "qrs_duration_ms", "pr_interval_ms",
        "p_duration_ms", "qt_interval_ms", "qtc_bazett_ms",
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", choices=["synthetic", "ptbxl", "chapman"],
                        default="synthetic")
    parser.add_argument("--ptbxl-root", type=Path, default=Path("data/ptbxl"))
    parser.add_argument("--chapman-root", type=Path, default=Path("data/chapman"))
    parser.add_argument("--official-split", action="store_true",
                        help="use PTB-XL's own stratified folds (ptbxl only)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap recordings loaded (chapman is 43k after filtering)")
    parser.add_argument("--n-recordings", type=int, default=1500,
                        help="synthetic cohort size")
    parser.add_argument("--n-interval-recordings", type=int, default=2200,
                        help="test recordings measured at 500 Hz for the sweep")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws-per-size", type=int, default=8,
                        help="random subsets evaluated at each feature-set size")
    parser.add_argument("--subset-sizes", type=int, nargs="*", default=None,
                        help="sizes to sweep; default is dense where the curve bends")
    parser.add_argument("--runs-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Checked before anything is loaded: this run trains a model first, and
    # failing an argument check afterwards would waste the whole training pass.
    if args.official_split and args.data != "ptbxl":
        raise SystemExit("--official-split requires --data ptbxl")

    data_config = load_config(DataConfig, CONFIG_DIR / "data.yaml")
    backbone_config = load_config(BackboneConfig, CONFIG_DIR / "backbone.yaml")
    signal_config = load_config(
        SignalProcessingConfig, CONFIG_DIR / "signal_processing.yaml"
    )
    validation_config = load_config(
        ValidationFrameworkConfig, CONFIG_DIR / "validation_framework.yaml"
    )
    training_config = dataclasses.replace(
        load_config(TrainingConfig, CONFIG_DIR / "training.yaml"),
        epochs=args.epochs, seed=args.seed,
        experiment_name="knowledge_accumulation",
    )

    chosen_split = None
    load_500hz = None          # callable: test row positions -> 500 Hz waveforms

    if args.data == "ptbxl":
        from ecg_discovery.data.ptbxl_dataset import (
            load_ptbxl, load_waveform_subset, official_split, summarise_ptbxl,
        )

        print(summarise_ptbxl(args.ptbxl_root, data_config))
        cohort = load_ptbxl(args.ptbxl_root, data_config, limit=args.limit,
                            sampling_rate_hz=data_config.sampling_rate_hz,
                            progress=True)
        signals_model = cohort.signals
        ages, sexes = cohort.ages, cohort.sexes.astype(np.float64)
        patient_ids, record_ids = cohort.patient_ids, cohort.record_ids
        if args.official_split:
            chosen_split = official_split(cohort.strat_folds, patient_ids)
            print(f"using PTB-XL official folds: {chosen_split.sizes}")

        # The 500 Hz view is loaded only for the rows actually measured. The
        # whole cohort at 500 Hz is ~5 GB, and the sweep needs the test split.
        def load_500hz(rows):
            return load_waveform_subset(
                args.ptbxl_root, cohort.metadata, rows,
                data_config.interval_sampling_rate_hz,
            )

    elif args.data == "chapman":
        from ecg_discovery.data.chapman_dataset import (
            load_chapman, load_chapman_waveform_subset, summarise_chapman,
        )

        print(summarise_chapman(args.chapman_root, data_config))
        cohort = load_chapman(args.chapman_root, data_config, limit=args.limit,
                              progress=True)
        signals_model = cohort.signals
        ages, sexes = cohort.ages, cohort.sexes.astype(np.float64)
        patient_ids, record_ids = cohort.patient_ids, cohort.record_ids

        def load_500hz(rows):
            return load_chapman_waveform_subset(cohort.metadata, rows)

    else:
        synthetic = dataclasses.replace(
            load_config(SyntheticConfig, CONFIG_DIR / "synthetic.yaml"),
            n_recordings=args.n_recordings,
        )
        print(f"generating {synthetic.n_recordings} synthetic recordings at "
              f"{synthetic.sampling_rate_hz} Hz ...")
        generated = generate_cohort(synthetic)
        signals_500 = generated.signals
        signals_model = resample_signals(
            generated.signals, synthetic.sampling_rate_hz,
            data_config.sampling_rate_hz,
        )
        ages, sexes = generated.ages, generated.sexes.astype(np.float64)
        patient_ids = generated.patient_ids
        record_ids = [record.record_id for record in generated]

        def load_500hz(rows):
            return signals_500[rows]

    runs_dir = args.runs_dir or Path(training_config.runs_dir)
    with RunContext(
        training_config.experiment_name, runs_dir, seed=args.seed,
        configs={"data": data_config, "backbone": backbone_config,
                 "training": training_config, "signal_processing": signal_config,
                 "validation": validation_config},
        extra={"source": args.data, "draws_per_size": args.draws_per_size},
    ) as run:
        print(f"run directory: {run.dir}\n")

        result = train_age_regressor(
            signals=signals_model, ages=ages, sexes=sexes, patient_ids=patient_ids,
            record_ids=record_ids, data_config=data_config,
            backbone_config=backbone_config, training_config=training_config,
            run=run, progress=True, splits=chosen_split,
        )
        metrics = result.test_metrics
        print(f"\ntest MAE {metrics['mae']:.2f} years | R^2 {metrics['r2']:.3f}")

        test = result.predictions["test"]
        rows = test.indices[: args.n_interval_recordings]
        # test.* is ordered by test.indices, so a row's position there is what
        # indexes the age gap - not the row id itself.
        position = {int(index): i for i, index in enumerate(test.indices)}
        gap = np.array([test.age_gap[position[int(r)]] for r in rows])
        true_age = np.array([test.true_age[position[int(r)]] for r in rows])

        print(f"\nmeasuring classical intervals on {len(rows)} test recordings "
              f"at {data_config.interval_sampling_rate_hz} Hz ...")
        features = interval_features_table(
            load_500hz(rows), float(data_config.interval_sampling_rate_hz),
            signal_config, LEAD_NAMES,
            record_ids=[record_ids[i] for i in rows],
        )

        # Only name a subset if every feature in it was actually measured, so a
        # renamed or removed feature surfaces as a missing curve point rather
        # than an exception halfway through an hour-long run.
        named = {
            name: subset for name, subset in NAMED_SUBSETS.items()
            if all(feature in features.columns for feature in subset)
        }
        for name in set(NAMED_SUBSETS) - set(named):
            print(f"  note: named subset {name!r} skipped, features not all present")

        print(f"\nsweeping known-feature subsets "
              f"({args.draws_per_size} draws per size) ...")
        curve = sweep_known_features(
            gap, features, validation_config,
            ages=true_age, sexes=sexes[rows], patient_ids=patient_ids[rows],
            subset_sizes=args.subset_sizes or None,
            draws_per_size=args.draws_per_size,
            named_subsets=named,
        )

        print()
        print(curve.summary_text())

        if curve.named_points:
            print("\nnamed feature sets, on the same axes:")
            for point in curve.named_points:
                print(f"    {point.label:<24} k={point.n_features:<3} "
                      f"attributable {point.r2_incremental:>7.2%}")

        run.save_json("knowledge_accumulation.json", {
            "source": args.data,
            "n_recordings": curve.n_recordings,
            "n_dropped": curve.n_dropped,
            "available_features": list(curve.available_features),
            "covariates": list(curve.covariates),
            "test_mae": metrics["mae"],
            "full_set_incremental": curve.full_set_incremental,
            "full_set_incremental_ci": list(curve.full_set_incremental_ci),
            "log_slope": curve.log_slope,
            "tail_gain": curve.tail_gain,
            "extrapolated_ceiling": curve.extrapolated_ceiling,
            "saturation_fraction": curve.saturation_fraction,
            "is_saturated": curve.is_saturated(),
            "levels": curve.to_frame().to_dict(orient="records"),
            "points": curve.points_frame().to_dict(orient="records"),
            "summary_text": curve.summary_text(),
        })
        curve.to_frame().to_csv(run.artifacts_dir / "accumulation_curve.csv",
                                index=False)
        curve.points_frame().to_csv(run.artifacts_dir / "accumulation_points.csv",
                                    index=False)

        print(f"\nartifacts written to {run.artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
