#!/usr/bin/env python3
"""Fiducial-segment attribution over a test set, focused on age-gap outliers.

Trains (or loads) the age regressor, finds the recordings the model got most
wrong in each direction - predicted much older, and predicted much younger -
and produces for each one a fiducial-segment summary plus an overlay plot of
attribution on the actual waveform with detected fiducial points marked.

The outliers are the interesting cases precisely because they are the errors:
a large age gap is either the model finding something, or the model being
wrong, and the point of the artifacts produced here is to let a cardiologist
tell which. Whether any of it corresponds to independently verifiable
information is not settled here - that is Phase 8.

Runs on the synthetic cohort by default, so it works without PTB-XL.

Examples
--------
Synthetic, quick::

    python scripts/run_attribution_analysis.py --n-recordings 600 --epochs 25

Real data, once downloaded via ``scripts/download_data.sh``::

    python scripts/run_attribution_analysis.py --data ptbxl --ptbxl-root data/ptbxl
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_discovery.analysis.visualization import (  # noqa: E402
    plot_age_gap_scatter,
    plot_attribution_overlay,
    plot_segment_summary,
)
from ecg_discovery.config import (  # noqa: E402
    BackboneConfig,
    DataConfig,
    SignalProcessingConfig,
    SyntheticConfig,
    TrainingConfig,
    load_config,
)
from ecg_discovery.data.preprocessing import resample_signals  # noqa: E402
from ecg_discovery.data.synthetic_ecg import LEAD_NAMES, generate_cohort  # noqa: E402
from ecg_discovery.interpretability.attribution import integrated_gradients  # noqa: E402
from ecg_discovery.interpretability.fiducial_attribution import (  # noqa: E402
    aggregate_by_fiducial_segment,
    summarise_cohort,
)
from ecg_discovery.runtime import RunContext  # noqa: E402
from ecg_discovery.signal_processing.qrs_detection import detect_r_peaks  # noqa: E402
from ecg_discovery.signal_processing.wave_delineation import delineate_beats  # noqa: E402
from ecg_discovery.training.train import train_age_regressor  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", choices=["synthetic", "ptbxl"], default="synthetic")
    parser.add_argument("--ptbxl-root", type=Path, default=Path("data/ptbxl"))
    parser.add_argument("--n-recordings", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-outliers", type=int, default=5,
                        help="outliers to inspect in each direction")
    parser.add_argument("--n-attribution", type=int, default=200,
                        help="test recordings to attribute for the cohort summary")
    parser.add_argument("--ig-steps", type=int, default=64)
    parser.add_argument("--runs-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _load_data(args, data_config: DataConfig):
    """Return signals at the model rate plus labels, from either source."""
    if args.data == "ptbxl":
        from ecg_discovery.data.ptbxl_dataset import load_ptbxl, summarise_ptbxl

        print(summarise_ptbxl(args.ptbxl_root, data_config))
        data = load_ptbxl(args.ptbxl_root, data_config, progress=True)
        return (
            data.signals, data.ages, data.sexes.astype(np.float64),
            data.patient_ids, data.record_ids, LEAD_NAMES,
        )

    synthetic = dataclasses.replace(
        load_config(SyntheticConfig, CONFIG_DIR / "synthetic.yaml"),
        n_recordings=args.n_recordings,
    )
    print(f"generating {synthetic.n_recordings} synthetic recordings ...")
    cohort = generate_cohort(synthetic)
    signals = resample_signals(
        cohort.signals, synthetic.sampling_rate_hz, data_config.sampling_rate_hz
    )
    return (
        signals, cohort.ages, cohort.sexes.astype(np.float64),
        cohort.patient_ids, [r.record_id for r in cohort], LEAD_NAMES,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    data_config = load_config(DataConfig, CONFIG_DIR / "data.yaml")
    backbone_config = load_config(BackboneConfig, CONFIG_DIR / "backbone.yaml")
    signal_config = load_config(SignalProcessingConfig, CONFIG_DIR / "signal_processing.yaml")
    training_config = dataclasses.replace(
        load_config(TrainingConfig, CONFIG_DIR / "training.yaml"),
        epochs=args.epochs, seed=args.seed, experiment_name="attribution_analysis",
    )

    signals, ages, sexes, patient_ids, record_ids, lead_names = _load_data(
        args, data_config
    )
    rate = data_config.sampling_rate_hz

    runs_dir = args.runs_dir or Path(training_config.runs_dir)
    with RunContext(
        training_config.experiment_name, runs_dir, seed=args.seed,
        configs={"data": data_config, "backbone": backbone_config,
                 "training": training_config, "signal_processing": signal_config},
        extra={"source": args.data},
    ) as run:
        print(f"run directory: {run.dir}\n")

        result = train_age_regressor(
            signals=signals, ages=ages, sexes=sexes, patient_ids=patient_ids,
            record_ids=record_ids, data_config=data_config,
            backbone_config=backbone_config, training_config=training_config,
            run=run, progress=True,
        )
        metrics = result.test_metrics
        print(
            f"\ntest MAE {metrics['mae']:.2f} years | RMSE {metrics['rmse']:.2f} "
            f"| R^2 {metrics['r2']:.3f}"
        )

        test = result.predictions["test"]
        age_gap = test.age_gap
        print(f"age gap: mean {age_gap.mean():+.2f}, sd {age_gap.std():.2f} years")

        # -- Outliers in both directions --------------------------------------
        order = np.argsort(age_gap)
        youngest_looking = order[-args.n_outliers:][::-1]   # predicted much older
        oldest_looking = order[: args.n_outliers]           # predicted much younger
        selected = np.concatenate([youngest_looking, oldest_looking])

        normalised = result.normalizer.transform(signals)

        # -- Cohort-level attribution -----------------------------------------
        n_cohort = min(args.n_attribution, len(test.indices))
        cohort_positions = np.arange(n_cohort)
        print(f"\nattributing {n_cohort} test recordings (Integrated Gradients, "
              f"{args.ig_steps} steps) ...")

        summaries = []
        for start in range(0, n_cohort, 32):
            block = cohort_positions[start : start + 32]
            rows = test.indices[block]
            attribution = integrated_gradients(
                result.model,
                torch.tensor(normalised[rows]),
                torch.tensor(sexes[rows], dtype=torch.float32),
                n_steps=args.ig_steps,
            )
            for offset, row in enumerate(rows):
                detection = detect_r_peaks(signals[row], rate, signal_config, lead_names)
                beats = delineate_beats(
                    signals[row], rate, detection.r_peaks, signal_config, lead_names
                )
                summaries.append(
                    aggregate_by_fiducial_segment(
                        attribution.attributions[offset], beats, lead_names,
                        prediction=float(attribution.prediction[offset]),
                        baseline_prediction=float(attribution.baseline_prediction[offset]),
                        record_id=str(record_ids[row]),
                    )
                )

        cohort_summary = summarise_cohort(summaries)
        run.save_json("cohort_segment_summary.json",
                      cohort_summary.to_dict(orient="records"))

        by_segment = cohort_summary.groupby("segment", sort=False).agg(
            share=("share_mean", "sum"), density=("density_mean", "mean")
        )
        print("\ncohort-level attribution by segment:")
        print("  segment    share of total    density per sample")
        for segment in ("P", "QRS", "T", "other"):
            row = by_segment.loc[segment]
            print(f"  {segment:<9}  {row['share']:>13.1%}  {row['density']:>19.4g}")
        print("  (share is confounded by segment width; density is not)")

        # -- Per-outlier artifacts --------------------------------------------
        print(f"\nwriting artifacts for {len(selected)} age-gap outliers ...")
        outlier_report = []
        for rank, position in enumerate(selected):
            row = test.indices[position]
            attribution = integrated_gradients(
                result.model,
                torch.tensor(normalised[row : row + 1]),
                torch.tensor(sexes[row : row + 1], dtype=torch.float32),
                n_steps=args.ig_steps,
            )
            detection = detect_r_peaks(signals[row], rate, signal_config, lead_names)
            beats = delineate_beats(
                signals[row], rate, detection.r_peaks, signal_config, lead_names
            )
            summary = aggregate_by_fiducial_segment(
                attribution.attributions[0], beats, lead_names,
                prediction=float(attribution.prediction[0]),
                baseline_prediction=float(attribution.baseline_prediction[0]),
                record_id=str(record_ids[row]),
            )

            direction = "older" if age_gap[position] > 0 else "younger"
            label = f"{direction}_{abs(age_gap[position]):05.1f}y_{record_ids[row]}"
            title = (
                f"{record_ids[row]}: true age {test.true_age[position]:.0f}, "
                f"predicted {test.predicted_age[position]:.0f} "
                f"(gap {age_gap[position]:+.1f} years)"
            )
            plot_attribution_overlay(
                signals[row], attribution.attributions[0], beats, lead_names, rate,
                title=title, path=run.artifact_path(f"outliers/{label}_overlay.png"),
            )
            plot_segment_summary(
                summary, title=title,
                path=run.artifact_path(f"outliers/{label}_segments.png"),
            )
            outlier_report.append({
                "record_id": record_ids[row],
                "true_age": float(test.true_age[position]),
                "predicted_age": float(test.predicted_age[position]),
                "age_gap": float(age_gap[position]),
                "n_beats": summary.n_beats,
                "segment_share": {
                    name: summary.segment_share(name) for name in ("P", "QRS", "T", "other")
                },
                "segment_density": {
                    name: summary.segment_density(name) for name in ("P", "QRS", "T", "other")
                },
                "dominant_segment": summary.dominant_segment(),
                "summary_text": summary.summary_text(),
            })
            if rank < 2:
                print(f"\n{title}")
                print(summary.summary_text())

        run.save_json("age_gap_outliers.json", outlier_report)
        plot_age_gap_scatter(
            test.true_age, test.predicted_age, highlight=selected,
            title=f"ECG age gap on the {args.data} test set",
            path=run.artifact_path("age_gap_scatter.png"),
        )

        print(f"\nartifacts written to {run.artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
