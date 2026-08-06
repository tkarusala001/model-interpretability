#!/usr/bin/env python3
"""Test our own surviving result against a fuller enumeration of atrial knowledge.

WHAT THIS ANSWERS
-----------------
The paper's Result 2 reports that a model's age gap depends on atrial
information that classical P-wave measurement does not capture, where "classical"
meant three numbers: P duration, P amplitude and PR interval. Those three explain
0.7% of the age gap.

That is the paper's own thesis pointed at the paper. Result 1 shows that going
from five known features to fifteen quadrupled the attributable share and killed
an apparently significant finding. Leaving the P-wave claim tested against three
features, while naming four more as future work, applies our standard
asymmetrically - and it is exactly where a reviewer should press.

So we run it. This script adds the four measures the paper named:

    P terminal force (V1), P wave area, P notching, P wave dispersion

and re-measures how much of the age gap classical atrial measurement explains.

TWO DESIGN DECISIONS WORTH STATING
----------------------------------
1. **Full test set, not the attribution subset.** The published 0.7% was computed
   on the 400 recordings that attribution had been run on, because that script
   needed attribution for its other probes. The decomposition needs only age gaps
   and features, so it runs here on the entire test split. A negative result on
   400 recordings is weak evidence; on 2,151 it is worth reporting. We recompute
   the three-feature arm on the same recordings so the comparison is internal.

2. **Dispersion is gated on a measured reliability, not on judgement.** Its noise
   floor is estimated on a cohort whose true dispersion is zero by construction.
   If the floor swamps the between-subject spread, the feature enters as a
   reported sensitivity arm rather than as a measurement, and the number is what
   justifies that - not our opinion of it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_discovery.config import (  # noqa: E402
    DataConfig, SignalProcessingConfig, SyntheticConfig,
    ValidationFrameworkConfig, load_config,
)
from ecg_discovery.data.ptbxl_dataset import (  # noqa: E402
    load_ptbxl_metadata, load_waveform_subset,
)
from ecg_discovery.data.synthetic_ecg import LEAD_NAMES, generate_cohort  # noqa: E402
from ecg_discovery.signal_processing.atrial_features import (  # noqa: E402
    DispersionReliability, _dispersion_from_durations, dispersion_noise_floor,
    per_lead_p_durations,
)
from ecg_discovery.signal_processing.interval_features import (  # noqa: E402
    interval_features_table,
)
from ecg_discovery.signal_processing.qrs_detection import detect_r_peaks  # noqa: E402
from ecg_discovery.signal_processing.wave_delineation import delineate_beats  # noqa: E402
from ecg_discovery.validation.residual_decomposition import decompose_age_gap  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs"

#: The three measures the paper tested against.
CLASSICAL_P = ("p_duration_ms", "p_amplitude_mv", "pr_interval_ms")

#: The measures added here that need no new boundary detection.
EXTENDED_P = ("p_terminal_force_v1_mv_ms", "p_area_ii_mv_ms", "p_notch_depth_mv")

#: The measure that does, and whose reliability is therefore in question.
DISPERSION = ("p_dispersion_ms",)

# The paper's fifteen-feature known set is whatever validation_framework.yaml
# configures, and is read from there rather than restated here: a second copy
# would silently disagree with the one the published numbers were produced from.


def measure_dispersion_reliability(
    real_dispersion: np.ndarray, signal_config: SignalProcessingConfig,
    n_synthetic: int, seed: int,
) -> DispersionReliability:
    """Compare measured dispersion against what the method reports on zero truth.

    The synthetic cohort is generated with its default interference levels rather
    than clean, because the question is how the measurement behaves under the
    conditions it actually operates in. Its true dispersion is nonetheless
    exactly zero: every lead is a fixed multiple of one shared P component.
    """
    fs = 500.0
    cohort = generate_cohort(SyntheticConfig(
        n_recordings=n_synthetic, sampling_rate_hz=fs,
        duration_seconds=10.0, seed=seed,
    ))
    per_lead = []
    for signals in cohort.signals:
        detection = detect_r_peaks(signals, fs, signal_config, LEAD_NAMES)
        beats = delineate_beats(signals, fs, detection.r_peaks, signal_config, LEAD_NAMES)
        per_lead.append(per_lead_p_durations(signals, beats, fs, signal_config))

    finite = real_dispersion[np.isfinite(real_dispersion)]
    return DispersionReliability(
        noise_floor_ms=dispersion_noise_floor(per_lead),
        observed_sd_ms=float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
        n_synthetic=n_synthetic,
        n_real=int(finite.size),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptbxl-root", type=Path, default=Path("data/ptbxl"))
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("runs/discovery_experiment/20260729T050343Z"),
        help="run whose saved age gaps are analysed; no model is retrained",
    )
    parser.add_argument("--n-synthetic", type=int, default=200)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--n-stability-seeds", type=int, default=8)
    parser.add_argument("--refresh", action="store_true",
                        help="re-measure intervals instead of using the cache")
    parser.add_argument("--subset-size", type=int, default=400,
                        help="size of the attribution-sized arm, for comparison "
                             "with the published figure")
    args = parser.parse_args(argv)

    data_config = load_config(DataConfig, CONFIG_DIR / "data.yaml")
    signal_config = load_config(SignalProcessingConfig, CONFIG_DIR / "signal_processing.yaml")
    base_validation = load_config(
        ValidationFrameworkConfig, CONFIG_DIR / "validation_framework.yaml"
    )

    predictions = json.loads(
        (REPO_ROOT / args.run_dir / "artifacts" / "predictions.json").read_text()
    )["test"]
    rows = np.array(predictions["index"])
    gap = np.array(predictions["age_gap"])
    true_age = np.array(predictions["true_age"])

    print(f"analysing {len(rows)} test recordings from {args.run_dir}")
    print("no model is retrained; only the known-feature set changes\n")

    metadata = load_ptbxl_metadata(args.ptbxl_root, data_config)
    sexes = metadata.sex.to_numpy(dtype=np.float64)
    patients = metadata.patient_id.to_numpy()

    # Interval measurement dominates the runtime and does not depend on any of
    # the arms below, so it is cached against the run whose recordings it covers.
    cache = REPO_ROOT / "runs" / f"atrial_features_{args.run_dir.name}.csv"
    if cache.exists() and not args.refresh:
        import pandas as pd
        features = pd.read_csv(cache)
        print(f"reusing cached features from {cache}")
    else:
        print(f"measuring extended atrial features at "
              f"{data_config.interval_sampling_rate_hz} Hz ...")
        signals = load_waveform_subset(
            args.ptbxl_root, metadata, rows, data_config.interval_sampling_rate_hz
        )
        features = interval_features_table(
            signals, float(data_config.interval_sampling_rate_hz),
            signal_config, LEAD_NAMES,
        )
        del signals
        cache.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(cache, index=False)

    # ---------------------------------------------------------------- coverage
    print("\n" + "=" * 74)
    print("COVERAGE OF THE NEW MEASURES")
    print("=" * 74)
    print(f"{'feature':<30}{'measured':>10}{'non-zero':>10}{'median':>12}{'sd':>10}")
    for name in EXTENDED_P + DISPERSION:
        values = features[name].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        nonzero = float((finite != 0).mean()) if finite.size else float("nan")
        print(f"{name:<30}{finite.size / len(values):>9.1%}{nonzero:>10.1%}"
              f"{np.median(finite) if finite.size else float('nan'):>12.3f}"
              f"{finite.std(ddof=1) if finite.size > 1 else float('nan'):>10.3f}")

    # ------------------------------------------------------ dispersion gating
    print("\n" + "=" * 74)
    print("IS P-WAVE DISPERSION MEASURABLE HERE?")
    print("=" * 74)
    reliability = measure_dispersion_reliability(
        features["p_dispersion_ms"].to_numpy(dtype=np.float64),
        signal_config, args.n_synthetic, args.seed,
    )
    print(reliability.summary_text())
    print("\n  Clinical P-wave dispersion is reported in the 20-50 ms range, with")
    print("  abnormality thresholds near 40-50 ms. Compare that to the floor above.")

    # ------------------------------------------------------------ decomposition
    arms = [
        ("classical 3", CLASSICAL_P),
        ("extended 6", CLASSICAL_P + EXTENDED_P),
        ("extended 7 (+dispersion)", CLASSICAL_P + EXTENDED_P + DISPERSION),
    ]
    print("\n" + "=" * 74)
    print("HOW MUCH OF THE AGE GAP DOES ATRIAL MEASUREMENT EXPLAIN?")
    print("=" * 74)

    results = []
    for label, known in arms:
        validation = dataclasses.replace(
            base_validation, known_features=known,
            explainer_models=("gradient_boosting",), cv_folds=5,
        )
        decomposition = decompose_age_gap(
            gap, features, validation, ages=true_age,
            sexes=sexes[rows], patient_ids=patients[rows],
        )
        best = decomposition.most_explanatory
        print(f"\n{label:<26} attributable {best.r2_incremental:>7.2%}   "
              f"unexplained {best.unexplained_fraction:>7.2%}")
        for feature, score in sorted(best.univariate_r2.items(), key=lambda kv: -kv[1]):
            marker = "  <- new" if feature in EXTENDED_P + DISPERSION else ""
            print(f"    {feature:<30}{score:>+8.4f}{marker}")
        results.append({
            "arm": label, "known_features": list(known),
            "r2_incremental": best.r2_incremental,
            "unexplained_fraction": best.unexplained_fraction,
            "univariate_r2": best.univariate_r2,
        })

    classical, extended = results[0], results[1]
    change = extended["r2_incremental"] - classical["r2_incremental"]
    print("\n" + "-" * 74)
    print(f"adding terminal force, area and notching moves the attributable share "
          f"by {change:+.2%}")
    print(f"  {classical['r2_incremental']:.2%} -> {extended['r2_incremental']:.2%}")

    # ------------------------------------------------------------- stability
    # The explainer is a stochastic gradient-boosting fit, so a single run of
    # each arm gives a number with unstated run-to-run variation. The headline
    # here is a difference of about one percentage point; if that is inside the
    # estimator's own noise it must not be reported as a movement. Repeating both
    # arms across seeds, paired on the seed, answers that directly.
    print("\n" + "=" * 74)
    print("IS THE MOVEMENT LARGER THAN THE ESTIMATOR'S OWN NOISE?")
    print("=" * 74)
    paired: list[tuple[float, float]] = []
    for seed in range(args.n_stability_seeds):
        row = []
        for known in (CLASSICAL_P, CLASSICAL_P + EXTENDED_P):
            validation = dataclasses.replace(
                base_validation, known_features=known,
                explainer_models=("gradient_boosting",), cv_folds=5, seed=seed,
            )
            row.append(decompose_age_gap(
                gap, features, validation, ages=true_age,
                sexes=sexes[rows], patient_ids=patients[rows],
            ).most_explanatory.r2_incremental)
        paired.append((row[0], row[1]))
        print(f"  seed {seed}: classical {row[0]:>7.2%}   extended {row[1]:>7.2%}   "
              f"delta {row[1] - row[0]:>+7.2%}")

    classical_runs = np.array([p[0] for p in paired])
    extended_runs = np.array([p[1] for p in paired])
    deltas = extended_runs - classical_runs
    print(f"\n  classical 3  {classical_runs.mean():>7.2%}  "
          f"(sd {classical_runs.std(ddof=1):.2%})")
    print(f"  extended 6   {extended_runs.mean():>7.2%}  "
          f"(sd {extended_runs.std(ddof=1):.2%})")
    print(f"  paired delta {deltas.mean():>+7.2%}  (sd {deltas.std(ddof=1):.2%}, "
          f"{int((deltas > 0).sum())}/{len(deltas)} seeds positive)")

    # --------------------------------------------------- attribution-sized arm
    # The published 0.7% was measured on the recordings attribution had been run
    # on. Repeating both arms at that sample size separates "the new features
    # explain more" from "the original estimate was noisy".
    print("\n" + "=" * 74)
    print(f"THE SAME COMPARISON AT THE PUBLISHED SAMPLE SIZE (n={args.subset_size})")
    print("=" * 74)
    subset = slice(0, args.subset_size)
    for label, known in (("classical 3", CLASSICAL_P),
                         ("extended 6", CLASSICAL_P + EXTENDED_P)):
        validation = dataclasses.replace(
            base_validation, known_features=known,
            explainer_models=("gradient_boosting",), cv_folds=5,
        )
        decomposition = decompose_age_gap(
            gap[subset], features.iloc[subset].reset_index(drop=True), validation,
            ages=true_age[subset], sexes=sexes[rows[subset]],
            patient_ids=patients[rows[subset]],
        )
        best = decomposition.most_explanatory
        print(f"{label:<26} attributable {best.r2_incremental:>7.2%}   "
              f"unexplained {best.unexplained_fraction:>7.2%}")
        results.append({
            "arm": f"{label} (n={args.subset_size})", "known_features": list(known),
            "r2_incremental": best.r2_incremental,
            "unexplained_fraction": best.unexplained_fraction,
            "univariate_r2": best.univariate_r2,
        })

    # ------------------------------------------------------- Result 1 flank
    print("\n" + "=" * 74)
    print("THE SAME QUESTION FOR THE FIFTEEN-FEATURE SET (Result 1)")
    print("=" * 74)
    configured = tuple(base_validation.known_features)
    for label, known in ((f"{len(configured)} features", configured),
                         (f"{len(configured) + len(EXTENDED_P)} features (+atrial)",
                          configured + EXTENDED_P)):
        validation = dataclasses.replace(base_validation, known_features=known)
        decomposition = decompose_age_gap(
            gap, features, validation, ages=true_age,
            sexes=sexes[rows], patient_ids=patients[rows],
        )
        best = decomposition.most_explanatory
        print(f"{label:<26} attributable {best.r2_incremental:>7.2%}   "
              f"unexplained {best.unexplained_fraction:>7.2%}")
        results.append({
            "arm": label, "known_features": list(known),
            "r2_incremental": best.r2_incremental,
            "unexplained_fraction": best.unexplained_fraction,
            "univariate_r2": best.univariate_r2,
        })

    out = REPO_ROOT / "runs" / "atrial_enumeration.json"
    out.write_text(json.dumps({
        "run_dir": str(args.run_dir),
        "n_recordings": int(len(rows)),
        "dispersion_reliability": dataclasses.asdict(reliability)
        | {"usable": reliability.usable},
        "stability": {
            "classical": classical_runs.tolist(),
            "extended": extended_runs.tolist(),
            "delta_mean": float(deltas.mean()),
            "delta_sd": float(deltas.std(ddof=1)),
            "n_positive": int((deltas > 0).sum()),
        },
        "arms": results,
    }, indent=2, default=str))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
