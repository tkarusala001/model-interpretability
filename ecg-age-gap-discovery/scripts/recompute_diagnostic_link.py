#!/usr/bin/env python3
"""Recompute the diagnostic-link intervals under the corrected bootstrap.

An earlier version of ``unexplained_residual_diagnostic_link`` formed its
confidence interval from percentiles of the *per-fold* AUC deltas. That is not a
confidence interval: with five folds and a Bonferroni-corrected 0.5% tail the
0.5th percentile of five numbers is simply their minimum, so the significance
flag had degraded into "every fold happened to come out positive" - a sign test
with no stated error rate. Fold estimates are also not independent draws, since
their training sets overlap, so no percentile of them is calibrated for
anything.

The module now uses a paired percentile bootstrap over recordings, clustered by
patient where identifiers exist. This script re-derives the paper's Table 1
significance markers under that corrected procedure.

**No model is retrained.** Each run directory already contains the per-recording
age gaps, so only the interval measurement and the statistics are redone. The
AUC point estimates are unaffected by the fix; what changes is whether their
intervals exclude zero.
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
    DataConfig, SignalProcessingConfig, ValidationFrameworkConfig, load_config,
)
from ecg_discovery.data.ptbxl_dataset import (  # noqa: E402
    DIAGNOSTIC_SUPERCLASSES, load_ptbxl_metadata, load_waveform_subset,
)
from ecg_discovery.data.synthetic_ecg import LEAD_NAMES  # noqa: E402
from ecg_discovery.discovery_experiments.unexplained_residual_diagnostic_link import (  # noqa: E402
    evaluate_diagnostic_link,
)
from ecg_discovery.signal_processing.interval_features import (  # noqa: E402
    interval_features_table,
)
from ecg_discovery.validation.residual_decomposition import decompose_age_gap  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs"

# The three analyses reported as Table 1, in paper order.
CONFIGURATIONS = [
    ("A", "runs/discovery_experiment/20260729T034003Z", 5),
    ("B", "runs/discovery_experiment/20260729T035606Z", 5),
    ("C", "runs/discovery_experiment/20260729T050343Z", 15),
]

FIVE_FEATURES = ("heart_rate_bpm", "qrs_duration_ms", "pr_interval_ms",
                 "qt_interval_ms", "qtc_bazett_ms")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptbxl-root", type=Path, default=Path("data/ptbxl"))
    parser.add_argument("--max-recordings", type=int, default=2200,
                        help="cap per configuration, for tractable interval measurement")
    args = parser.parse_args(argv)

    data_config = load_config(DataConfig, CONFIG_DIR / "data.yaml")
    signal_config = load_config(SignalProcessingConfig, CONFIG_DIR / "signal_processing.yaml")
    base_validation = load_config(
        ValidationFrameworkConfig, CONFIG_DIR / "validation_framework.yaml"
    )

    print("loading PTB-XL metadata ...")
    metadata = load_ptbxl_metadata(args.ptbxl_root, data_config)
    labels_all = metadata[[f"dx_{n}" for n in DIAGNOSTIC_SUPERCLASSES]].to_numpy()
    sexes_all = metadata.sex.to_numpy(dtype=np.float64)
    patients_all = metadata.patient_id.to_numpy()

    results = []
    for name, run_dir, n_features in CONFIGURATIONS:
        artifacts = REPO_ROOT / run_dir / "artifacts"
        predictions = json.loads((artifacts / "predictions.json").read_text())["test"]
        rows = np.array(predictions["index"])[: args.max_recordings]
        gap = np.array(predictions["age_gap"])[: args.max_recordings]
        true_age = np.array(predictions["true_age"])[: args.max_recordings]

        print(f"\n[{name}] {len(rows)} recordings, {n_features} known features; "
              "measuring intervals at 500 Hz ...")
        signals = load_waveform_subset(
            args.ptbxl_root, metadata, rows, data_config.interval_sampling_rate_hz
        )
        features = interval_features_table(
            signals, float(data_config.interval_sampling_rate_hz),
            signal_config, LEAD_NAMES,
        )
        del signals

        validation = base_validation
        if n_features == 5:
            validation = dataclasses.replace(base_validation, known_features=FIVE_FEATURES)

        decomposition = decompose_age_gap(
            gap, features, validation, ages=true_age, sexes=sexes_all[rows],
            patient_ids=patients_all[rows],
        )
        best = decomposition.most_explanatory

        known = features[list(validation.known_features)].to_numpy(dtype=np.float64)
        usable = np.isfinite(known).all(axis=1) & np.isfinite(gap)
        kept = rows[usable]

        report = evaluate_diagnostic_link(
            best.unexplained_residual,
            features[usable].reset_index(drop=True),
            labels_all[kept],
            DIAGNOSTIC_SUPERCLASSES,
            validation,
            ages=true_age[usable], sexes=sexes_all[kept], patient_ids=patients_all[kept],
        )

        print(f"  attributable {best.r2_incremental:6.1%} | "
              f"unexplained {best.unexplained_fraction:6.1%}")
        print(report.summary_text())

        results.append({
            "configuration": name,
            "n_known_features": n_features,
            "n_recordings": report.n_recordings,
            "attributable": best.r2_incremental,
            "attributable_ci": list(best.r2_incremental_ci),
            "unexplained": best.unexplained_fraction,
            "links": report.to_frame().to_dict(orient="records"),
            "any_improvement": report.any_improvement,
            "all_equivalent": report.all_equivalent,
            "inconclusive": list(report.inconclusive),
            "summary": report.summary_text(),
        })

    print("\n" + "=" * 78)
    print("TABLE 1 UNDER THE CORRECTED BOOTSTRAP")
    print("=" * 78)
    print(f"{'':<22}" + "".join(f"{r['configuration']:>16}" for r in results))
    print(f"{'attributable':<22}" + "".join(f"{r['attributable']:>15.1%}" for r in results))
    print(f"{'unexplained':<22}" + "".join(f"{r['unexplained']:>15.1%}" for r in results))
    for superclass in DIAGNOSTIC_SUPERCLASSES:
        cells = []
        for result in results:
            row = next((x for x in result["links"] if x["superclass"] == superclass), None)
            if row is None:
                cells.append(f"{'skipped':>15}")
            else:
                mark = "*" if row["improves"] else " "
                cells.append(f"{row['delta_auc']:>+14.4f}{mark}")
        print(f"{'dAUC ' + superclass:<22}" + "".join(cells))
    print("\n* interval excludes zero (paired bootstrap, Bonferroni-corrected)")

    out = REPO_ROOT / "runs" / "corrected_diagnostic_link.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
