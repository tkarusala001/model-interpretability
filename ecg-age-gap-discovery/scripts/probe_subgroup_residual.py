#!/usr/bin/env python3
"""Does the unexplained residual differ by subgroup - and does that mean anything?

A natural fairness analysis is to split the test set by a protected attribute and
compare how much of the age gap goes unexplained. If one group's residual is
larger, the model looks like it found more novel signal there, which in turn
looks like evidence that prior knowledge is worse for that group.

Run on PTB-XL by sex, that analysis returns a clean positive: the unexplained
share is about five points higher for women, in every explainer seed. It is
nonetheless not a finding about knowledge.

    unexplained  = 1 - R2_full
    attributable = R2_full - R2_base

Only *attributable* measures how much the known ECG features explain. The
unexplained share also moves with R2_base, which depends on how age is
distributed inside the stratum - and the female cohort here is older and more
age-dispersed. This script reports all three quantities per stratum, so the
difference can be located rather than assumed.

The general point is the paper's own thesis pointed at subgroup analysis:
comparing unexplained variance across groups with unequal covariate
distributions manufactures disparities the same way a thin known-feature set
manufactures discoveries.

No model is retrained; saved age gaps are reused.

NOTE ON SCOPE: PTB-XL and Chapman record age and sex only. Neither has race or
ethnicity, so the axis most fairness work cares about cannot be examined with
this data at all, and no result here should be read as covering it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_discovery.config import (  # noqa: E402
    DataConfig, ValidationFrameworkConfig, load_config,
)
from ecg_discovery.data.ptbxl_dataset import load_ptbxl_metadata  # noqa: E402
from ecg_discovery.validation.residual_decomposition import decompose_age_gap  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptbxl-root", type=Path, default=Path("data/ptbxl"))
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("runs/discovery_experiment/20260729T050343Z"),
    )
    parser.add_argument("--n-seeds", type=int, default=6)
    args = parser.parse_args(argv)

    validation = load_config(ValidationFrameworkConfig, CONFIG_DIR / "validation_framework.yaml")
    data_config = load_config(DataConfig, CONFIG_DIR / "data.yaml")
    metadata = load_ptbxl_metadata(args.ptbxl_root, data_config)

    predictions = json.loads(
        (REPO_ROOT / args.run_dir / "artifacts" / "predictions.json").read_text()
    )["test"]
    rows = np.array(predictions["index"])
    gap = np.array(predictions["age_gap"])
    age = np.array(predictions["true_age"])

    cache = REPO_ROOT / "runs" / f"atrial_features_{args.run_dir.name}.csv"
    if not cache.exists():
        print(f"error: {cache} not found. Run scripts/probe_atrial_enumeration.py first.",
              file=sys.stderr)
        return 1
    features = pd.read_csv(cache)

    sex = metadata.sex.to_numpy(dtype=np.float64)[rows]
    patients = metadata.patient_id.to_numpy()[rows]
    strata = {"male": sex == 0, "female": sex == 1}

    print(f"{len(rows)} test recordings; "
          f"{int(strata['female'].sum())} female, {int(strata['male'].sum())} male")
    print("\nage distribution per stratum (the confound to watch):")
    for name, mask in strata.items():
        print(f"  {name:<8} mean {age[mask].mean():5.1f}   SD {age[mask].std():5.1f}")

    results: dict[str, list[tuple[float, float, float]]] = {k: [] for k in strata}
    for seed in range(args.n_seeds):
        config = dataclasses.replace(validation, seed=seed)
        for name, mask in strata.items():
            # Sex is constant inside a stratum, so it cannot serve as a covariate
            # here; age remains one.
            best = decompose_age_gap(
                gap[mask], features[mask].reset_index(drop=True), config,
                ages=age[mask], patient_ids=patients[mask],
            ).most_explanatory
            results[name].append(
                (best.r2_baseline, best.r2_incremental, best.unexplained_fraction)
            )

    print(f"\nover {args.n_seeds} explainer seeds (mean, SD):")
    print(f"{'stratum':<10}{'R2_base':>16}{'attributable':>18}{'unexplained':>18}")
    summary = {}
    for name in strata:
        array = np.array(results[name])
        summary[name] = array
        print(f"{name:<10}"
              + "".join(f"{array[:, i].mean():>11.2%} ±{array[:, i].std(ddof=1):>5.2%}"
                        for i in range(3)))

    female, male = summary["female"], summary["male"]
    print("\npaired female - male:")
    for i, label in enumerate(("R2_base", "attributable", "unexplained")):
        delta = female[:, i] - male[:, i]
        direction = int((delta > 0).sum())
        print(f"  {label:<14}{delta.mean():>+8.2%}  (SD {delta.std(ddof=1):.2%}, "
              f"{direction}/{len(delta)} higher for women)")

    attributable_delta = female[:, 1] - male[:, 1]
    baseline_delta = female[:, 0] - male[:, 0]
    print("\nWhere does the unexplained gap come from?")
    print(f"  from the demographic baseline : {-baseline_delta.mean():+.2%}")
    print(f"  from the known ECG features   : {-attributable_delta.mean():+.2%}")
    print("\nIf the second is ~0 while the first is not, the apparent subgroup")
    print("difference is a covariate-distribution artefact, not evidence that")
    print("classical measurement serves one group worse than the other.")

    out = REPO_ROOT / "runs" / "subgroup_residual.json"
    out.write_text(json.dumps(
        {name: {"r2_baseline": array[:, 0].tolist(),
                "attributable": array[:, 1].tolist(),
                "unexplained": array[:, 2].tolist()}
         for name, array in summary.items()},
        indent=2,
    ))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
