#!/usr/bin/env python3
"""Is the atrial dependence mediated by rhythm, and therefore already known?

THE CONFOUND
------------
Occluding the P wave produces a trace resembling atrial fibrillation. AF
prevalence rises steeply with age, and the AF-age association is entirely
classical knowledge. A model that had learned only "no organised P wave means
older" would reproduce this project's P-wave attribution and occlusion results
exactly, and that would be rediscovery rather than discovery.

THE TEST
--------
The framework's own answer is to enumerate rhythm and see whether the residual
shrinks. Rhythm diagnosis is about as classical as ECG knowledge gets, yet the
known-feature set is entirely continuous measurements and contains no rhythm
class at all - a gap worth closing whether or not the confound is real.

Four arms, all on the same recordings, no model retrained:

  1. known measurements                       (the published enumeration)
  2. known measurements + rhythm indicators   (does rhythm add anything?)
  3. classical atrial measures                (the surviving claim's baseline)
  4. classical atrial measures + rhythm       (is the atrial gap rhythm-shaped?)

If arm 4 closes most of what arm 3 leaves, the atrial dependence is mediated by
rhythm and the claim should be withdrawn or heavily narrowed. If it does not,
the confound is not the explanation.

The script also reports the size of the sinus-only subset, which is what the
stratified attribution and occlusion re-runs need.

Example
-------
    python scripts/probe_rhythm_confound.py --run-dir runs/ptbxl_official/<stamp>
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
    DataConfig, SignalProcessingConfig, ValidationFrameworkConfig, load_config,
)
from ecg_discovery.data.ptbxl_dataset import (  # noqa: E402
    load_ptbxl_metadata, load_waveform_subset,
)
from ecg_discovery.data.rhythm_labels import (  # noqa: E402
    RHYTHM_FEATURE_NAMES, is_sinus_rhythm, rhythm_indicators,
)
from ecg_discovery.data.synthetic_ecg import LEAD_NAMES  # noqa: E402
from ecg_discovery.signal_processing.atrial_features import (  # noqa: E402
    ATRIAL_FEATURE_NAMES,
)
from ecg_discovery.signal_processing.interval_features import (  # noqa: E402
    interval_features_table,
)
from ecg_discovery.validation.residual_decomposition import decompose_age_gap  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs"
CLASSICAL_P = ("p_duration_ms", "p_amplitude_mv", "pr_interval_ms")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="run directory holding artifacts/predictions.json")
    parser.add_argument("--ptbxl-root", type=Path, default=Path("data/ptbxl"))
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "runs" / "rhythm_confound.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_config = load_config(DataConfig, CONFIG_DIR / "data.yaml")
    signal_config = load_config(SignalProcessingConfig,
                                CONFIG_DIR / "signal_processing.yaml")
    base = load_config(ValidationFrameworkConfig,
                       CONFIG_DIR / "validation_framework.yaml")

    predictions = json.loads(
        (REPO_ROOT / args.run_dir / "artifacts" / "predictions.json").read_text()
    )["test"]
    rows = np.array(predictions["index"])
    gap = np.array(predictions["age_gap"])
    true_age = np.array(predictions["true_age"])
    print(f"analysing {len(rows)} test recordings from {args.run_dir}")
    print("no model is retrained; only the known-feature set changes\n")

    metadata = load_ptbxl_metadata(args.ptbxl_root, data_config)
    sexes = metadata.sex.to_numpy(dtype=np.float64)[rows]
    patients = metadata.patient_id.to_numpy()[rows]
    scp = metadata.scp_codes.to_numpy()[rows]

    rhythm = rhythm_indicators(scp).reset_index(drop=True)
    sinus = is_sinus_rhythm(scp)
    print("rhythm composition of the analysed recordings:")
    for name in RHYTHM_FEATURE_NAMES:
        n = int(rhythm[name].sum())
        print(f"  {name:<32} {n:>5}  ({n / len(rows):.1%})")
    unlabelled = int((rhythm.sum(axis=1) == 0).sum())
    print(f"  {'(no rhythm statement)':<32} {unlabelled:>5}  "
          f"({unlabelled / len(rows):.1%})")
    print(f"\nsinus-only subset: {int(sinus.sum())} of {len(rows)} "
          f"({sinus.mean():.1%}) - this is what the stratified attribution and "
          "occlusion re-runs should use.\n")

    print(f"measuring intervals at {data_config.interval_sampling_rate_hz} Hz ...")
    signals = load_waveform_subset(args.ptbxl_root, metadata, rows,
                                   data_config.interval_sampling_rate_hz)
    features = interval_features_table(
        signals, float(data_config.interval_sampling_rate_hz),
        signal_config, LEAD_NAMES,
    ).reset_index(drop=True)
    features = pd.concat([features, rhythm], axis=1)

    configured = tuple(base.known_features)
    atrial = tuple(f for f in ATRIAL_FEATURE_NAMES if f in features.columns)
    arms = [
        ("known measurements", configured),
        ("known + rhythm", configured + RHYTHM_FEATURE_NAMES),
        ("classical atrial", CLASSICAL_P + atrial),
        ("classical atrial + rhythm", CLASSICAL_P + atrial + RHYTHM_FEATURE_NAMES),
    ]

    results = []
    print(f"{'arm':<30}{'attributable':>14}{'unexplained':>14}{'n':>7}")
    for label, known in arms:
        missing = [f for f in known if f not in features.columns]
        if missing:
            print(f"  {label}: skipped, missing {missing}")
            continue
        decomposition = decompose_age_gap(
            gap, features,
            dataclasses.replace(base, known_features=known),
            ages=true_age, sexes=sexes, patient_ids=patients,
            compute_univariate=False,
        )
        best = decomposition.most_explanatory
        low, high = best.r2_incremental_ci
        print(f"{label:<30}{best.r2_incremental:>13.2%} "
              f"{best.unexplained_fraction:>13.2%} {decomposition.n_recordings:>6}"
              f"   [{low:.2%}, {high:.2%}]")
        results.append({
            "arm": label, "known_features": list(known),
            "r2_incremental": best.r2_incremental,
            "r2_incremental_ci": list(best.r2_incremental_ci),
            "unexplained_fraction": best.unexplained_fraction,
            "n_recordings": decomposition.n_recordings,
        })

    by_arm = {r["arm"]: r for r in results}
    print()
    if "classical atrial" in by_arm and "classical atrial + rhythm" in by_arm:
        before = by_arm["classical atrial"]["r2_incremental"]
        after = by_arm["classical atrial + rhythm"]["r2_incremental"]
        print(f"VERDICT: adding rhythm to the atrial enumeration moves the "
              f"attributable share {before:.2%} -> {after:.2%} "
              f"({after - before:+.2%} pp).")
        print("A large move means the atrial dependence is rhythm-mediated and "
              "therefore already known; a small one means the confound is not "
              "the explanation. Neither is settled by this arm alone - the "
              "sinus-only re-run above is the decisive test, because within "
              "sinus rhythm P-wave presence is constant by construction.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "run_dir": str(args.run_dir),
        "n_recordings": int(len(rows)),
        "n_sinus": int(sinus.sum()),
        "sinus_indices": [int(i) for i in np.asarray(rows)[sinus]],
        "rhythm_counts": {n: int(rhythm[n].sum()) for n in RHYTHM_FEATURE_NAMES},
        "arms": results,
    }, indent=2))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
