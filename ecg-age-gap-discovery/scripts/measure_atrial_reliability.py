#!/usr/bin/env python3
"""How reliably do we measure the P wave, and what does that do to the residual?

WHY THIS EXISTS
---------------
Inequality (ii) of the protocol says a measurement correlating with truth at
$r$ recovers only $r^2$ of the variance it genuinely explains, so an unreliable
known feature inflates the unexplained residual and makes a discovery claim
look larger than it is.

This project reports reliability for QRS duration, PR and QT -- and nothing for
any atrial measurement, which is precisely where its surviving claim lives. P
waves are roughly 0.1 mV, comparable to the noise floor of a real recording,
and the boundary search failed on 19.5% of real recordings before it was fixed.
The attenuation term in our own bound is therefore unmeasured exactly where it
matters most.

This script measures it against constructed ground truth, at several noise
levels, using the real measurement pipeline.

WHY COHORTS ARE POOLED ACROSS PARAMETER VALUES
----------------------------------------------
The synthetic generator holds P duration and amplitude fixed across a cohort,
so within one cohort the true value has no variance and a correlation against
it is undefined. We therefore generate several small cohorts spanning
physiological ranges and pool them, which gives the true value the between-
recording variance that reliability is defined against.

WHAT CANNOT BE MEASURED THIS WAY
---------------------------------
Notch depth has a true value of *zero* in every synthetic recording, since the
P wave is a smooth raised cosine. Correlation against a constant is undefined,
so for notching we report the false-positive rate under noise instead -- the
quantity that actually matters for a detector whose failure mode is finding
structure that is not there.

Example
-------
    python scripts/measure_atrial_reliability.py --per-cell 12
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
    SignalProcessingConfig, SyntheticConfig, load_config,
)
from ecg_discovery.data.synthetic_ecg import LEAD_NAMES, generate_cohort  # noqa: E402
from ecg_discovery.signal_processing.interval_features import (  # noqa: E402
    interval_features_table,
)

CONFIG_DIR = REPO_ROOT / "configs"

#: Physiological spread. Normal P duration is about 80-110 ms and amplitude
#: about 0.05-0.25 mV; the grid runs slightly wider so the correlation is not
#: computed over a range too narrow to be informative.
P_DURATIONS_MS = (80.0, 90.0, 100.0, 110.0, 120.0)
P_AMPLITUDES_MV = (0.08, 0.12, 0.16, 0.20)

#: Recording quality. The configured default is 0.02 mV; real 12-lead
#: recordings are frequently worse, and the point of the sweep is to show where
#: the measurement stops carrying information.
NOISE_LEVELS_MV = (0.0, 0.02, 0.05, 0.10)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--per-cell", type=int, default=10,
                        help="recordings generated per (duration, amplitude) cell")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "runs" / "atrial_reliability.json")
    return parser.parse_args(argv)


def _measure(synthetic: SyntheticConfig, signal_config: SignalProcessingConfig):
    cohort = generate_cohort(synthetic)
    return interval_features_table(
        cohort.signals, float(synthetic.sampling_rate_hz), signal_config, LEAD_NAMES
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = load_config(SyntheticConfig, CONFIG_DIR / "synthetic.yaml")
    signal_config = load_config(SignalProcessingConfig,
                                CONFIG_DIR / "signal_processing.yaml")

    results = []
    for noise in NOISE_LEVELS_MV:
        rows = []
        for i, duration in enumerate(P_DURATIONS_MS):
            for j, amplitude in enumerate(P_AMPLITUDES_MV):
                synthetic = dataclasses.replace(
                    base,
                    n_recordings=args.per_cell,
                    p_duration_ms=duration,
                    p_amplitude_mv=amplitude,
                    noise_mv_sd=noise,
                    # Vary the seed per cell so cells are independent draws
                    # rather than the same cohort with a shifted P wave.
                    seed=args.seed + 1000 * i + j,
                )
                measured = _measure(synthetic, signal_config)
                measured["true_p_duration_ms"] = duration
                measured["true_p_amplitude_mv"] = amplitude
                # A raised cosine of amplitude A and width W has area A*W/2;
                # the constant cancels in a correlation.
                measured["true_p_area"] = amplitude * duration / 2.0
                rows.append(measured)

        frame = pd.concat(rows, ignore_index=True)
        pairs = [
            ("p_duration_ms", "true_p_duration_ms"),
            ("p_amplitude_mv", "true_p_amplitude_mv"),
            ("p_area_ii_mv_ms", "true_p_area"),
            # Reported for contrast: the measurement this project already
            # validated, so the atrial numbers can be read against something.
            ("pr_interval_ms", None),
        ]
        print(f"\nnoise {noise:.3f} mV  ({len(frame)} recordings)")
        print(f"  {'measurement':<22}{'n':>6}{'r':>8}{'r^2':>8}   attenuation")
        for measured_name, truth_name in pairs:
            if measured_name not in frame.columns or truth_name is None:
                continue
            usable = frame[[measured_name, truth_name]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            n = len(usable)
            if n < 10 or usable[measured_name].std() == 0:
                print(f"  {measured_name:<22}{n:>6}{'  n/a':>8}"
                      "     not estimable (no variance or too few measured)")
                results.append({"noise_mv": noise, "measurement": measured_name,
                                "n": n, "r": None, "r2": None})
                continue
            r = float(np.corrcoef(usable[measured_name], usable[truth_name])[0, 1])
            print(f"  {measured_name:<22}{n:>6}{r:>8.3f}{r ** 2:>8.3f}"
                  f"   {1 - r ** 2:.0%} of explainable variance lost")
            results.append({"noise_mv": noise, "measurement": measured_name,
                            "n": n, "r": r, "r2": r ** 2,
                            "detected_fraction": n / len(frame)})

        # Notching: true depth is zero everywhere, so the meaningful quantity is
        # how often the detector claims a notch that is not there.
        if "p_notch_depth_mv" in frame.columns:
            notch = frame["p_notch_depth_mv"].dropna()
            rate = float((notch > 0).mean()) if len(notch) else float("nan")
            print(f"  {'p_notch_depth_mv':<22}{len(notch):>6}{'  --':>8}"
                  f"        false-positive rate {rate:.1%} (true depth is 0)")
            results.append({"noise_mv": noise, "measurement": "p_notch_depth_mv",
                            "n": int(len(notch)), "r": None, "r2": None,
                            "false_positive_rate": rate})

    print("\nBy inequality (ii), a measurement with reliability r^2 leaves "
          "(1 - r^2) of a genuinely known effect sitting in the unexplained "
          "residual, where it looks like a discovery. These figures bound how "
          "much of this project's atrial residual is its own measurement error "
          "rather than signal.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "per_cell": args.per_cell,
        "p_durations_ms": list(P_DURATIONS_MS),
        "p_amplitudes_mv": list(P_AMPLITUDES_MV),
        "noise_levels_mv": list(NOISE_LEVELS_MV),
        "results": results,
    }, indent=2))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
