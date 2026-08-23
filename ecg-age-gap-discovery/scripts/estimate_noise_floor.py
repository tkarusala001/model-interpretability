#!/usr/bin/env python3
"""How much of the age gap is training noise rather than signal.

The unexplained share of an age gap is reported as the ceiling on a discovery
claim, but part of it is the model's own estimation error and was never
eligible to be a discovery: retrain with a different seed and it changes. This
script estimates that share from age gaps of independently seeded models,
correlating them per recording.

No model is retrained. Each run directory already contains per-recording age
gaps, so only the statistic is computed.

Examples
--------
    python scripts/estimate_noise_floor.py \
        runs/ptbxl_official/<seed0> runs/ptbxl_official/<seed1> runs/ptbxl_official/<seed2>

    # subtract the floor from a reported unexplained share
    python scripts/estimate_noise_floor.py runs/a runs/b --unexplained 0.650
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_discovery.validation.noise_floor import estimate_noise_floor  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dirs", type=Path, nargs="+",
                        help="two or more run directories from independently seeded models")
    parser.add_argument("--unexplained", type=float, default=None,
                        help="reported unexplained share to correct, e.g. 0.650")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None,
                        help="write the result as JSON here")
    return parser.parse_args(argv)


def _load(run_dir: Path) -> dict[int, float]:
    """Per-recording age gaps from a run, keyed by recording index.

    Keyed rather than positional: two runs can hold the same test recordings in
    a different order, and correlating them positionally would silently pair
    unrelated patients and report the result as a noise floor.
    """
    for candidate in (run_dir / "artifacts" / "predictions.json",
                      run_dir / "predictions.json"):
        if candidate.exists():
            test = json.loads(candidate.read_text())["test"]
            return {int(i): float(g) for i, g in zip(test["index"], test["age_gap"])}
    raise SystemExit(f"no predictions.json under {run_dir}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if len(args.run_dirs) < 2:
        raise SystemExit("need at least two run directories")

    per_run = {d.name or str(d): _load(d) for d in args.run_dirs}

    shared = set.intersection(*(set(v) for v in per_run.values()))
    if len(shared) < 3:
        raise SystemExit(
            f"only {len(shared)} recordings are common to all runs; the runs must "
            "share a test split for this comparison to mean anything"
        )
    order = sorted(shared)
    for name, mapping in per_run.items():
        dropped = len(mapping) - len(shared)
        if dropped:
            print(f"  {name}: {dropped} recordings not shared by every run, dropped")

    gaps = {name: np.array([mapping[i] for i in order])
            for name, mapping in per_run.items()}

    result = estimate_noise_floor(
        gaps, bootstrap_iterations=args.bootstrap_iterations
    )
    print()
    print(result.summary_text())

    if args.unexplained is not None:
        corrected = result.corrected_unexplained(args.unexplained)
        print()
        print(f"Reported unexplained share : {args.unexplained:.1%}")
        print(f"Less the noise floor       : {result.nu:.1%}")
        print(f"Eligible to be a discovery : {corrected:.1%}")
        print("This is the quantity the bound N <= U - nu actually concerns.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "runs": [str(d) for d in args.run_dirs],
            "n_recordings": result.n_recordings,
            "n_seeds": result.n_seeds,
            "mean_correlation": result.mean_correlation,
            "nu": result.nu,
            "nu_ci": list(result.nu_ci),
            "pairwise": {f"{a}|{b}": r for (a, b), r in result.pairwise.items()},
            "unexplained_input": args.unexplained,
            "corrected_unexplained": (
                result.corrected_unexplained(args.unexplained)
                if args.unexplained is not None else None
            ),
            "summary_text": result.summary_text(),
        }, indent=2))
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
