#!/usr/bin/env python3
"""Download a subset of the Chapman-Shaoxing-Ningbo ECG database in parallel.

ACCESS TERMS, VERIFIED 2026-07-30 against
https://physionet.org/content/ecg-arrhythmia/1.0.0/
  * Licence: Creative Commons Attribution 4.0 (CC BY 4.0)
  * Access:  fully open - no credentialing, data use agreement, CITI training
             or account required.
  * Content: 45,152 twelve-lead ECGs, 10 s at 500 Hz, with age and sex.

Attribution required by the licence:
    Zheng J, Zhang J, Danioko S, Yao H, Guo H, Rakovski C. A 12-lead
    electrocardiogram database for arrhythmia research covering more than
    10,000 patients. Scientific Data 7, 48 (2020).
    PhysioNet: doi:10.13026/wgex-er52

WHY NOT THE ZIP, AND WHY NOT ALL OF IT
--------------------------------------
PhysioNet's ``get-zip`` endpoint builds the archive on demand and is heavily
throttled - measured at **4.4 kB/s**, an estimated 6.5 days for 2.5 GB. The
static per-file endpoint serves the same data at ~210 kB/s per connection, so
fetching files directly and in parallel is roughly two orders of magnitude
faster.

A subset is downloaded rather than the whole database because this cohort is
used for *external replication*, not for a competitive model: 12,000 recordings
train a perfectly adequate age regressor and download in about ten minutes.

Records are sampled at an even stride across the full listing rather than taken
from the front. The database is organised into directories that group
recordings by acquisition batch, so the first N would over-represent one batch
and could differ systematically from the cohort as a whole.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://physionet.org/files/ecg-arrhythmia/1.0.0"
RECORDS_URL = f"{BASE}/RECORDS"


def fetch(url: str, destination: Path, timeout: float = 60.0) -> bool:
    """Download one file, skipping it if already present and non-empty."""
    if destination.exists() and destination.stat().st_size > 0:
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False
    if not payload:
        return False
    destination.write_bytes(payload)
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/chapman"))
    parser.add_argument("--limit", type=int, default=12000,
                        help="recordings to download (evenly spaced across the database)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    print(f"fetching record listing from {RECORDS_URL} ...")
    try:
        with urllib.request.urlopen(RECORDS_URL, timeout=60) as response:
            listing = [line.strip() for line in response.read().decode().splitlines()
                       if line.strip()]
    except Exception as error:                       # noqa: BLE001
        print(f"error: could not fetch the record listing: {error}", file=sys.stderr)
        return 1

    print(f"  {len(listing):,} records available")

    # Even stride, not the first N - directories group recordings by acquisition
    # batch, so taking the front would over-represent one batch.
    if args.limit and args.limit < len(listing):
        stride = len(listing) / args.limit
        selected = [listing[int(i * stride)] for i in range(args.limit)]
    else:
        selected = listing
    print(f"  downloading {len(selected):,} records "
          f"({2 * len(selected):,} files) with {args.workers} workers")

    jobs: list[tuple[str, Path]] = []
    for relative in selected:
        for suffix in (".hea", ".mat"):
            jobs.append((f"{BASE}/{relative}{suffix}",
                         args.dest / f"{relative}{suffix}"))

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, url, path): url for url, path in jobs}
        for future in as_completed(futures):
            if future.result():
                done += 1
            else:
                failed += 1
            total = done + failed
            if total % 2000 == 0:
                print(f"  {total:,}/{len(jobs):,} files ({failed} failed)")

    print(f"\ndone: {done:,} files, {failed} failed")
    headers = list(args.dest.rglob("*.hea"))
    print(f"{len(headers):,} records present at {args.dest}")
    if failed > len(jobs) * 0.02:
        print("warning: more than 2% of files failed; re-run to retry them",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
