"""Fetching PTB-XL from PhysioNet.

ACCESS TERMS, VERIFIED 2026-07-28
---------------------------------
Checked against https://physionet.org/content/ptb-xl/ rather than assumed, and
two details differ from what this project's build plan originally stated:

- **The licence is Creative Commons Attribution 4.0 (CC BY 4.0)**, not ODC-BY.
  Both are permissive attribution licences, so nothing about how this project
  uses the data changes, but the paper must cite the correct one.
- **Current version is 1.0.3** (published 9 November 2022), not 1.0.1.

Confirmed as expected: access is fully open. There is no credentialing step, no
data use agreement and no CITI training requirement - PhysioNet states that
anyone may access the files subject to the licence. A free PhysioNet account is
*not* needed for this dataset.

Size: 1.7 GB compressed, roughly 3.0 GB on disk. The 100 Hz waveforms alone are
enough for model training; the 500 Hz set is needed for interval measurement
(see :class:`~ecg_discovery.config.DataConfig`).

Attribution required by the licence - cite in any publication:
    Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I.,
    Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available
    electrocardiography dataset. Scientific Data, 7(1), 154.
    https://doi.org/10.1038/s41597-020-0495-6
    and the PhysioNet resource: https://doi.org/10.13026/kfzx-aw45
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

__all__ = ["PTBXL_VERSION", "PTBXL_BASE_URL", "PTBXL_ZIP_URL", "PTBXL_LICENCE",
           "download_ptbxl", "is_downloaded"]

PTBXL_VERSION = "1.0.3"
PTBXL_BASE_URL = f"https://physionet.org/files/ptb-xl/{PTBXL_VERSION}/"
#: Single-archive endpoint, far faster than fetching ~44,000 files individually.
PTBXL_ZIP_URL = f"https://physionet.org/content/ptb-xl/get-zip/{PTBXL_VERSION}/"
PTBXL_LICENCE = "CC BY 4.0"

#: Files that must exist for the dataset to be considered present.
REQUIRED_FILES = ("ptbxl_database.csv", "scp_statements.csv")


def is_downloaded(root: str | Path, sampling_rate_hz: int = 100) -> bool:
    """Whether PTB-XL appears to be present and usable at ``root``."""
    root = Path(root)
    if not all((root / name).is_file() for name in REQUIRED_FILES):
        return False
    records = root / ("records100" if sampling_rate_hz == 100 else "records500")
    return records.is_dir() and any(records.iterdir())


def download_ptbxl(root: str | Path, quiet: bool = False) -> Path:
    """Download PTB-XL into ``root`` using ``wget``.

    Mirrors the command PhysioNet itself documents. ``wget -c`` resumes a
    partial download, so an interrupted transfer can simply be re-run.

    Parameters
    ----------
    root:
        Destination directory. Created if absent.
    quiet:
        Suppress wget progress output.

    Returns
    -------
    pathlib.Path
        The directory the dataset was written to.

    Raises
    ------
    RuntimeError
        If ``wget`` is unavailable or the transfer fails. No partial dataset is
        silently accepted: a truncated download would otherwise surface much
        later as unexplained missing records.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # curl rather than wget: curl ships with macOS and most Linux images, wget
    # frequently does not, and a missing wget produces a silent no-op that looks
    # identical to a successful download until the loader fails much later.
    if shutil.which("curl") is None:
        raise RuntimeError(
            "curl is required to download PTB-XL and was not found. Install it, "
            "or download manually from " + PTBXL_BASE_URL
        )

    archive = root.with_suffix(".zip")
    command = [
        "curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "5",
        "-o", str(archive), PTBXL_ZIP_URL,
    ]
    if quiet:
        command.insert(1, "-s")

    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"curl exited with status {result.returncode}. The download may be "
            "incomplete; re-running resumes it."
        )
    subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(root)], check=True)
    # PhysioNet archives unpack into a versioned directory; flatten it.
    for candidate in root.glob("*/ptbxl_database.csv"):
        for item in candidate.parent.iterdir():
            item.rename(root / item.name)
        break
    if not is_downloaded(root):
        raise RuntimeError(
            f"download finished but {root} does not contain the expected files "
            f"{REQUIRED_FILES}. Check the wget output above."
        )
    return root
