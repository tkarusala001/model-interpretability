#!/usr/bin/env bash
# Download PTB-XL from PhysioNet.
#
# ACCESS TERMS, VERIFIED 2026-07-28 against https://physionet.org/content/ptb-xl/
#   * Licence:  Creative Commons Attribution 4.0 (CC BY 4.0)
#               -- NOT ODC-BY, as this project's build plan originally stated.
#   * Version:  1.0.3, published 2022-11-09.
#   * Access:   fully open. No credentialing, no data use agreement, no CITI
#               training, and no PhysioNet account required.
#   * Size:     ~1.7 GB compressed, ~3.0 GB on disk.
#
# The licence requires attribution. Cite in any publication:
#   Wagner P, Strodthoff N, Bousseljot R-D, Kreiseler D, Lunze FI, Samek W,
#   Schaeffter T. PTB-XL, a large publicly available electrocardiography
#   dataset. Scientific Data 7, 154 (2020). doi:10.1038/s41597-020-0495-6
#   PhysioNet resource: doi:10.13026/kfzx-aw45
#
# Uses curl rather than wget. curl ships with macOS and most Linux images;
# wget frequently does not, and a silent failure here looks identical to a
# successful run until the loader reports a missing directory much later.
# A single ZIP is also far faster than recursively fetching ~44,000 files.
#
# Usage:
#   bash scripts/download_data.sh [destination]     # default: data/ptbxl

set -euo pipefail

DEST="${1:-data/ptbxl}"
ZIP="${DEST%/}.zip"
URL="https://physionet.org/content/ptb-xl/get-zip/1.0.3/"

if ! command -v curl >/dev/null 2>&1; then
    echo "error: curl is required but not installed." >&2
    exit 1
fi
if ! command -v unzip >/dev/null 2>&1; then
    echo "error: unzip is required but not installed." >&2
    exit 1
fi

if [[ -f "${DEST}/ptbxl_database.csv" ]]; then
    echo "PTB-XL already present at ${DEST}; nothing to do."
    exit 0
fi

echo "Downloading PTB-XL 1.0.3 (~1.7 GB) to ${ZIP}"
echo "Licence: CC BY 4.0 -- attribution required, see header of this script."
mkdir -p "$(dirname "${ZIP}")"

# -C - resumes a partial transfer, so an interrupted download can be re-run.
curl -L -C - --retry 5 --retry-delay 5 -o "${ZIP}" "${URL}"

echo "Extracting to ${DEST} ..."
mkdir -p "${DEST}"
unzip -q -o "${ZIP}" -d "${DEST}"

# PhysioNet ZIPs unpack into a versioned top-level directory; flatten it so the
# loader finds ptbxl_database.csv where it expects.
NESTED=$(find "${DEST}" -maxdepth 2 -name "ptbxl_database.csv" -print -quit)
if [[ -n "${NESTED}" && "$(dirname "${NESTED}")" != "${DEST}" ]]; then
    echo "flattening $(dirname "${NESTED}") -> ${DEST}"
    mv "$(dirname "${NESTED}")"/* "${DEST}/"
fi

if [[ -f "${DEST}/ptbxl_database.csv" && -f "${DEST}/scp_statements.csv" ]]; then
    echo
    echo "Done. Verify with:"
    echo "  python -c \"from ecg_discovery.data.ptbxl_dataset import summarise_ptbxl; print(summarise_ptbxl('${DEST}'))\""
    echo
    echo "The ZIP can be deleted once verified:  rm ${ZIP}"
else
    echo "error: download appears incomplete - expected metadata files are missing." >&2
    exit 1
fi
