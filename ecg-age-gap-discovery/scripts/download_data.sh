#!/usr/bin/env bash
# Download PTB-XL from PhysioNet.
#
# ACCESS TERMS, VERIFIED 2026-07-28 against https://physionet.org/content/ptb-xl/
#   * Licence:  Creative Commons Attribution 4.0 (CC BY 4.0)
#               -- NOT ODC-BY, as this project's build plan originally stated.
#   * Version:  1.0.3, published 2022-11-09.
#   * Access:   fully open. No credentialing, no data use agreement, no CITI
#               training, and no PhysioNet account required.
#   * Size:     1.7 GB compressed, ~3.0 GB on disk.
#
# The licence requires attribution. Cite in any publication:
#   Wagner P, Strodthoff N, Bousseljot R-D, Kreiseler D, Lunze FI, Samek W,
#   Schaeffter T. PTB-XL, a large publicly available electrocardiography
#   dataset. Scientific Data 7, 154 (2020). doi:10.1038/s41597-020-0495-6
#   PhysioNet resource: doi:10.13026/kfzx-aw45
#
# Usage:
#   bash scripts/download_data.sh [destination]     # default: data/ptbxl

set -euo pipefail

DEST="${1:-data/ptbxl}"
VERSION="1.0.3"
URL="https://physionet.org/files/ptb-xl/${VERSION}/"

if ! command -v wget >/dev/null 2>&1; then
    echo "error: wget is required but not installed." >&2
    echo "       macOS: brew install wget" >&2
    exit 1
fi

echo "Downloading PTB-XL ${VERSION} (~1.7 GB) into ${DEST}"
echo "Licence: CC BY 4.0 -- attribution required, see header of this script."
mkdir -p "${DEST}"

# -c resumes a partial transfer, so re-running after an interruption is safe.
wget -r -N -c -np -nH --cut-dirs=3 -P "${DEST}" "${URL}"

if [[ -f "${DEST}/ptbxl_database.csv" && -f "${DEST}/scp_statements.csv" ]]; then
    echo
    echo "Done. Verify with:"
    echo "  python -c \"from ecg_discovery.data.ptbxl_dataset import summarise_ptbxl; print(summarise_ptbxl('${DEST}'))\""
else
    echo "error: download appears incomplete - expected metadata files are missing." >&2
    exit 1
fi
