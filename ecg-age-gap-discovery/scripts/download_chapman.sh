#!/usr/bin/env bash
# Download the Chapman-Shaoxing-Ningbo 12-lead ECG database, used as the
# external replication cohort.
#
# ACCESS TERMS, VERIFIED 2026-07-30 against
# https://physionet.org/content/ecg-arrhythmia/1.0.0/
#   * Licence:  Creative Commons Attribution 4.0 (CC BY 4.0)
#   * Access:   fully open. No credentialing, data use agreement, CITI training
#               or account required.
#   * Size:     2.5 GB compressed, ~5.1 GB on disk.
#   * Content:  45,152 twelve-lead ECGs, 10 s at 500 Hz, with age and sex.
#
# Attribution required by the licence. Cite in any publication:
#   Zheng J, Zhang J, Danioko S, Yao H, Guo H, Rakovski C. A 12-lead
#   electrocardiogram database for arrhythmia research covering more than
#   10,000 patients. Scientific Data 7, 48 (2020).
#   PhysioNet: doi:10.13026/wgex-er52
#
# Uses curl rather than wget: curl ships with macOS, wget frequently does not,
# and a single ZIP is far faster than recursively fetching 90,000 small files.
#
# Usage:
#   bash scripts/download_chapman.sh [destination]     # default: data/chapman

set -euo pipefail

DEST="${1:-data/chapman}"
ZIP="${DEST%/}.zip"
URL="https://physionet.org/content/ecg-arrhythmia/get-zip/1.0.0/"

if ! command -v curl >/dev/null 2>&1; then
    echo "error: curl is required but not installed." >&2
    exit 1
fi

echo "Downloading Chapman-Shaoxing-Ningbo (~2.5 GB) to ${ZIP}"
echo "Licence: CC BY 4.0 -- attribution required, see header of this script."
mkdir -p "$(dirname "${ZIP}")"

# -C - resumes a partial transfer, so an interrupted download can be re-run.
curl -L -C - --retry 5 --retry-delay 5 -o "${ZIP}" "${URL}"

echo "Extracting to ${DEST} ..."
mkdir -p "${DEST}"
unzip -q -o "${ZIP}" -d "${DEST}"

COUNT=$(find "${DEST}" -name "*.hea" | wc -l | tr -d ' ')
if [[ "${COUNT}" -lt 1000 ]]; then
    echo "error: only ${COUNT} header files found - the download looks incomplete." >&2
    exit 1
fi

echo
echo "Done: ${COUNT} records. Verify with:"
echo "  python -c \"from ecg_discovery.data.chapman_dataset import summarise_chapman; print(summarise_chapman('${DEST}'))\""
echo
echo "The ZIP can be deleted once extraction is confirmed:  rm ${ZIP}"
