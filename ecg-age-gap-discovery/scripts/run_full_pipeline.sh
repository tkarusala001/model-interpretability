#!/usr/bin/env bash
# Every phase of the project, end to end, on synthetic data.
#
# Runs from a clean checkout with no external data: the synthetic cohort is
# generated procedurally, so this needs nothing downloaded. That is the point -
# the entire pipeline is validated against constructed ground truth before any
# real recording is involved.
#
#   bash scripts/run_full_pipeline.sh [output_dir]
#
# Takes roughly 10-20 minutes on a laptop CPU. Set FAST=1 for a ~3 minute
# smoke run with a smaller cohort.
#
# WHAT THIS DOES NOT DO
# ---------------------
# It does not touch PTB-XL. Every number it prints comes from a synthetic
# cohort whose answers this project constructed, which makes it a test of the
# machinery and NOT evidence about real hearts. For real data:
#   bash scripts/download_data.sh
#   python scripts/run_discovery_experiment.py --data ptbxl

set -euo pipefail

RUNS_DIR="${1:-runs/full_pipeline}"
PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then PYTHON="python3"; fi

if [[ "${FAST:-0}" == "1" ]]; then
    N_RECORDINGS=400; EPOCHS=12; PYTEST_ARGS="-q -x -k 'not end_to_end'"
else
    N_RECORDINGS=1500; EPOCHS=35; PYTEST_ARGS="-q"
fi

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

banner "Environment"
"${PYTHON}" -c "import torch, numpy, scipy, sklearn; \
print(f'python  {__import__(\"sys\").version.split()[0]}'); \
print(f'torch   {torch.__version__}'); print(f'numpy   {numpy.__version__}'); \
print(f'scipy   {scipy.__version__}'); print(f'sklearn {sklearn.__version__}')"

banner "Phases 1-10: test suite (ground-truth checks for every component)"
# The suite is the validation. Among other things it verifies that R peaks are
# located to within one sample, that measured intervals track constructed ones,
# that fiducial attribution localises a known injected effect, and that the
# residual decomposition behaves correctly in BOTH directions.
eval "${PYTHON} -m pytest tests/ ${PYTEST_ARGS}"

banner "Phase 7: attribution analysis and age-gap outliers"
"${PYTHON}" scripts/run_attribution_analysis.py \
    --n-recordings "${N_RECORDINGS}" --epochs "${EPOCHS}" \
    --n-outliers 3 --n-attribution 100 --runs-dir "${RUNS_DIR}"

banner "Phases 8-10: decomposition, discovery experiment, figures"
"${PYTHON}" scripts/run_discovery_experiment.py \
    --n-recordings "${N_RECORDINGS}" --epochs "${EPOCHS}" \
    --runs-dir "${RUNS_DIR}"

banner "Done"
cat <<EOF
Artifacts are under ${RUNS_DIR}/.

Every figure and number above comes from SYNTHETIC data with constructed
answers. It demonstrates that the machinery works; it is not a finding about
real ECGs. In particular the synthetic diagnostic labels are linked to the
unexplained channel BY CONSTRUCTION, so a positive Phase 9 result here confirms
the experiment can detect a link - nothing more.

To run on real data:
    bash scripts/download_data.sh
    python scripts/run_discovery_experiment.py --data ptbxl
EOF
