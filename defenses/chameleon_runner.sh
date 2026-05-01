#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# runner_chameleon_defense.sh
# Runs chameleon_defense.py for all 3 datasets
# Budgets : 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90
#
# Output files per dataset (named automatically):
#   kg_sanitized_budget_05pct.tsv
#   kg_sanitized_budget_10pct.tsv
#   ...
#   kg_sanitized_budget_90pct.tsv
# ─────────────────────────────────────────────────────────────────

# ── PATHS — edit these before running ────────────────────────────

# Public graphs
NELL_PUBLIC="path/to/nell/public.tsv"
FB15K_PUBLIC="path/to/fb15k/public.tsv"
HEALTH_PUBLIC="path/to/healthkg/public.tsv"

# Sensitive directories
# Each directory should contain one .tsv file per sensitive relation
# e.g.:  sensitive/concept:worksfor.tsv
#        sensitive/concept:proxyfor.tsv
NELL_SENS_DIR="path/to/nell/sensitive/"
FB15K_SENS_DIR="path/to/fb15k/sensitive/"
HEALTH_SENS_DIR="path/to/healthkg/sensitive/"

# Output base directory
# Sanitized graphs will be saved in OUTDIR/DATASET/
OUTDIR="path/to/results/chameleon_defense"

# Python script
SCRIPT="path/to/chameleon_defense.py"

# ── SHARED CONFIG ────────────────────────────────────────────────

BUDGETS="0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90"
SEED=42

# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTION
# Args: DATASET_NAME  PUBLIC_PATH  SENS_DIR
# ─────────────────────────────────────────────────────────────────

run_dataset() {
    local DATASET_NAME=$1
    local PUBLIC_PATH=$2
    local SENS_DIR=$3
    local DATASET_OUTDIR="${OUTDIR}/${DATASET_NAME}"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  DATASET : ${DATASET_NAME}"
    echo "  Output  : ${DATASET_OUTDIR}"
    echo "════════════════════════════════════════════════════════"

    mkdir -p "${DATASET_OUTDIR}"

    python "${SCRIPT}" \
        --public-path  "${PUBLIC_PATH}" \
        --sens-dir     "${SENS_DIR}" \
        --outdir       "${DATASET_OUTDIR}" \
        --budgets      ${BUDGETS} \
        --seed         "${SEED}"

    if [ $? -ne 0 ]; then
        echo "  [ERROR] Failed: dataset=${DATASET_NAME}"
    else
        echo "  [OK]    dataset=${DATASET_NAME} — sanitized graphs saved to ${DATASET_OUTDIR}"
    fi
}

# ─────────────────────────────────────────────────────────────────
# RUN ALL DATASETS
# ─────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════"
echo "  CHAMELEON DEFENSE RUNNER"
echo "  Budgets : ${BUDGETS}"
echo "  Seed    : ${SEED}"
echo "  Output  : ${OUTDIR}"
echo "════════════════════════════════════════════════════════"

run_dataset "NELL"     "${NELL_PUBLIC}"   "${NELL_SENS_DIR}"
run_dataset "FB15k"    "${FB15K_PUBLIC}"  "${FB15K_SENS_DIR}"
run_dataset "HealthKG" "${HEALTH_PUBLIC}" "${HEALTH_SENS_DIR}"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ALL DONE — Results → ${OUTDIR}"
echo "════════════════════════════════════════════════════════"