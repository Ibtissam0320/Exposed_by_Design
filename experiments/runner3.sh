#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# runner_ablation_attack3.sh
# Runs ablation_attack3.py for all feature combinations x all datasets
#
# Feature combinations:
#   1. ni_head+ni_tail
#   2. ni_head+ni_tail+Ii
#   3. ni_head+ni_tail+Ei
#   4. Ii_head+Ii_tail
#   5. all  (ni_head+ni_tail+Ii+Ei)
#
# Datasets : NELL, FB15k, Health-KG
# knn_k    : 120
# Device   : GPU (cuda)
# ─────────────────────────────────────────────────────────────────

# ── PATHS — edit these before running ────────────────────────────

# Public graphs
NELL_PUBLIC="path/to/nell/public.tsv"
FB15K_PUBLIC="path/to/fb15k/public.tsv"
HEALTH_PUBLIC="path/to/healthkg/public.tsv"

# Sensitive directories
# These directories should contain one .tsv file per sensitive relation
# e.g.:  sensitive/concept:athleteplaysforteam.tsv
#        sensitive/concept:worksfor.tsv
#        sensitive/concept:proxyfor.tsv
# The runner will automatically build the --sensitive_files argument
# by listing all .tsv files found in each directory.
NELL_SENS_DIR="path/to/nell/sensitive/"
FB15K_SENS_DIR="path/to/fb15k/sensitive/"
HEALTH_SENS_DIR="path/to/healthkg/sensitive/"

# Output base directory
OUTDIR="path/to/results/ablation_attack3"

# Python script
SCRIPT="path/to/attack3_featuresstudy.py"

# ── SHARED CONFIG ────────────────────────────────────────────────

KNN_K=120
SEED=42
SEED_FRAC=0.15
MAX_PRED_PER_HEAD=1
VOTE_THRESHOLD=0.0
HITS_K=10
MAX_LAYER=2

FEATURE_COMBINATIONS=(
    "ni_head+ni_tail"
    "ni_head+ni_tail+Ii"
    "ni_head+ni_tail+Ei"
    "Ii_head+Ii_tail"
    "all"
)

# ─────────────────────────────────────────────────────────────────
# HELPER: build comma-separated list of .tsv filenames from a dir
# ─────────────────────────────────────────────────────────────────

build_sens_files() {
    local SENS_DIR=$1
    local FILES=""

    for F in "${SENS_DIR}"/*.tsv; do
        if [ ! -f "$F" ]; then
            continue
        fi
        FNAME=$(basename "$F")
        if [ -z "$FILES" ]; then
            FILES="${FNAME}"
        else
            FILES="${FILES},${FNAME}"
        fi
    done

    echo "${FILES}"
}

# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTION
# Args: DATASET_NAME  PUBLIC_PATH  SENS_DIR
# ─────────────────────────────────────────────────────────────────

run_dataset() {
    local DATASET_NAME=$1
    local PUBLIC_PATH=$2
    local SENS_DIR=$3

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  DATASET : ${DATASET_NAME}"
    echo "════════════════════════════════════════════════════════"

    # Auto-build sensitive files list from directory
    SENS_FILES=$(build_sens_files "${SENS_DIR}")

    if [ -z "$SENS_FILES" ]; then
        echo "  [WARN] No .tsv files found in ${SENS_DIR} — skipping dataset"
        return
    fi

    echo "  Sensitive files found:"
    IFS=',' read -ra FILES_ARR <<< "$SENS_FILES"
    for F in "${FILES_ARR[@]}"; do
        echo "    - ${F}"
    done

    for FC in "${FEATURE_COMBINATIONS[@]}"; do

        # Safe name for output directory (replace + with _)
        FC_SAFE="${FC//+/_}"

        echo ""
        echo "  ── Feature combination : ${FC}"
        echo "  [RUN] dataset=${DATASET_NAME}  features=${FC}  L=${MAX_LAYER}  k=${KNN_K}"

        python "${SCRIPT}" \
            --public_tsv           "${PUBLIC_PATH}" \
            --sensitive_dir        "${SENS_DIR}" \
            --sensitive_files      "${SENS_FILES}" \
            --feature-combination  "${FC}" \
            --max-layer            "${MAX_LAYER}" \
            --knn_k                "${KNN_K}" \
            --seed_frac            "${SEED_FRAC}" \
            --seed                 "${SEED}" \
            --max_pred_per_head    "${MAX_PRED_PER_HEAD}" \
            --vote_threshold       "${VOTE_THRESHOLD}" \
            --hits_k               "${HITS_K}" \
            --outdir               "${OUTDIR}/${DATASET_NAME}/${FC_SAFE}"

        if [ $? -ne 0 ]; then
            echo "  [ERROR] Failed: dataset=${DATASET_NAME}  features=${FC}"
        else
            echo "  [OK]    dataset=${DATASET_NAME}  features=${FC}"
        fi

    done
}

# ─────────────────────────────────────────────────────────────────
# RUN ALL DATASETS
# ─────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════"
echo "  ABLATION ATTACK3 RUNNER"
echo "  Feature combinations : ${FEATURE_COMBINATIONS[*]}"
echo "  Max layer            : ${MAX_LAYER}"
echo "  knn_k                : ${KNN_K}"
echo "  Output               : ${OUTDIR}"
echo "════════════════════════════════════════════════════════"

run_dataset "NELL"     "${NELL_PUBLIC}"   "${NELL_SENS_DIR}"
run_dataset "FB15k"    "${FB15K_PUBLIC}"  "${FB15K_SENS_DIR}"
run_dataset "HealthKG" "${HEALTH_PUBLIC}" "${HEALTH_SENS_DIR}"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ALL DONE — Results → ${OUTDIR}"
echo "════════════════════════════════════════════════════════"