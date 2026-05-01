#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# runner_kanonymity_defense.sh
# Runs defense_kanonymity.py for all 3 datasets x all k values
#
# k values : 5, 10, 15, 20, 25
# Datasets  : NELL, FB15k, HealthKG
#
# Output files per dataset (named automatically):
#   kg_kanon_k5.tsv
#   kg_kanon_k10.tsv
#   ...
#   kg_kanon_k25.tsv
# ─────────────────────────────────────────────────────────────────

# ── PATHS — edit these before running ────────────────────────────

# Public graphs (input)
NELL_PUBLIC="path/to/nell/public.tsv"
FB15K_PUBLIC="path/to/fb15k/public.tsv"
HEALTH_PUBLIC="path/to/healthkg/public.tsv"

# Output base directory
# Sanitized graphs will be saved in OUTDIR/DATASET/
OUTDIR="path/to/results/kanonymity_defense"

# Python script
SCRIPT="path/to/defense_kanonymity.py"

# ── SHARED CONFIG ────────────────────────────────────────────────

K_VALUES=(5 10 15 20 25)
SEED=42

# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTION
# Args: DATASET_NAME  PUBLIC_PATH
# ─────────────────────────────────────────────────────────────────

run_dataset() {
    local DATASET_NAME=$1
    local PUBLIC_PATH=$2
    local DATASET_OUTDIR="${OUTDIR}/${DATASET_NAME}"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  DATASET : ${DATASET_NAME}"
    echo "  Output  : ${DATASET_OUTDIR}"
    echo "════════════════════════════════════════════════════════"

    mkdir -p "${DATASET_OUTDIR}"

    for K in "${K_VALUES[@]}"; do

        OUTPUT_PATH="${DATASET_OUTDIR}/kg_kanon_k${K}.tsv"

        echo ""
        echo "  ── k=${K}"
        echo "  [RUN] dataset=${DATASET_NAME}  k=${K}"

        python "${SCRIPT}" \
            --input  "${PUBLIC_PATH}" \
            --output "${OUTPUT_PATH}" \
            --k      "${K}" \
            --seed   "${SEED}"

        if [ $? -ne 0 ]; then
            echo "  [ERROR] Failed: dataset=${DATASET_NAME}  k=${K}"
        else
            echo "  [OK]    dataset=${DATASET_NAME}  k=${K} → ${OUTPUT_PATH}"
        fi

    done
}

# ─────────────────────────────────────────────────────────────────
# RUN ALL DATASETS
# ─────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════"
echo "  K-ANONYMITY DEFENSE RUNNER"
echo "  k values : ${K_VALUES[*]}"
echo "  Seed     : ${SEED}"
echo "  Output   : ${OUTDIR}"
echo "════════════════════════════════════════════════════════"

run_dataset "NELL"     "${NELL_PUBLIC}"
run_dataset "FB15k"    "${FB15K_PUBLIC}"
run_dataset "HealthKG" "${HEALTH_PUBLIC}"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ALL DONE — Results → ${OUTDIR}"
echo "════════════════════════════════════════════════════════"