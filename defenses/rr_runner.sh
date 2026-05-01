#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# runner_randomized_response.sh
# Runs defense_randomized_response.py for all 3 datasets x all epsilon values
#
# Epsilon values : 0.1, 0.5, 1.0, 2.0, 5.0
# Datasets       : NELL, FB15k, HealthKG
#
# Epsilon guide:
#   0.1  → very strong privacy, low utility
#   0.5  → strong privacy
#   1.0  → balanced (recommended)
#   2.0  → moderate privacy
#   5.0  → weak privacy, high utility
#
# Output files per dataset:
#   kg_rr_eps0.1.tsv
#   kg_rr_eps0.5.tsv
#   kg_rr_eps1.0.tsv
#   kg_rr_eps2.0.tsv
#   kg_rr_eps5.0.tsv
# ─────────────────────────────────────────────────────────────────

# ── PATHS — edit these before running ────────────────────────────

# Public graphs (input)
NELL_PUBLIC="path/to/nell/public.tsv"
FB15K_PUBLIC="path/to/fb15k/public.tsv"
HEALTH_PUBLIC="path/to/healthkg/public.tsv"

# Output base directory
# Sanitized graphs will be saved in OUTDIR/DATASET/
OUTDIR="path/to/results/randomized_response_defense"

# Python script
SCRIPT="path/to/defense_randomized_response.py"

# ── SHARED CONFIG ────────────────────────────────────────────────

EPSILON_VALUES=(0.1 0.5 1.0 2.0 3.0 4.0 5.0)
SEED=42

# For large graphs set this below 1.0 to sample non-edges
# (e.g. 0.01 for very large graphs like FB15k)
# Use 1.0 to iterate all non-edges (safe for small graphs)
NELL_SAMPLE_NONEDGES=1.0
FB15K_SAMPLE_NONEDGES=0.01
HEALTH_SAMPLE_NONEDGES=1.0

# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTION
# Args: DATASET_NAME  PUBLIC_PATH  SAMPLE_NONEDGES
# ─────────────────────────────────────────────────────────────────

run_dataset() {
    local DATASET_NAME=$1
    local PUBLIC_PATH=$2
    local SAMPLE_NONEDGES=$3
    local DATASET_OUTDIR="${OUTDIR}/${DATASET_NAME}"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  DATASET          : ${DATASET_NAME}"
    echo "  sample_nonedges  : ${SAMPLE_NONEDGES}"
    echo "  Output           : ${DATASET_OUTDIR}"
    echo "════════════════════════════════════════════════════════"

    mkdir -p "${DATASET_OUTDIR}"

    for EPS in "${EPSILON_VALUES[@]}"; do

        OUTPUT_PATH="${DATASET_OUTDIR}/kg_rr_eps${EPS}.tsv"

        echo ""
        echo "  ── epsilon=${EPS}"
        echo "  [RUN] dataset=${DATASET_NAME}  epsilon=${EPS}"

        python "${SCRIPT}" \
            --input             "${PUBLIC_PATH}" \
            --output            "${OUTPUT_PATH}" \
            --epsilon           "${EPS}" \
            --seed              "${SEED}" \
            --sample_nonedges   "${SAMPLE_NONEDGES}"

        if [ $? -ne 0 ]; then
            echo "  [ERROR] Failed: dataset=${DATASET_NAME}  epsilon=${EPS}"
        else
            echo "  [OK]    dataset=${DATASET_NAME}  epsilon=${EPS} → ${OUTPUT_PATH}"
        fi

    done
}

# ─────────────────────────────────────────────────────────────────
# RUN ALL DATASETS
# ─────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════"
echo "  RANDOMIZED RESPONSE DEFENSE RUNNER"
echo "  Epsilon values : ${EPSILON_VALUES[*]}"
echo "  Seed           : ${SEED}"
echo "  Output         : ${OUTDIR}"
echo "════════════════════════════════════════════════════════"

run_dataset "NELL"     "${NELL_PUBLIC}"   "${NELL_SAMPLE_NONEDGES}"
run_dataset "FB15k"    "${FB15K_PUBLIC}"  "${FB15K_SAMPLE_NONEDGES}"
run_dataset "HealthKG" "${HEALTH_PUBLIC}" "${HEALTH_SAMPLE_NONEDGES}"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ALL DONE — Results → ${OUTDIR}"
echo "════════════════════════════════════════════════════════"