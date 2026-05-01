#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# runner_ablation_attack2.sh
# Runs ablation_attack2.py for all 14 experiments x all datasets
# Datasets  : NELL, FB15k, Health-KG
# Device    : GPU (cuda)
#
# Dataset-specific thresholds (a1h_thr / a1t_thr):
#   NELL     : 0.5 / 0.5
#   FB15k    : 0.0 / 0.0
#   HealthKG : 0.5 / 0.8
# ─────────────────────────────────────────────────────────────────

# ── PATHS — edit these before running ────────────────────────────

# Public graphs
NELL_PUBLIC="path/to/nell/public.tsv"
FB15K_PUBLIC="path/to/fb15k/public.tsv"
HEALTH_PUBLIC="path/to/healthkg/public.tsv"

# Sensitive files directories
NELL_SENS_DIR="path/to/nell/sensitive/"
FB15K_SENS_DIR="path/to/fb15k/sensitive/"
HEALTH_SENS_DIR="path/to/healthkg/sensitive/"

# Attack1 head scores directory (one file per relation: <relation>_scores.tsv)
NELL_A1H_SCORES="path/to/nell/attack1/head/scores/"
FB15K_A1H_SCORES="path/to/fb15k/attack1/head/scores/"
HEALTH_A1H_SCORES="path/to/healthkg/attack1/head/scores/"

# Attack1 tail scores directory (one file per relation: <relation>_scores.tsv)
NELL_A1T_SCORES="path/to/nell/attack1/tail/scores/"
FB15K_A1T_SCORES="path/to/fb15k/attack1/tail/scores/"
HEALTH_A1T_SCORES="path/to/healthkg/attack1/tail/scores/"

# Output base directory
OUTDIR="path/to/results/attack3_featuresstudy.py"

# Python script
SCRIPT="path/to/ablation_attack2.py"

# ── SHARED CONFIG ────────────────────────────────────────────────

DEVICE="cuda"
SEED=42
NEG_PER_POS=10
BATCH_SIZE=512
EPOCHS=100

# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTION
# Args: DATASET_NAME  PUBLIC_PATH  SENS_DIR
#       A1H_SCORES_DIR  A1T_SCORES_DIR
#       A1H_THR  A1T_THR
# ─────────────────────────────────────────────────────────────────

run_dataset() {
    local DATASET_NAME=$1
    local PUBLIC_PATH=$2
    local SENS_DIR=$3
    local A1H_DIR=$4
    local A1T_DIR=$5
    local A1H_THR=$6
    local A1T_THR=$7

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  DATASET   : ${DATASET_NAME}"
    echo "  Threshold : a1h_thr=${A1H_THR}  a1t_thr=${A1T_THR}"
    echo "════════════════════════════════════════════════════════"

    for SENS_PATH in "${SENS_DIR}"/*.tsv; do

        if [ ! -f "$SENS_PATH" ]; then
            echo "  [WARN] No .tsv files found in ${SENS_DIR}"
            continue
        fi

        RELATION=$(basename "${SENS_PATH}" .tsv)

        A1H_SCORES="${A1H_DIR}/${RELATION}_scores.tsv"
        A1T_SCORES="${A1T_DIR}/${RELATION}_scores.tsv"

        if [ ! -f "$A1H_SCORES" ]; then
            echo "  [SKIP] Missing head scores: ${A1H_SCORES}"
            continue
        fi
        if [ ! -f "$A1T_SCORES" ]; then
            echo "  [SKIP] Missing tail scores: ${A1T_SCORES}"
            continue
        fi

        echo ""
        echo "  ── Relation : ${RELATION}"
        echo "     head scores : ${A1H_SCORES}"
        echo "     tail scores : ${A1T_SCORES}"
        echo ""
        echo "  [RUN] dataset=${DATASET_NAME}  relation=${RELATION}  (14 experiments)"

        python "${SCRIPT}" \
            --public-path          "${PUBLIC_PATH}" \
            --sens-path            "${SENS_PATH}" \
            --attack1-head-scores  "${A1H_SCORES}" \
            --attack1-tail-scores  "${A1T_SCORES}" \
            --a1h-thr              "${A1H_THR}" \
            --a1t-thr              "${A1T_THR}" \
            --neg-per-pos          "${NEG_PER_POS}" \
            --outdir               "${OUTDIR}/${DATASET_NAME}/${RELATION}" \
            --device               "${DEVICE}" \
            --seed                 "${SEED}"

        if [ $? -ne 0 ]; then
            echo "  [ERROR] Failed: dataset=${DATASET_NAME}  relation=${RELATION}"
        else
            echo "  [OK]    dataset=${DATASET_NAME}  relation=${RELATION}"
        fi

    done
}

# ─────────────────────────────────────────────────────────────────
# RUN ALL DATASETS
# ─────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════"
echo "  ABLATION ATTACK2 RUNNER"
echo "  14 experiments per relation"
echo "  Device  : ${DEVICE}"
echo "  Output  : ${OUTDIR}"
echo "════════════════════════════════════════════════════════"

# NELL     — a1h_thr=0.5  a1t_thr=0.5
run_dataset "NELL" \
    "${NELL_PUBLIC}" \
    "${NELL_SENS_DIR}" \
    "${NELL_A1H_SCORES}" \
    "${NELL_A1T_SCORES}" \
    0.5 0.5

# FB15k    — a1h_thr=0.0  a1t_thr=0.0
run_dataset "FB15k" \
    "${FB15K_PUBLIC}" \
    "${FB15K_SENS_DIR}" \
    "${FB15K_A1H_SCORES}" \
    "${FB15K_A1T_SCORES}" \
    0.0 0.0

# HealthKG — a1h_thr=0.5  a1t_thr=0.8
run_dataset "HealthKG" \
    "${HEALTH_PUBLIC}" \
    "${HEALTH_SENS_DIR}" \
    "${HEALTH_A1H_SCORES}" \
    "${HEALTH_A1T_SCORES}" \
    0.5 0.8

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ALL DONE — Results → ${OUTDIR}"
echo "════════════════════════════════════════════════════════"