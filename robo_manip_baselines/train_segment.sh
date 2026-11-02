#!/usr/bin/env bash
# Train the policy of one learned segment (doc/segment_data_collection.md).
#
#   ./train_segment.sh TASK SEGMENT ITER
#   ./train_segment.sh CupPickPlace grasp 0
#
# Run from robo_manip_baselines/ (Git Bash on Windows, or bash on Linux).
#
# 1. Resample every demo / rollout directory of the segment to the policy
#    rate (misc/ResampleRmbData.py, 5 Hz = the segment runner's 0.2 s loop)
#    into dataset/resampled_${HZ}hz/ (already resampled episodes are kept).
# 2. Train a diffusion policy on them with --skip 1 into
#    checkpoint/DiffusionPolicy/${TASK}_${SEGMENT}_iterNNN/.
#
# Data used:
#   ITER 0: dataset/${TASK}_${SEGMENT}_SM_iter000*  (SpaceMouse demos; one
#           directory per teleop session, all of them)
#   ITER N: the above, extra demos dataset/${TASK}_${SEGMENT}_SM_iter001..N*,
#           and the segment-runner attempts dataset/${TASK}_${SEGMENT}_rollout_iter001..N
#           with --rollout_data_filter d8 (only the human corrections of
#           successful attempts, weighted), warm-started from iteration N-1
#           (its weights and its normalization), fewer epochs + early stopping.
#
# Validation: a fixed set of the ITER 0 demos (every 10th, listed in
# dataset/resampled_${HZ}hz/${TASK}_${SEGMENT}_val.txt, written on the first
# run), so the val loss is comparable across iterations.
#
# Environment overrides: PYTHON, HZ (5), NUM_EPOCHS (ITER 0: policy default,
# ITER N: 50), PATIENCE (10, ITER N only), BATCH_SIZE, INTERVENTION_WEIGHT
# (2.5), EXTRA_ARGS, DATASET_ROOT (./dataset), CHECKPOINT_ROOT (./checkpoint).
# Every run appends its start, end and exit code to summary.log.

set -u

if [ $# -lt 3 ]; then
    sed -n '2,8p' "$0"
    exit 1
fi
TASK="$1"
SEGMENT="$2"
ITER="$3"
ITER3=$(printf "%03d" "$ITER")
HZ="${HZ:-5}"
cd "$(dirname "$0")"

if [ -z "${PYTHON:-}" ]; then
    if [ -x ../venv_train/Scripts/python ]; then
        PYTHON=../venv_train/Scripts/python
    elif [ -x ../venv_train/bin/python ]; then
        PYTHON=../venv_train/bin/python
    else
        PYTHON=python
    fi
fi

DATASET_ROOT="${DATASET_ROOT:-./dataset}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoint}"
RESAMPLED_ROOT="${DATASET_ROOT}/resampled_${HZ}hz"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/DiffusionPolicy/${TASK}_${SEGMENT}_iter${ITER3}"
SUMMARY_LOG="./summary.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY_LOG"
}

# Resample one raw directory; prints the resampled directory.
resample_dir() {
    local in_dir="$1"
    local out_dir="${RESAMPLED_ROOT}/$(basename "$in_dir")"
    "$PYTHON" ./misc/ResampleRmbData.py --in_dir "$in_dir" --out_dir "$out_dir" \
        --target_hz "$HZ" >&2 || return 1
    echo "$out_dir"
}

collect_dirs() {
    # Raw directories whose name starts with $1 (sorted), excluding resampled copies.
    local prefix="$1"
    local d
    for d in "${DATASET_ROOT}"/${prefix}*; do
        [ -d "$d" ] || continue
        case "$d" in
            "$RESAMPLED_ROOT"*) continue ;;
        esac
        echo "$d"
    done
}

log "START train_segment.sh TASK=$TASK SEGMENT=$SEGMENT ITER=$ITER (HZ=$HZ, PYTHON=$PYTHON)"

DEMO_DIRS=()
while IFS= read -r d; do DEMO_DIRS+=("$d"); done < <(collect_dirs "${TASK}_${SEGMENT}_SM_iter000")
RAW_DIRS=("${DEMO_DIRS[@]}")
for ((i = 1; i <= ITER; i++)); do
    I3=$(printf "%03d" "$i")
    while IFS= read -r d; do RAW_DIRS+=("$d"); done < <(collect_dirs "${TASK}_${SEGMENT}_SM_iter${I3}")
    while IFS= read -r d; do RAW_DIRS+=("$d"); done < <(collect_dirs "${TASK}_${SEGMENT}_rollout_iter${I3}")
done
if [ ${#DEMO_DIRS[@]} -eq 0 ]; then
    log "No demo directories ${DATASET_ROOT}/${TASK}_${SEGMENT}_SM_iter000* found."
    log "END exit=1"
    exit 1
fi

TRAIN_DIRS=()
for d in "${RAW_DIRS[@]}"; do
    out=$(resample_dir "$d") || { log "Resampling $d failed."; log "END exit=1"; exit 1; }
    TRAIN_DIRS+=("$out")
done
log "Training data: ${TRAIN_DIRS[*]}"

# Fixed validation episodes from the initial demos (written once).
VAL_LIST="${RESAMPLED_ROOT}/${TASK}_${SEGMENT}_val.txt"
if [ ! -f "$VAL_LIST" ]; then
    DEMO_RESAMPLED=()
    for d in "${DEMO_DIRS[@]}"; do DEMO_RESAMPLED+=("${RESAMPLED_ROOT}/$(basename "$d")"); done
    find "${DEMO_RESAMPLED[@]}" -name "*.rmb" -type d | sort | awk 'NR % 10 == 1' > "$VAL_LIST"
    log "Wrote the fixed validation list $VAL_LIST ($(wc -l < "$VAL_LIST") episodes)"
fi
VAL_FILES=()
while IFS= read -r f; do [ -n "$f" ] && VAL_FILES+=("$f"); done < "$VAL_LIST"

TRAIN_ARGS=(
    --dataset_dir "${TRAIN_DIRS[@]}"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --relative_eef_action
    --state_keys measured_gripper_joint_pos
    --action_keys command_eef_pose command_gripper_joint_pos
    --camera_names hand
    --skip 1
    --image_size 320 200
    --image_crop_size 288 180
)
if [ ${#VAL_FILES[@]} -gt 0 ]; then TRAIN_ARGS+=(--val_dataset_dir "${VAL_FILES[@]}"); fi
if [ "$ITER" -gt 0 ]; then
    PREV_DIR="${CHECKPOINT_ROOT}/DiffusionPolicy/${TASK}_${SEGMENT}_iter$(printf "%03d" $((ITER - 1)))"
    if [ ! -f "$PREV_DIR/policy_best.ckpt" ]; then
        log "Previous iteration checkpoint $PREV_DIR/policy_best.ckpt not found."
        log "END exit=1"
        exit 1
    fi
    TRAIN_ARGS+=(
        --rollout_data_filter d8
        --intervention_weight "${INTERVENTION_WEIGHT:-2.5}"
        --pretrain_checkpoint "$PREV_DIR/policy_best.ckpt"
        --reuse_stats_from "$PREV_DIR/model_meta_info.pkl"
        --num_epochs "${NUM_EPOCHS:-50}"
        --early_stopping_patience "${PATIENCE:-10}"
    )
    log "Warm start from $PREV_DIR"
elif [ -n "${NUM_EPOCHS:-}" ]; then
    TRAIN_ARGS+=(--num_epochs "$NUM_EPOCHS")
fi
if [ -n "${BATCH_SIZE:-}" ]; then TRAIN_ARGS+=(--batch_size "$BATCH_SIZE"); fi

# shellcheck disable=SC2086
"$PYTHON" ./bin/Train.py DiffusionPolicy "${TRAIN_ARGS[@]}" ${EXTRA_ARGS:-}
STATUS=$?
log "END exit=$STATUS checkpoint=$CHECKPOINT_DIR"
exit $STATUS
