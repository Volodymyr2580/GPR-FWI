#!/usr/bin/env bash
set -euo pipefail

# Run the fixed-parameter epsilon-only stage once and save a reusable checkpoint.

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
NP="${NP:-3}"
DATA_PATH="${DATA_PATH:-data/overthrust/OverThrust.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/overthrust/jax_top20_stage1_eps_checkpoint}"
STAGE1_CHECKPOINT_PATH="${STAGE1_CHECKPOINT_PATH:-${OUTPUT_DIR}/stage1_eps_checkpoint.pt}"
NUM_EPOCHS_STAGE1="${NUM_EPOCHS_STAGE1:-5000}"
MAX_JAX_SHOT_BATCH_SIZE="${MAX_JAX_SHOT_BATCH_SIZE:-5}"

export CUDA_VISIBLE_DEVICES
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_DIR"

mpirun -np "$NP" "$PYTHON_BIN" -m gpr_inversion.experiments.overthrust.mode2_eps_then_sig.twopara_epsfirst \
  --fdtd-backend jax \
  --illumination \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --stage1-checkpoint-path "$STAGE1_CHECKPOINT_PATH" \
  --fixed-top-rows "${FIXED_TOP_ROWS:-20}" \
  --num-epochs-stage1 "$NUM_EPOCHS_STAGE1" \
  --num-epochs-stage2 0 \
  --stage1-data-loss "${STAGE1_DATA_LOSS:-l1}" \
  --stage2-data-loss "${STAGE2_DATA_LOSS:-l2}" \
  --device "${DEVICE:-auto}" \
  --jax-device "${JAX_DEVICE:-auto}" \
  --auto-jax-shot-batch-size \
  --max-jax-shot-batch-size "$MAX_JAX_SHOT_BATCH_SIZE" \
  --snapshot-interval "${SNAPSHOT_INTERVAL:-1000}" \
  --learning-rate-eps "${LEARNING_RATE_EPS:-1e-4}" \
  --lr-eps-stage2 "${LR_EPS_STAGE2:-1e-6}" \
  --learning-rate-sig "${LEARNING_RATE_SIG:-1e-5}" \
  --alpha-tv-eps "${ALPHA_TV_EPS:-0}" \
  --alpha-tv-sig "${ALPHA_TV_SIG:-5e-4}" \
  --alpha-l1-data "${ALPHA_L1_DATA:-0.0}" \
  --alpha-tikhonov-eps "${ALPHA_TIKHONOV_EPS:-0.0}" \
  --alpha-tikhonov-sig "${ALPHA_TIKHONOV_SIG:-0.0}"

echo "Stage1 checkpoint: ${STAGE1_CHECKPOINT_PATH}"
