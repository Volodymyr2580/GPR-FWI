#!/usr/bin/env bash
set -euo pipefail

# Optuna search for JAX OverThrust top-20-fixed eps-first-then-sig inversion.
# Objective: model quality, using active-region NMSE and SSIM against the true model.

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
NP="${NP:-3}"
N_TRIALS="${N_TRIALS:-50}"
NUM_EPOCHS_STAGE1="${NUM_EPOCHS_STAGE1:-5000}"
NUM_EPOCHS_STAGE2="${NUM_EPOCHS_STAGE2:-2500}"
STAGE1_CHECKPOINT_PATH="${STAGE1_CHECKPOINT_PATH:-}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-1000}"
DATA_PATH="${DATA_PATH:-data/overthrust/OverThrust.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/overthrust/optuna_jax_model_quality}"
STUDY_NAME="${STUDY_NAME:-overthrust_jax_model_quality}"
MAX_JAX_SHOT_BATCH_SIZE="${MAX_JAX_SHOT_BATCH_SIZE:-5}"
INSTALL_OPTUNA="${INSTALL_OPTUNA:-1}"

export CUDA_VISIBLE_DEVICES
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$INSTALL_OPTUNA" == "1" ]]; then
  "$PYTHON_BIN" - <<'PY' || "$PYTHON_BIN" -m pip install optuna
import optuna
PY
fi

args=(
  --data-path "$DATA_PATH"
  --output-dir "$OUTPUT_DIR"
  --python-executable "$PYTHON_BIN"
  --mpi-processes "$NP"
  --study-name "$STUDY_NAME"
  --n-trials "$N_TRIALS"
  --num-epochs-stage1 "$NUM_EPOCHS_STAGE1"
  --num-epochs-stage2 "$NUM_EPOCHS_STAGE2"
  --snapshot-interval "$SNAPSHOT_INTERVAL"
  --fixed-top-rows "${FIXED_TOP_ROWS:-20}"
  --stage1-data-loss "${STAGE1_DATA_LOSS:-l1}"
  --stage2-data-loss "${STAGE2_DATA_LOSS:-l2}"
  --fdtd-backend "${FDTD_BACKEND:-jax}"
  --device "${DEVICE:-auto}"
  --jax-device "${JAX_DEVICE:-auto}"
  --max-jax-shot-batch-size "$MAX_JAX_SHOT_BATCH_SIZE"
  --objective-region "${OBJECTIVE_REGION:-active}"
  --eps-weight "${EPS_WEIGHT:-1.0}"
  --sig-weight "${SIG_WEIGHT:-1.0}"
  --mse-weight "${MSE_WEIGHT:-1.0}"
  --ssim-weight "${SSIM_WEIGHT:-1.0}"
  --learning-rate-eps "${LEARNING_RATE_EPS:-1e-4}"
  --lr-eps-stage2-min "${LR_EPS_STAGE2_MIN:-1e-6}"
  --lr-eps-stage2-max "${LR_EPS_STAGE2_MAX:-1e-4}"
  --learning-rate-sig-min "${LR_SIG_MIN:-1e-6}"
  --learning-rate-sig-max "${LR_SIG_MAX:-1e-4}"
  --alpha-tv-eps-min "${ALPHA_TV_EPS_MIN:-1e-2}"
  --alpha-tv-eps-max "${ALPHA_TV_EPS_MAX:-5e-2}"
  --alpha-tv-sig-min "${ALPHA_TV_SIG_MIN:-3e-2}"
  --alpha-tv-sig-max "${ALPHA_TV_SIG_MAX:-4e-2}"
  --alpha-l1-data-max "${ALPHA_L1_DATA_MAX:-0.5}"
  --alpha-tikhonov-eps "${ALPHA_TIKHONOV_EPS:-0.0}"
  --alpha-tikhonov-sig "${ALPHA_TIKHONOV_SIG:-0.0}"
)

if [[ -n "$STAGE1_CHECKPOINT_PATH" ]]; then
  args+=(--stage1-checkpoint-path "$STAGE1_CHECKPOINT_PATH")
fi

if [[ "${NO_AUTO_JAX_SHOT_BATCH_SIZE:-0}" == "1" ]]; then
  args+=(--no-auto-jax-shot-batch-size --jax-shot-batch-size "${JAX_SHOT_BATCH_SIZE:-5}")
fi

if [[ "${NO_ILLUMINATION:-0}" == "1" ]]; then
  args+=(--no-illumination)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

"$PYTHON_BIN" scripts/overthrust_optuna_model_quality_search.py "${args[@]}"
