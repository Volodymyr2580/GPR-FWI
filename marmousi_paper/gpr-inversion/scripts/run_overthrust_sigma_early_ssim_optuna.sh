#!/usr/bin/env bash
set -euo pipefail

# Search sigma-only hyperparameters with fixed epsilon-side settings.
# Objective: maximize the best sigma SSIM found within the first 1000 stage2 epochs.

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
NP="${NP:-3}"
N_TRIALS="${N_TRIALS:-50}"
NUM_EPOCHS_STAGE1="${NUM_EPOCHS_STAGE1:-5000}"
NUM_EPOCHS_STAGE2="${NUM_EPOCHS_STAGE2:-1000}"
HISTORY_MAX_EPOCH="${HISTORY_MAX_EPOCH:-1000}"
STAGE1_CHECKPOINT_PATH="${STAGE1_CHECKPOINT_PATH:-outputs/overthrust/jax_top20_stage1_eps_checkpoint/stage1_eps_checkpoint.pt}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-100}"
DATA_PATH="${DATA_PATH:-data/overthrust/OverThrust.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/overthrust/optuna_jax_sigma_early_ssim}"
STUDY_NAME="${STUDY_NAME:-overthrust_jax_sigma_early_ssim}"
MAX_JAX_SHOT_BATCH_SIZE="${MAX_JAX_SHOT_BATCH_SIZE:-5}"
INSTALL_OPTUNA="${INSTALL_OPTUNA:-1}"

if [[ ! -f "$STAGE1_CHECKPOINT_PATH" ]]; then
  echo "Stage1 checkpoint not found: $STAGE1_CHECKPOINT_PATH" >&2
  echo "Run scripts/run_overthrust_jax_stage1_eps_checkpoint.sh first, or set STAGE1_CHECKPOINT_PATH." >&2
  exit 2
fi

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
  --stage1-checkpoint-path "$STAGE1_CHECKPOINT_PATH"
  --snapshot-interval "$SNAPSHOT_INTERVAL"
  --fixed-top-rows "${FIXED_TOP_ROWS:-20}"
  --stage1-data-loss "${STAGE1_DATA_LOSS:-l1}"
  --stage2-data-loss "${STAGE2_DATA_LOSS:-l2}"
  --fdtd-backend "${FDTD_BACKEND:-jax}"
  --device "${DEVICE:-auto}"
  --jax-device "${JAX_DEVICE:-auto}"
  --max-jax-shot-batch-size "$MAX_JAX_SHOT_BATCH_SIZE"
  --objective-region "${OBJECTIVE_REGION:-full}"
  --objective-source best-history
  --history-max-epoch "$HISTORY_MAX_EPOCH"
  --eps-weight 0.0
  --sig-weight 1.0
  --mse-weight 0.0
  --ssim-weight 1.0
  --learning-rate-eps "${LEARNING_RATE_EPS:-1e-4}"
  --lr-eps-stage2 "${LR_EPS_STAGE2:-2e-6}"
  --learning-rate-sig-min "${LR_SIG_MIN:-1e-6}"
  --learning-rate-sig-max "${LR_SIG_MAX:-1e-4}"
  --alpha-tv-eps "${ALPHA_TV_EPS:-0.04}"
  --alpha-tv-sig-min "${ALPHA_TV_SIG_MIN:-5e-3}"
  --alpha-tv-sig-max "${ALPHA_TV_SIG_MAX:-8e-2}"
  --alpha-l1-data "${ALPHA_L1_DATA:-0.2}"
  --alpha-tikhonov-eps "${ALPHA_TIKHONOV_EPS:-0.0}"
  --alpha-tikhonov-sig "${ALPHA_TIKHONOV_SIG:-0.0}"
)

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
