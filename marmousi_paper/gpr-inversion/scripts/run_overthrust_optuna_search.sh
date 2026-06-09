#!/usr/bin/env bash
set -euo pipefail

# One-command Optuna search launcher for a Linux server.
# Usage:
#   STAGE1_CHECKPOINT_PATH=/path/to/stage1_eps_checkpoint.pt bash scripts/run_overthrust_optuna_search.sh
#
# Common overrides:
#   DEVICE=cuda:1 N_TRIALS=50 NUM_EPOCHS_STAGE2=1000 bash scripts/run_overthrust_optuna_search.sh

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MPI_PROCESSES="${MPI_PROCESSES:-30}"
DEVICE="${DEVICE:-cuda:1}"
N_TRIALS="${N_TRIALS:-50}"
NUM_EPOCHS_STAGE2="${NUM_EPOCHS_STAGE2:-1000}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-1000}"
STAGE1_CHECKPOINT_PATH="${STAGE1_CHECKPOINT_PATH:-stage1_eps_checkpoint.pt}"
DATA_PATH="${DATA_PATH:-data/overthrust/OverThrust.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/overthrust/optuna_eps_then_sig}"
STUDY_NAME="${STUDY_NAME:-overthrust_mode2_eps_then_sig}"
INSTALL_OPTUNA="${INSTALL_OPTUNA:-1}"
FDTD_BACKEND="${FDTD_BACKEND:-numpy}"

# Search ranges. Learning rates use log-scale sampling inside Optuna.
LR_EPS_STAGE2_MIN="${LR_EPS_STAGE2_MIN:-1e-6}"
LR_EPS_STAGE2_MAX="${LR_EPS_STAGE2_MAX:-1e-4}"
LR_SIG_MIN="${LR_SIG_MIN:-1e-6}"
LR_SIG_MAX="${LR_SIG_MAX:-1e-4}"
ALPHA_TV_EPS_MIN="${ALPHA_TV_EPS_MIN:-1e-2}"
ALPHA_TV_EPS_MAX="${ALPHA_TV_EPS_MAX:-5e-2}"

# You previously found alpha-tv-sig around 3e-2 to 4e-2 useful.
# Default: search that narrow range. Set SEARCH_ALPHA_TV_SIG=0 to keep ALPHA_TV_SIG fixed.
SEARCH_ALPHA_TV_SIG="${SEARCH_ALPHA_TV_SIG:-1}"
ALPHA_TV_SIG="${ALPHA_TV_SIG:-3.5e-2}"
ALPHA_TV_SIG_MIN="${ALPHA_TV_SIG_MIN:-3e-2}"
ALPHA_TV_SIG_MAX="${ALPHA_TV_SIG_MAX:-4e-2}"

# Default: search the mixed L1 data-loss weight from [0, ALPHA_L1_DATA_MAX].
# Set SEARCH_ALPHA_L1_DATA=0 to keep ALPHA_L1_DATA fixed.
SEARCH_ALPHA_L1_DATA="${SEARCH_ALPHA_L1_DATA:-1}"
ALPHA_L1_DATA="${ALPHA_L1_DATA:-0.0}"
ALPHA_L1_DATA_MAX="${ALPHA_L1_DATA_MAX:-0.5}"

if [[ "$INSTALL_OPTUNA" == "1" ]]; then
  "$PYTHON_BIN" - <<'PY' || "$PYTHON_BIN" -m pip install optuna
import optuna
PY
fi

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

args=(
  --stage1-checkpoint-path "$STAGE1_CHECKPOINT_PATH"
  --data-path "$DATA_PATH"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --mpi-processes "$MPI_PROCESSES"
  --python-executable "$PYTHON_BIN"
  --study-name "$STUDY_NAME"
  --n-trials "$N_TRIALS"
  --num-epochs-stage2 "$NUM_EPOCHS_STAGE2"
  --snapshot-interval "$SNAPSHOT_INTERVAL"
  --fdtd-backend "$FDTD_BACKEND"
  --lr-eps-stage2-min "$LR_EPS_STAGE2_MIN"
  --lr-eps-stage2-max "$LR_EPS_STAGE2_MAX"
  --learning-rate-sig-min "$LR_SIG_MIN"
  --learning-rate-sig-max "$LR_SIG_MAX"
  --alpha-tv-eps-min "$ALPHA_TV_EPS_MIN"
  --alpha-tv-eps-max "$ALPHA_TV_EPS_MAX"
  --alpha-tv-sig "$ALPHA_TV_SIG"
  --alpha-tv-sig-min "$ALPHA_TV_SIG_MIN"
  --alpha-tv-sig-max "$ALPHA_TV_SIG_MAX"
  --alpha-l1-data "$ALPHA_L1_DATA"
  --alpha-l1-data-max "$ALPHA_L1_DATA_MAX"
)

if [[ "$SEARCH_ALPHA_TV_SIG" == "1" ]]; then
  args+=(--search-alpha-tv-sig)
fi

if [[ "$SEARCH_ALPHA_L1_DATA" == "1" ]]; then
  args+=(--search-alpha-l1-data)
fi

if [[ "${NO_ILLUMINATION:-0}" == "1" ]]; then
  args+=(--no-illumination)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

"$PYTHON_BIN" scripts/overthrust_optuna_search.py "${args[@]}"
