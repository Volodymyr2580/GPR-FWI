#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU MPI JAX OverThrust top-50 fixed sweep.
# Each MPI rank selects a CUDA device automatically by rank modulo visible GPU count.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

NP="${NP:-4}"
DATA_PATH="${DATA_PATH:-data/overthrust/OverThrust.npy}"
ROOT_OUTPUT_DIR="${ROOT_OUTPUT_DIR:-outputs/overthrust/jax_top50_multigpu_sweep}"
MAX_JAX_SHOT_BATCH_SIZE="${MAX_JAX_SHOT_BATCH_SIZE:-5}"
LOG_DIR="${ROOT_OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --fdtd-backend jax
  --illumination
  --data-path "${DATA_PATH}"
  --fixed-top-rows 50
  --num-epochs-stage1 5000
  --num-epochs-stage2 8000
  --stage1-data-loss l1
  --stage2-data-loss l2
  --device auto
  --jax-device auto
  --auto-jax-shot-batch-size
  --max-jax-shot-batch-size "${MAX_JAX_SHOT_BATCH_SIZE}"
  --snapshot-interval 1000
)

run_case() {
  local name="$1"
  shift
  local output_dir="${ROOT_OUTPUT_DIR}/${name}"
  local log_file="${LOG_DIR}/${name}.log"
  echo "[$(date '+%F %T')] Starting ${name}"
  echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "  NP=${NP}"
  echo "  MAX_JAX_SHOT_BATCH_SIZE=${MAX_JAX_SHOT_BATCH_SIZE}"
  echo "  output_dir=${output_dir}"
  mpirun -np "${NP}" python -m gpr_inversion.experiments.overthrust.mode2_eps_then_sig.twopara_epsfirst "${COMMON_ARGS[@]}" "$@" --output-dir "${output_dir}" 2>&1 | tee "${log_file}"
  echo "[$(date '+%F %T')] Finished ${name}"
}

run_case base --learning-rate-eps 1e-4 --lr-eps-stage2 1e-6 --learning-rate-sig 1e-5 --alpha-tv-eps 0 --alpha-tv-sig 5e-4
run_case epslr5e5 --learning-rate-eps 5e-5 --lr-eps-stage2 5e-7 --learning-rate-sig 1e-5 --alpha-tv-eps 0 --alpha-tv-sig 5e-4
run_case epslr2e4 --learning-rate-eps 2e-4 --lr-eps-stage2 2e-6 --learning-rate-sig 1e-5 --alpha-tv-eps 0 --alpha-tv-sig 5e-4
run_case siglr3e5 --learning-rate-eps 1e-4 --lr-eps-stage2 1e-6 --learning-rate-sig 3e-5 --alpha-tv-eps 0 --alpha-tv-sig 5e-4
run_case tvsig1e4 --learning-rate-eps 1e-4 --lr-eps-stage2 1e-6 --learning-rate-sig 1e-5 --alpha-tv-eps 0 --alpha-tv-sig 1e-4
run_case tvsig1e3 --learning-rate-eps 1e-4 --lr-eps-stage2 1e-6 --learning-rate-sig 1e-5 --alpha-tv-eps 0 --alpha-tv-sig 1e-3

echo "All multi-GPU MPI sweep cases finished."
