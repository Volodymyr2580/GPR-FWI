#!/usr/bin/env bash
set -euo pipefail

# Run the JAX GPU validation sequence and save JSON outputs.
# Use after activating a JAX-GPU-capable environment:
#   conda activate gpr-jax-cu121
#   bash scripts/run_jax_gpu_validation.sh

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/jax_fdtd_lab/server_validation}"
RUN_FULLISH="${RUN_FULLISH:-0}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

JAX_CUDA_LIB_PATHS="$(python "$PROJECT_DIR/scripts/jax_cuda_lib_paths.py")"
if [[ -n "$JAX_CUDA_LIB_PATHS" ]]; then
  export LD_LIBRARY_PATH="$JAX_CUDA_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "Writing validation outputs to $OUTPUT_DIR"
echo "JAX_CUDA_LIB_PATHS=$JAX_CUDA_LIB_PATHS"

python -B -m jax_fdtd_lab.run_sanity_checks --steps 8 \
  | tee "$OUTPUT_DIR/sanity_steps8.json"

python -B -m jax_fdtd_lab.benchmark \
  --case tiny \
  --steps 80 \
  --repeats 1 \
  | tee "$OUTPUT_DIR/benchmark_tiny_steps80.json"

python -B -m jax_fdtd_lab.benchmark \
  --case smooth \
  --xl 40 \
  --zl 80 \
  --steps 120 \
  --sources 4 \
  --receivers 16 \
  --repeats 1 \
  | tee "$OUTPUT_DIR/benchmark_smooth_40x80_steps120.json"

if [[ "$RUN_FULLISH" == "1" ]]; then
  python -B -m jax_fdtd_lab.benchmark \
    --case smooth \
    --xl 100 \
    --zl 200 \
    --steps 1000 \
    --sources 40 \
    --receivers 100 \
    --repeats 1 \
    --shot-batch-size 5 \
    --skip-wavefield \
    | tee "$OUTPUT_DIR/benchmark_smooth_100x200_steps1000_batched.json"
fi

echo "JAX GPU validation complete."
