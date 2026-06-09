#!/usr/bin/env bash

# Source this before running JAX GPU commands:
#   source scripts/use_gpr_jax_gpu_env.sh

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-gpr-jax-gpu}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Please load Anaconda/Miniconda first." >&2
  return 1 2>/dev/null || exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

JAX_CUDA_LIB_PATHS="$(python "$PROJECT_DIR/scripts/jax_cuda_lib_paths.py")"
if [[ -n "$JAX_CUDA_LIB_PATHS" ]]; then
  export LD_LIBRARY_PATH="$JAX_CUDA_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "Activated $ENV_NAME for JAX GPU"
echo "PYTHONPATH=$PYTHONPATH"
echo "JAX_CUDA_LIB_PATHS=$JAX_CUDA_LIB_PATHS"
