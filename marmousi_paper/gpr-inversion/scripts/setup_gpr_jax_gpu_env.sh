#!/usr/bin/env bash
set -euo pipefail

# Build a clean GPU-JAX environment for the JAX FDTD prototype.
# This environment intentionally does not install PyTorch. Keeping PyTorch and
# JAX GPU in separate environments avoids CUDA/cuDNN/NCCL wheel conflicts.
#
# Usage:
#   bash scripts/setup_gpr_jax_gpu_env.sh

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-gpr-jax-gpu}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/environment-gpr-jax-gpu.yml}"

cd "$PROJECT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Please load Anaconda/Miniconda first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Environment $ENV_NAME already exists."
  echo "To keep this setup clean, create a new env name or remove it manually."
  echo "Example: ENV_NAME=gpr-jax-gpu-v2 bash scripts/setup_gpr_jax_gpu_env.sh"
  exit 1
fi

conda env create -n "$ENV_NAME" -f "$ENV_FILE"

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

python -m pip install --no-deps -e .

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

# Prefer CUDA libraries from pip-installed nvidia packages over system paths.
JAX_CUDA_LIB_PATHS="$(python "$PROJECT_DIR/scripts/jax_cuda_lib_paths.py")"
if [[ -n "$JAX_CUDA_LIB_PATHS" ]]; then
  export LD_LIBRARY_PATH="$JAX_CUDA_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python - <<'PY'
import jax
import jax.numpy as jnp

print("jax", jax.__version__)
print("devices", jax.devices())
print("backend", jax.default_backend())
print("sum", float(jnp.sum(jnp.ones((2, 3)))))
PY

python -B -m jax_fdtd_lab.run_sanity_checks --steps 8

echo "gpr-jax-gpu environment is ready."
