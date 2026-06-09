#!/usr/bin/env bash
set -euo pipefail

# Create or update the Linux server environment used by the GPR/JAX checks and
# the PyTorch+MPI inversion jobs.
#
# Usage:
#   bash scripts/setup_gpr_jax_env.sh
#
# Optional overrides:
#   ENV_NAME=gpr-jax bash scripts/setup_gpr_jax_env.sh
#   SKIP_VERIFY=1 bash scripts/setup_gpr_jax_env.sh

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/environment-gpr-jax.yml}"
ENV_NAME="${ENV_NAME:-gpr-jax}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"

cd "$PROJECT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Please load Anaconda/Miniconda first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Updating existing conda environment: $ENV_NAME"
  conda env update -n "$ENV_NAME" -f "$ENV_FILE"
else
  echo "Creating conda environment: $ENV_NAME"
  conda env create -f "$ENV_FILE"
fi

if [[ "$SKIP_VERIFY" == "1" ]]; then
  echo "Environment setup finished. Verification skipped."
  exit 0
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

# Some servers preload an older NCCL/CUDA library through LD_LIBRARY_PATH. The
# pip CUDA wheels used by torch and JAX ship their own matching nvidia/*/lib
# directories, so put those first before importing torch.
NVIDIA_PIP_LIB_PATHS="$(
  python - <<'PY'
import os
import site

roots = []
for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
    torch_lib = os.path.join(site_dir, "torch", "lib")
    if os.path.isdir(torch_lib):
        roots.append(torch_lib)
    nvidia_root = os.path.join(site_dir, "nvidia")
    if os.path.isdir(nvidia_root):
        package_names = sorted(os.listdir(nvidia_root))
        if "nccl" in package_names:
            package_names.remove("nccl")
            package_names.insert(0, "nccl")
        for package_name in package_names:
            lib_dir = os.path.join(nvidia_root, package_name, "lib")
            if os.path.isdir(lib_dir):
                roots.append(lib_dir)
unique_roots = []
for root in roots:
    if root not in unique_roots:
        unique_roots.append(root)
print(":".join(unique_roots))
PY
)"
if [[ -n "$NVIDIA_PIP_LIB_PATHS" ]]; then
  export LD_LIBRARY_PATH="$(python - "$NVIDIA_PIP_LIB_PATHS" "${LD_LIBRARY_PATH:-}" <<'PY'
import sys

paths = []
for chunk in sys.argv[1:]:
    paths.extend(path for path in chunk.split(":") if path)
deduped = []
for path in paths:
    if path not in deduped:
        deduped.append(path)
print(":".join(deduped))
PY
)"
fi
echo "NVIDIA_PIP_LIB_PATHS=$NVIDIA_PIP_LIB_PATHS"

echo "Running import and device checks..."
python - <<'PY'
import jax
import jax.numpy as jnp
import numpy as np
import optuna
import scipy
import torch
from mpi4py import MPI

import gpr_inversion
from gpr_inversion.experiments.overthrust.mode2_eps_then_sig.unet import UNet
import jax_fdtd_lab.losses

print("python imports ok")
print(f"numpy: {np.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"torch: {torch.__version__}")
print(f"torch cuda available: {torch.cuda.is_available()}")
print(f"torch cuda device count: {torch.cuda.device_count()}")
print(f"jax: {jax.__version__}")
print(f"jax devices: {jax.devices()}")
print(f"optuna: {optuna.__version__}")
print(f"mpi rank/size: {MPI.COMM_WORLD.Get_rank()}/{MPI.COMM_WORLD.Get_size()}")
print(f"unet class import ok: {UNet.__name__}")
print(f"jax sample sum: {float(jnp.sum(jnp.ones((2, 3))))}")
PY

echo "Running a tiny MPI check..."
mpirun -np 2 python -c 'from mpi4py import MPI; comm = MPI.COMM_WORLD; print(f"hello from rank {comm.Get_rank()} of {comm.Get_size()}", flush=True)'

echo "gpr-jax environment is ready."
