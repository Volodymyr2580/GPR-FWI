#!/usr/bin/env bash
set -euo pipefail

# Install the pip-side dependencies for the active conda environment.
# Run after:
#   conda activate gpr-jax-cu121
#
# PyTorch is installed in a separate command with --index-url so pip cannot
# silently choose a newer CUDA family from the default PyPI index.

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "No active conda environment. Run: conda activate gpr-jax-cu121" >&2
  exit 1
fi

JAX_FLAVOR="${JAX_FLAVOR:-cuda12}"

python -m pip uninstall -y torch torchvision torchaudio
python -m pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt
python -m pip uninstall -y nvidia-nccl-cu11 nvidia-nccl-cu12 nvidia-nccl-cu13 || true
python -m pip uninstall -y nvidia-cublas-cu13 nvidia-cuda-runtime-cu13 nvidia-cudnn-cu13 || true

python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1+cu121 \
  torchvision==0.20.1+cu121

if [[ "$JAX_FLAVOR" == "cpu" ]]; then
  python -m pip install --no-cache-dir jax==0.4.35 jaxlib==0.4.35
elif [[ "$JAX_FLAVOR" == "cuda12" ]]; then
  python -m pip install --no-cache-dir "jax[cuda12]==0.4.35"
else
  echo "Unsupported JAX_FLAVOR=$JAX_FLAVOR. Use cuda12 or cpu." >&2
  exit 1
fi

python -m pip install --no-cache-dir -r requirements-gpr-jax-cu121.txt

python -m pip check
