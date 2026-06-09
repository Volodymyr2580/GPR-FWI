#!/usr/bin/env bash
set -euo pipefail

# Diagnose the CUDA/NCCL libraries that PyTorch will load in the active conda
# environment. Run after `conda activate ...` and, ideally, after
# `source scripts/use_gpr_jax_env.sh`.

echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "which python: $(command -v python)"
echo

echo "Python package versions:"
python -m pip show torch torchvision jax jaxlib nvidia-nccl-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 || true
echo

echo "Candidate NCCL libraries:"
python - <<'PY'
import glob
import os
import site

roots = []
for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
    roots.extend(glob.glob(os.path.join(site_dir, "nvidia", "nccl", "lib", "libnccl.so*")))
    roots.extend(glob.glob(os.path.join(site_dir, "torch", "lib", "libnccl.so*")))
if os.environ.get("CONDA_PREFIX"):
    roots.extend(glob.glob(os.path.join(os.environ["CONDA_PREFIX"], "lib", "libnccl.so*")))
for path in sorted(set(roots)):
    print(path)
PY
echo

TORCH_CUDA_SO="$(
  python - <<'PY'
import glob
import os
import site

for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
    matches = glob.glob(os.path.join(site_dir, "torch", "lib", "libtorch_cuda.so"))
    if matches:
        print(matches[0])
        break
PY
)"

if [[ -z "$TORCH_CUDA_SO" ]]; then
  echo "Could not find torch/lib/libtorch_cuda.so"
  exit 1
fi

echo "libtorch_cuda.so:"
echo "$TORCH_CUDA_SO"
echo

echo "ldd CUDA/NCCL links:"
ldd "$TORCH_CUDA_SO" | grep -E "nccl|cuda|cudnn|cublas|cufft|cusparse|cusolver" || true
echo

echo "First LD_LIBRARY_PATH entries:"
echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | sed -n '1,40p'
