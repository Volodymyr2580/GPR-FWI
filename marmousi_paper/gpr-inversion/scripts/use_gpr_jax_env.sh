#!/usr/bin/env bash

# Source this file before running server experiments:
#   source scripts/use_gpr_jax_env.sh
#
# It activates the gpr-jax conda environment and makes sure Python imports the
# CUDA/NCCL libraries installed inside that environment before any older system
# libraries.

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-gpr-jax}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Please load Anaconda/Miniconda first." >&2
  return 1 2>/dev/null || exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

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

echo "Activated $ENV_NAME"
echo "PYTHONPATH=$PYTHONPATH"
echo "NVIDIA_PIP_LIB_PATHS=$NVIDIA_PIP_LIB_PATHS"
