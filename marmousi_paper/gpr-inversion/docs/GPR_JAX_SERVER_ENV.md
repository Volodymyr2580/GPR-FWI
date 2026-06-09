# GPR-JAX Server Environment

This project can use one Conda environment for both:

- JAX sanity checks under `jax_fdtd_lab/`.
- PyTorch + MPI inversion jobs under `src/gpr_inversion/`.

## One-command setup

From the `gpr-inversion` folder on the Linux server:

```bash
bash scripts/setup_gpr_jax_env.sh
```

The script creates or updates the Conda environment named `gpr-jax`, installs the
project in editable mode, and runs small import/CUDA/MPI checks.

The Conda-level dependencies are listed in `environment-gpr-jax.yml`. The pip
dependencies, including PyTorch CUDA 12 and JAX CUDA 12 wheels, are listed in
`requirements-gpr-jax-cu121.txt`.

If pip has already mixed incompatible PyTorch/CUDA packages, repair the active
environment with:

```bash
conda activate gpr-jax-cu121
bash scripts/install_gpr_jax_cu121_pip.sh
```

If JAX CUDA continues to conflict with cuDNN on the server, keep PyTorch on GPU
and install CPU-only JAX for the lightweight lab checks:

```bash
conda activate gpr-jax-cu121
JAX_FLAVOR=cpu bash scripts/install_gpr_jax_cu121_pip.sh
```

## Manual setup

If you prefer to run Conda directly:

```bash
conda env create -f environment-gpr-jax.yml
conda activate gpr-jax
export PYTHONPATH="$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1
```

If the environment already exists:

```bash
conda env update -n gpr-jax -f environment-gpr-jax.yml
conda activate gpr-jax
```

## Smoke checks

```bash
python -B -m unittest jax_fdtd_lab.test_fdtd_lab
python -B -m jax_fdtd_lab.run_sanity_checks --steps 8
mpirun -np 2 python -m gpr_inversion.experiments.overthrust.mode2_eps_then_sig.twopara_epsfirst \
  --device cuda:0 \
  --fixed-top-rows 50 \
  --stage1-data-loss l1 \
  --stage2-data-loss l2 \
  --num-epochs-stage1 1 \
  --num-epochs-stage2 1 \
  --snapshot-interval 1 \
  --output-dir outputs/overthrust/smoke_top50
```

## Notes

`environment-gpr-jax.yml` uses CUDA 12 Python wheels for PyTorch and JAX. The
repair script intentionally installs PyTorch with `--index-url` against the
official cu121 wheel index, pins JAX to `0.4.35`, then installs the remaining
pip requirements. This avoids accidentally resolving a newer CUDA 13 wheel set
or a newer JAX build that requires cuDNN 9.8. This does not install a system
NVIDIA driver. The server still needs a compatible NVIDIA driver that can run
CUDA 12 workloads.

If `import torch` fails with an NCCL symbol error such as `undefined symbol:
ncclCommResume`, the environment is probably seeing an older system NCCL before
the pip-installed CUDA libraries. Source the runtime helper before checks or
experiments:

```bash
source scripts/use_gpr_jax_env.sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`PYTHONPATH` tells Python where this project package lives. `PYTHONDONTWRITEBYTECODE=1`
prevents Python from creating `__pycache__` files during checks, which keeps the
server workspace cleaner.
