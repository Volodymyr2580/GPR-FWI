# GPR JAX GPU Environment

This environment is for the JAX GPU prototype only. It deliberately keeps
PyTorch out of the environment, because PyTorch cu121 currently brings cuDNN
9.1 while current JAX CUDA 12 wheels need cuDNN 9.8 or newer.

The setup script installs this project with `pip install --no-deps -e .`. That
means the project package is importable, but pip will not pull the PyTorch
dependency declared in `pyproject.toml`.

## Create

From the `gpr-inversion` folder:

```bash
bash scripts/setup_gpr_jax_gpu_env.sh
```

If `gpr-jax-gpu` already exists, create a fresh named environment:

```bash
ENV_NAME=gpr-jax-gpu-v2 bash scripts/setup_gpr_jax_gpu_env.sh
```

## Use

```bash
source scripts/use_gpr_jax_gpu_env.sh
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

The helper unsets `LD_LIBRARY_PATH` on purpose. JAX's CUDA pip wheels include
their own `nvidia-*` CUDA/cuDNN packages, and a system CUDA path can override
those libraries.

## Checks

```bash
python -B -m jax_fdtd_lab.run_sanity_checks --steps 8
python -B -m jax_fdtd_lab.benchmark --case tiny --steps 80 --repeats 1
```

To save a repeatable server validation bundle:

```bash
bash scripts/run_jax_gpu_validation.sh
```

To include the larger `100x200, 1000 steps, 40 shots, 100 receivers` batched
benchmark:

```bash
RUN_FULLISH=1 bash scripts/run_jax_gpu_validation.sh
```

## Training Smoke Test

The OverThrust eps-then-sig entry point now has an experimental JAX backend.
This smoke test requires both PyTorch and JAX in the same environment, so run it
from the hybrid `gpr-jax-cu121` environment that you validated on the server,
not from the pure `gpr-jax-gpu` environment:

```bash
conda activate gpr-jax-cu121

python -m gpr_inversion.experiments.overthrust.mode2_eps_then_sig.twopara_epsfirst \
  --fdtd-backend jax \
  --illumination \
  --fixed-top-rows 50 \
  --stage1-data-loss l1 \
  --stage2-data-loss l2 \
  --num-epochs-stage1 1 \
  --num-epochs-stage2 1 \
  --snapshot-interval 1 \
  --device cuda:0 \
  --output-dir outputs/overthrust/jax_smoke_top50
```

The JAX backend supports illumination compensation by accumulating `sum(Ey^2)`
during forward propagation, then applying the same gradient normalization as the
NumPy path. It uses the hand-written JAX adjoint path and does not use
`jax.grad` on the forward computation graph.

Keep using `gpr-jax-cu121` for the PyTorch inversion path and JAX-backed
training smoke tests. Use the pure `gpr-jax-gpu` environment only when running
standalone JAX GPU validation code.
