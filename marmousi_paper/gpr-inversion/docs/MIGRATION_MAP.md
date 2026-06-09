# Migration Map

## Stable shared modules

These modules currently have identical copies across many old folders, so they
were migrated first:

- `Add_CPML.py` -> `src/gpr_inversion/common/Add_CPML.py`
- `Wavelet.py` -> `src/gpr_inversion/common/Wavelet.py`

## First migrated experiment

Source:

```text
OverThrust_Tests/Unet/mode2/twopara_epsfirst.py
```

Target:

```text
src/gpr_inversion/experiments/overthrust/mode2_eps_then_sig/
```

Copied experiment-specific modules:

- `Time_loop.py`
- `forward.py`
- `gradient.py`
- `unet.py`
- `twopara_epsfirst.py`

Refactored helper modules:

- `dataset.py` <- the small torch Dataset wrapper from `twopara_epsfirst.py`
- `regularization.py` <- TV and Tikhonov helper code
- `runtime.py` <- safe result-directory creation, seed setup, gradient cleanup, and parameter counting
- `visualization.py` <- model, gradient, and loss plotting functions

Project-level scaffold now available for future experiment lines:

- `config.py` and `run.py` provide the config-driven entry point.
- `configs/overthrust_mode2_unet_eps_then_sig.yaml` is the first migrated config.
- `acquisition/` centralizes mode1/mode2 geometry.
- `models/` centralizes model loading and initial-model creation.
- `io/` centralizes safe output folders and `.npy`/`.bin` snapshots.
- `regularization/` and `visualization/` provide shared helpers for later migrations.
- `docs/EXPERIMENT_MATRIX.md` records which old experiment lines are supported
  and which are still planned.

## Traditional Marmousi epsilon-only migration

Source folders:

```text
traditional/mode1_LBFGS/
traditional/mode1_RMSprop/
traditional/mode2_LBFGS/
traditional/mode2_RMSprop/
```

Target:

```text
src/gpr_inversion/experiments/marmousi/traditional_eps_only/
```

The four config-registered lines are now dispatched through
`traditional_eps_only/runner.py`. The runner keeps the old numerical behavior
split where the legacy kernels genuinely differ:

- `mode1/` is shared by mode1 LBFGS and RMSprop.
- `mode2_lbfgs/` keeps the mode2 LBFGS FDTD/gradient variant.
- `mode2_rmsprop/` keeps the mode2 RMSprop FDTD/gradient variant.

The old `ensure_clean_dir` deletion behavior is not migrated. Outputs now use
the project-level unique-directory helper so previous runs remain untouched.

## UNet Marmousi epsilon-only migration

Source folders:

```text
unet/mode1_unet_fix/
unet/mode1_unet_fix_illu/
unet/mode2_unet_fix/
unet/mode2_unet_fix_illu/
```

Target:

```text
src/gpr_inversion/experiments/marmousi/unet_eps_only/
```

The four UNet Marmousi configs are now dispatched through the unified runner.
The old training scripts were kept as behavior-preserving migrated entry
points, but their hard-coded values were exposed as CLI arguments so the YAML
configs control data path, output path, device, learning rate, Tikhonov weight,
epoch count, and fixed shallow rows.

Numerical kernels remain split by old-line behavior:

- `mode1/` keeps mode1 no-illumination kernels.
- `mode1_illumination/` keeps mode1 illumination training behavior.
- `mode2/` keeps mode2 no-illumination kernels.
- `mode2_illumination/` keeps mode2 illumination kernels.

Like the traditional migration, the old recursive output deletion behavior is
not used. Existing run folders are preserved and new suffixes are created when
needed.

## Important Design Decisions

- Old folders are not modified or deleted.
- MPI remains the default runtime.
- torch remains the tensor and neural-network stack.
- Mode1 and Mode2 should remain separate modules.
- Illumination should become a runtime option, not a copied script family.
- Inversion strategy should become explicit in folder/config names:
  - `eps_only`
  - `sig_only`
  - `twopara`
  - `eps_then_sig`

## Safety Rule

New code must not recursively delete result folders. When output already exists,
create a new run folder instead.
