# gpr-inversion

This folder is the cleaned migration target for the GPR inversion code.

The old project folders are kept untouched. New work should move here one
experiment line at a time, starting from the latest OverThrust Mode2
epsilon-then-sigma implementation.

## Current migrated path

- Model: OverThrust
- Mode: Mode2
- Runtime: MPI
- Tensor stack: torch
- Inversion mode: epsilon first, then sigma
- Latest source script: `OverThrust_Tests/Unet/mode2/twopara_epsfirst.py`

## Run

From this folder:

```powershell
python -m pip install -e .
mpirun -np 20 python -m gpr_inversion.run --config configs/overthrust_mode2_unet_eps_then_sig.yaml
```

Outputs are written under:

```text
outputs/overthrust/mode2_unet_eps_then_sig/
```

If a result folder already exists, the program creates a new suffix such as
`_run001` instead of deleting previous results.

## Beginner Notes

`src/gpr_inversion` is the Python package. A package is a folder that Python can
import with `python -m ...`.

`common` contains code that should not change between experiments, such as CPML
and wavelet helpers.

`experiments` contains code that may vary by model, mode, illumination setting,
or inversion strategy.

`configs` contains experiment parameter files. For a beginner: a config file is
just a readable table of choices such as model, mode, optimizer, regularization,
device, and output directory. You should normally change a config before editing
Python source code.

## Current Code Organization

The project now has a shared structure for future migrations:

- `acquisition`: mode1/mode2 source and receiver geometry.
- `models`: Marmousi/OverThrust loading, resizing, and initial-model helpers.
- `regularization`: shared TV gradient composition helpers.
- `visualization`: shared paper-style model, gradient, and loss plotting.
- `io`: safe output-folder and model-snapshot helpers.
- `run.py`: unified config-driven runner.
- `experiments`: thin migrated experiment packages.
- `docs/EXPERIMENT_MATRIX.md`: migration status for old experiment lines.

The runner key includes illumination because the same model/mode/optimizer can
have two different numerical paths: one with illumination compensation and one
without it.

Eight Marmousi epsilon-only lines are now connected to the same runner:

- `marmousi_mode1_traditional_eps_lbfgs.yaml`
- `marmousi_mode1_traditional_eps_rmsprop.yaml`
- `marmousi_mode2_traditional_eps_lbfgs.yaml`
- `marmousi_mode2_traditional_eps_rmsprop.yaml`
- `marmousi_mode1_unet_eps_adam.yaml`
- `marmousi_mode1_unet_eps_adam_illumination.yaml`
- `marmousi_mode2_unet_eps_adam.yaml`
- `marmousi_mode2_unet_eps_adam_illumination.yaml`

The traditional lines use migrated NumPy/FDTD kernels under
`experiments/marmousi/traditional_eps_only/`. The UNet lines use migrated
training scripts and kernels under `experiments/marmousi/unet_eps_only/`.
Existing result folders are not deleted; the output helper creates a suffixed
folder when needed.

See `docs/MARMOUSI_EXPERIMENTS_GUIDE.md` for the complete beginner-friendly
Marmousi migration and run guide.
See `docs/PROJECT_STATUS_AND_PROGRESS.md` for the current project overview and
experiment progress log.

The first migrated experiment remains split into smaller files:

- `twopara_epsfirst.py`: experiment entry point and training loop.
- `forward.py`, `Time_loop.py`, `gradient.py`: mode2 FDTD forward and adjoint-gradient code.
- `regularization.py`: TV and Tikhonov regularization helpers.
- `visualization.py`: model, gradient, and loss-curve plotting helpers.
- `runtime.py`: seed setting, parameter counting, gradient cleanup, and safe result-folder creation.
- `dataset.py`: tiny torch Dataset wrapper for observed data and geometry.
- `unet.py`: UNet reparameterization network.

For a beginner: the entry point is the file you run; helper modules are files
imported by the entry point. Splitting them does not change the math by itself,
but it makes later mode1/mode2, Marmousi/OverThrust, and regularization variants
easier to compare without copying the same plotting or utility code everywhere.

The current script saves intermediate epsilon and sigma arrays every 1000 epochs
as `.npy` files, so later plotting or paper-figure processing can reuse the raw
model data without rerunning inversion.

Regularization note: TV is applied in the custom gradient path through
`--alpha-tv-eps` and `--alpha-tv-sig`. Tikhonov is optional through
`--alpha-tikhonov-eps` and `--alpha-tikhonov-sig`; with the default value `0`,
it does not change the loss.

## OverThrust Optuna Search

The OverThrust eps-then-sig line has a Linux server launcher for hyperparameter
search:

```bash
STAGE1_CHECKPOINT_PATH=/path/to/stage1_eps_checkpoint.pt \
DEVICE=cuda:1 \
N_TRIALS=50 \
NUM_EPOCHS_STAGE2=1000 \
bash scripts/run_overthrust_optuna_search.sh
```

By default it uses Optuna's TPE sampler to minimize `data_misfit_mse` after
each 1000-epoch trial. The default search space is:

- `lr_eps_stage2`: log scale from `1e-6` to `1e-4`
- `learning_rate_sig`: log scale from `1e-6` to `1e-4`
- `alpha_tv_eps`: linear scale from `1e-2` to `5e-2`
- `alpha_tv_sig`: linear scale from `3e-2` to `4e-2`

The mixed L1 data term is available as `alpha_l1_data`. The launcher searches
it from `0.0` to `ALPHA_L1_DATA_MAX` by default:

```bash
ALPHA_L1_DATA_MAX=0.5 \
STAGE1_CHECKPOINT_PATH=/path/to/stage1_eps_checkpoint.pt \
bash scripts/run_overthrust_optuna_search.sh
```

Set `SEARCH_ALPHA_L1_DATA=0` to keep `ALPHA_L1_DATA` fixed for a narrower run.

Search outputs are written under `outputs/overthrust/optuna_eps_then_sig/`.
Each trial has its own directory, and the study state is stored in
`optuna_study.db`. The plain-text summary file `trials_summary.jsonl` is useful
for quick inspection or plotting.

## Lightweight Checks

Use the existing conda environment without installing anything to C drive:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONDONTWRITEBYTECODE="1"
conda run -n gprfwi python -m gpr_inversion.run --config configs\overthrust_mode2_unet_eps_then_sig.yaml --describe
conda run -n gprfwi python -m gpr_inversion.run --config configs\overthrust_mode2_unet_eps_then_sig.yaml --dry-run
conda run -n gprfwi python -m gpr_inversion.run --list-supported
conda run -n gprfwi python -m gpr_inversion.run --list-planned
conda run -n gprfwi python -m gpr_inversion.run --check-configs
conda run -n gprfwi python -m unittest tests\test_config_and_helpers.py
```

`--describe` only prints the parsed config. `--dry-run` checks that the unified
runner can translate the config into the migrated experiment's current CLI
arguments without starting the full inversion.
