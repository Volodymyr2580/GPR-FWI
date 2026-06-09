# Architecture Notes

## Axes That Must Stay Separate

The old project mixes several independent choices into copied script names. The
new project should keep these choices explicit:

| Axis | Meaning | Recommended Location |
| --- | --- | --- |
| Model | Marmousi or OverThrust data and geometry | `experiments/<model>/` |
| Mode | Mode1 or Mode2 numerical setup | `mode1_*` or `mode2_*` package |
| Illumination | Whether illumination compensation is computed | runtime flag/config |
| Inversion strategy | What parameter is inverted and in what order | package/config name |
| Runtime | MPI execution | shared runner behavior |
| Tensor stack | torch tensors and autograd bridge | inversion module |

## Current First Target

```text
experiments/overthrust/mode2_eps_then_sig/
```

This target means:

- model: OverThrust
- mode: Mode2
- inversion strategy: first epsilon, then sigma
- illumination: optional via `--illumination`
- runtime: MPI
- tensor stack: torch

## Module Rules

`common` is for files that should be identical in every experiment:

- CPML boundary setup
- wavelet generation

Mode-specific code should not go into `common`. If `forward.py`,
`gradient.py`, or `Time_loop.py` differs between Mode1 and Mode2, keep separate
Mode1 and Mode2 implementations.

Illumination should be controlled by a parameter. Avoid creating separate files
such as `with_illumination.py` and `without_illumination.py` when the numerical
path is otherwise the same.

Top-level shared packages are the preferred home for code that will be reused
across models and inversion styles:

| Package | Role |
| --- | --- |
| `acquisition` | mode1 self-transmit/self-receive and mode2 multi-offset geometry |
| `models` | model loading, resizing, sigma construction, and initial models |
| `regularization` | reusable TV helpers; experiment-specific autograd bridges can call these |
| `visualization` | paper-style plotting helpers |
| `io` | output directories and raw model snapshots |
| `run.py` / `config.py` | config-driven experiment entry point and validation |

Inside each migrated experiment package, keep the script roles small:

| File | Role |
| --- | --- |
| `twopara_epsfirst.py` | CLI entry point and inversion/training flow |
| `forward.py` / `Time_loop.py` / `gradient.py` | mode-specific numerical kernels |
| `regularization.py` | TV and Tikhonov regularization helpers |
| `visualization.py` | paper-style model, gradient, and loss plotting |
| `runtime.py` | run-folder safety, seed setup, gradient cleanup, parameter counts |
| `dataset.py` | torch Dataset wrapper for observed data and geometry |

Regularization should be callable from a dedicated helper module. The current
migrated experiment uses `regularization.py` for both TV gradient composition
and the optional Tikhonov loss. TV is controlled by `--alpha-tv-*`; Tikhonov is
controlled by `--alpha-tikhonov-*` and is only added to the loss when one of
those coefficients is non-zero.

Model snapshots should be saved as raw arrays at a fixed interval. The migrated
experiment currently writes `epoch_<n>_epsilon.npy` and `epoch_<n>_sigma.npy`
every 1000 epochs.

## Inversion Strategy Names

Use these names consistently:

- `eps_only`: invert epsilon only
- `sig_only`: invert sigma only
- `twopara`: invert epsilon and sigma together
- `eps_then_sig`: invert epsilon first, then sigma

## Migration Rule

Migrate one working experiment at a time. For each migration:

1. Write or copy a config in `configs/` that states the experiment axes.
2. Move reusable geometry/model/output/plotting logic to top-level packages.
3. Keep the migrated experiment package thin and mode-specific.
4. Register the experiment in `gpr_inversion.run.SUPPORTED_EXPERIMENTS`.
5. Verify with `--describe` and `--dry-run` before running inversion.
6. Run a short smoke test only after the config-driven dispatch is correct.
