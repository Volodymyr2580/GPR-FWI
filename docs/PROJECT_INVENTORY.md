# Project Inventory

This document records the current local folder organization in
`E:\sci_research\GPR`.

The purpose is to keep the research map clear without moving or deleting large
experiment assets.

## Top-Level Summary

| Folder | Approx. role | Local size observed | GitHub status |
| --- | --- | ---: | --- |
| `隐式FWI/` | implicit FWI reproduction scaffold | 2.06 GB | commit code/config/docs/tests; ignore data, logs, runs, venvs |
| `marmousi_paper/` | Marmousi/OverThrust paper experiments and cleaned package | 66.46 GB | commit `gpr-inversion` skeleton; ignore outputs/data |
| `Gpr_fwi/` | earlier traditional and UNet GPR-FWI experiments | 1.19 GB | keep as legacy code; avoid committing generated data |
| `mpi_gprfwi/` | MPI/GPU experiments and large model/data tests | 5.36 GB | keep selected code; ignore data/results/model files |
| `CE4_CPU/` | compact CE4 CPU workflow | 0.23 GB | commit scripts; ignore observed/synthetic arrays |
| `CE4_data/` | CE4 processing and test results | 0.26 GB | commit scripts; ignore data/results |
| `GPR_references/` | papers and literature references | 0.26 GB | keep local; do not push PDFs by default |

## Research Line 1: Implicit FWI

Primary folder:

- `隐式FWI/`

Important subfolders/files:

- `src/ifwi_gpr/`: Python package source code.
- `configs/`: experiment configurations.
- `tests/`: reproducibility and smoke tests.
- `plan.md`: planned reproduction path.
- `progress.md`: experiment progress notes.
- `paper_cross_setting_audit.md`: cross-setting audit notes.
- `pyproject.toml`, `uv.lock`: Python project metadata and lock file.

Local-only:

- `.venv/`, `.venv312/`: Python dependency environments.
- `data/`: input data or generated data.
- `logs/`, `runs/`: training/inversion outputs.
- paper PDFs and extracted text unless there is a clear license reason to share.

## Research Line 2: Traditional and Network-Reparameterized GPR-FWI

Primary folders:

- `Gpr_fwi/`
- `mpi_gprfwi/`
- `marmousi_paper/gpr-inversion/`

`Gpr_fwi/` contains many early variants:

- `mode1`, `mode2`: baseline mode-specific experiments.
- `mode*_Tikhonov`: traditional inversion with Tikhonov regularization.
- `mode*_Hybrid`: hybrid variants.
- `mode*_unet*`: network reparameterization experiments.
- `AE/`: autoencoder-related experiments.
- `RMSprop/`: RMSprop-based traditional workflow.

`mpi_gprfwi/` contains larger MPI and GPU experiment branches:

- `mode*_unet_*`: MPI/GPU UNet variants.
- `mode*_Tikhonov`: MPI/GPU traditional variants.
- `moving_downsample_big*`: downsampling and CuPy tests.
- `marmousi_and_overthrust/`: large model assets and related tests.

`marmousi_paper/gpr-inversion/` is the clean migration target:

- `src/gpr_inversion/`: package source.
- `configs/`: YAML experiment configs.
- `docs/`: architecture, migration, and experiment guides.
- `tests/`: checks.
- `scripts/`: helper scripts.
- `pyproject.toml`, `requirements-*.txt`, `environment-*.yml`: environment metadata.

Recommended direction:

Use `marmousi_paper/gpr-inversion/` as the future stable home. Move only the
best-understood experiment lines into it, and keep old folders as references.

## Research Line 3: CE4 Data Processing

Primary folders:

- `CE4_CPU/`
- `CE4_data/`
- `mpi_gprfwi/CE4_data/`

Important code:

- `main.py`, `main_test.py`: entry points.
- `utils/forward*.py`, `utils/gradient*.py`, `utils/Time_loop*.py`: forward
  modeling and gradient utilities.
- `migrate_to_gpu.py`, `test_mpi_gpu.py`: GPU/MPI migration experiments.

Local-only:

- observed/synthetic `.npy` arrays.
- generated result folders.

## Cleanup Notes

No batch deletion has been performed.

The current organization is documentation-first: the messy historical folders
remain available locally, while GitHub receives a safer project skeleton.
