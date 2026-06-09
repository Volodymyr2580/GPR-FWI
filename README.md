# GPR-FWI

This repository organizes local research code and notes for ground penetrating
radar full waveform inversion (GPR-FWI).

The local workspace contains many historical experiments, generated figures,
NumPy arrays, model data, and CE4 processing results. The GitHub repository is
intended to keep the project skeleton: source code, configs, tests, and research
navigation documents. Large data and experiment outputs stay local.

## Research Lines

### 1. Implicit FWI Reproduction for Penetrating Radar Data

Main local folder:

- `隐式FWI/`

This is the current reproduction scaffold for implicit multiparameter FWI. It
already has a Python package layout:

- `隐式FWI/src/ifwi_gpr/`: reusable source code.
- `隐式FWI/configs/`: JSON experiment configs.
- `隐式FWI/tests/`: smoke tests and checks.
- `隐式FWI/plan.md`, `progress.md`, `paper_cross_setting_audit.md`: research notes.

Local-only folders such as `隐式FWI/data/`, `logs/`, `runs/`, `.venv/`, and
`.venv312/` are ignored by Git.

### 2. Traditional and Network-Reparameterized GPR-FWI

Main local folders:

- `Gpr_fwi/`: earlier single-machine experiments, including traditional,
  Tikhonov, hybrid, UNet, and autoencoder variants.
- `mpi_gprfwi/`: MPI/GPU-oriented experiment variants and larger test lines.
- `marmousi_paper/gpr-inversion/`: the clean migration target with package,
  config, docs, tests, and scripts.

For future work, prefer moving stable experiment code into
`marmousi_paper/gpr-inversion/` one line at a time instead of copying another
large experimental folder.

### 3. CE4 Data Processing

Main local folders:

- `CE4_CPU/`: compact CPU-side CE4 processing / inversion scripts.
- `CE4_data/`: CE4 processing scripts and result folders.
- `mpi_gprfwi/CE4_data/`: MPI-related CE4 data experiments.

The raw CE4 arrays and generated outputs are local-only assets and are not
intended for GitHub.

## Repository Map

See:

- `docs/PROJECT_INVENTORY.md` for the folder-by-folder inventory.
- `docs/DATA_AND_GIT_POLICY.md` for what should be committed and what should
  remain local.

## Beginner Notes

Git tracks files for version history. It is very good for source code, configs,
and notes, but it is a poor fit for huge generated files such as `.npy`, `.png`,
`.mat`, `.segy`, videos, and virtual environments.

The `.gitignore` file tells Git: "do not track these generated or heavy files."
It does not delete the files from your computer. It only prevents accidental
upload to GitHub.
