# Data and Git Policy

This project contains many experiment outputs and large scientific data files.
The GitHub repository should contain the reproducible skeleton, not the full
local experiment archive.

## Commit to GitHub

Good candidates for Git:

- source code: `.py`
- package metadata: `pyproject.toml`, `requirements*.txt`,
  `environment*.yml`, `uv.lock`
- configs: `.json`, `.yaml`, `.yml`
- tests
- small research notes: `.md`, `.txt` when they are written by us and safe to
  share

## Keep Local

Keep these out of Git by default:

- raw or generated arrays: `.npy`, `.npz`, `.mat`
- seismic/radar/model binaries: `.segy`, `.sgy`, `.bin`, `.dat`, `.sav`
- generated plots and media: `.png`, `.jpg`, `.mp4`, `.wav`
- compressed outputs: `.tar.gz`, `.zip`, `.7z`, `.rar`
- virtual environments: `.venv/`, `.venv312/`
- experiment outputs: `outputs/`, `runs/`, `logs/`, `results/`
- raw data folders: `data/`
- reference PDFs unless license and redistribution are clear

## Why This Matters

Git stores file history. If a 2 GB generated array is committed once, it remains
inside repository history even if the visible file is later removed. That makes
clone, push, and pull very slow.

For large research artifacts, use one of these instead:

- keep them on the local disk with a clear folder map;
- put selected data releases in cloud storage or a data repository;
- use Git LFS only after deciding which files genuinely need versioned binary
  storage.

## Current Decision

For the first GitHub sync, commit only the project skeleton:

- top-level README and organization docs;
- Git ignore rules;
- code/config/test/docs from the active subprojects;
- no raw data, no generated figures, no virtual environments, no large outputs.

This keeps the remote repository useful and lightweight.
