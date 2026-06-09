"""Prepare Overthrust arrays for paper-style IFWI experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare paper-style Overthrust data.")
    parser.add_argument(
        "--source",
        default="/mnt/e/sci_research/GPR/marmousi_paper/gpr-inversion/data/overthrust/OverThrust.npy",
        help="Reference Overthrust NumPy file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/overthrust_paper",
        help="Directory where prepared arrays are written.",
    )
    parser.add_argument("--nx", type=int, default=94, help="Subsurface depth samples.")
    parser.add_argument("--nz", type=int, default=250, help="Horizontal samples.")
    parser.add_argument("--air-rows", type=int, default=10, help="50 cm air layer for dx=0.05 m.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = prepare_overthrust(
        source=Path(args.source),
        output_dir=Path(args.output_dir),
        nx=args.nx,
        nz=args.nz,
        air_rows=args.air_rows,
    )
    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))


def prepare_overthrust(
    *,
    source: Path,
    output_dir: Path,
    nx: int = 94,
    nz: int = 250,
    air_rows: int = 10,
) -> dict[str, Path]:
    raw = np.load(source).astype(np.float32)
    resized = zoom(raw, (nx / raw.shape[0], nz / raw.shape[1]), order=1).astype(np.float32)
    normalized = (resized - float(resized.min())) / max(float(resized.max() - resized.min()), 1e-12)

    epsilon_subsurface = 1.0 + normalized * 29.0
    sigma_subsurface = normalized * 0.02

    epsilon_true = _with_air_layer(epsilon_subsurface, air_rows, air_value=1.0)
    sigma_true = _with_air_layer(sigma_subsurface, air_rows, air_value=0.0)
    epsilon_init_accurate = _with_air_layer(
        _linear_depth_model(nx, nz, top=1.0, bottom=30.0),
        air_rows,
        air_value=1.0,
    )
    sigma_init_accurate = _with_air_layer(
        _linear_depth_model(nx, nz, top=0.0, bottom=0.02),
        air_rows,
        air_value=0.0,
    )
    epsilon_init_inaccurate = _with_air_layer(
        _linear_depth_model(nx, nz, top=1.0, bottom=25.0),
        air_rows,
        air_value=1.0,
    )
    sigma_init_inaccurate = _with_air_layer(
        _linear_depth_model(nx, nz, top=0.0, bottom=0.015),
        air_rows,
        air_value=0.0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "epsilon_true": output_dir / "epsilon_true.npy",
        "sigma_true": output_dir / "sigma_true.npy",
        "epsilon_init_accurate": output_dir / "epsilon_init_accurate.npy",
        "sigma_init_accurate": output_dir / "sigma_init_accurate.npy",
        "epsilon_init_inaccurate": output_dir / "epsilon_init_inaccurate.npy",
        "sigma_init_inaccurate": output_dir / "sigma_init_inaccurate.npy",
        "metadata": output_dir / "metadata.json",
    }
    np.save(outputs["epsilon_true"], epsilon_true.astype(np.float32))
    np.save(outputs["sigma_true"], sigma_true.astype(np.float32))
    np.save(outputs["epsilon_init_accurate"], epsilon_init_accurate.astype(np.float32))
    np.save(outputs["sigma_init_accurate"], sigma_init_accurate.astype(np.float32))
    np.save(outputs["epsilon_init_inaccurate"], epsilon_init_inaccurate.astype(np.float32))
    np.save(outputs["sigma_init_inaccurate"], sigma_init_inaccurate.astype(np.float32))
    outputs["metadata"].write_text(
        json.dumps(
            {
                "source": str(source),
                "raw_shape": list(raw.shape),
                "subsurface_shape": [nx, nz],
                "full_shape": list(epsilon_true.shape),
                "air_rows": air_rows,
                "epsilon_true_range": [float(epsilon_true.min()), float(epsilon_true.max())],
                "sigma_true_range": [float(sigma_true.min()), float(sigma_true.max())],
                "unit_note": "sigma is stored in S/m; 0.02 S/m equals 20 mS/m.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return outputs


def _with_air_layer(model: np.ndarray, air_rows: int, *, air_value: float) -> np.ndarray:
    if air_rows <= 0:
        return model
    air = np.full((air_rows, model.shape[1]), air_value, dtype=np.float32)
    return np.vstack([air, model.astype(np.float32, copy=False)])


def _linear_depth_model(nx: int, nz: int, *, top: float, bottom: float) -> np.ndarray:
    column = np.linspace(top, bottom, nx, dtype=np.float32)[:, None]
    return np.repeat(column, nz, axis=1)


if __name__ == "__main__":
    main()

