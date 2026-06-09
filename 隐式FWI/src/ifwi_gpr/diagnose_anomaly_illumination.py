"""Diagnose per-anomaly data visibility for the Cross-shape experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ifwi_gpr.acquisition import build_acquisition
from ifwi_gpr.config import load_config, require_section
from ifwi_gpr.train import build_solver, build_true_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate left/right anomaly waveform contributions by removal tests."
    )
    parser.add_argument("--config", required=True, help="Experiment config path.")
    parser.add_argument(
        "--output-dir",
        default="runs/diagnostics",
        help="Directory for JSON and figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epsilon_true, sigma_true = build_true_model(config)
    model_cfg = require_section(config, "model")
    shape = tuple(model_cfg["shape"])
    epsilon_background = float(model_cfg.get("epsilon_background", 4.0))
    sigma_background = float(model_cfg.get("sigma_background", 0.003))
    acquisition_cfg = require_section(config, "acquisition")
    sources, receivers = build_acquisition(acquisition_cfg, shape)
    bridge, settings = build_solver(config)

    left_mask = epsilon_true < epsilon_background
    right_mask = epsilon_true > epsilon_background
    epsilon_no_left, sigma_no_left = _remove_mask(
        epsilon_true,
        sigma_true,
        left_mask,
        epsilon_background=epsilon_background,
        sigma_background=sigma_background,
    )
    epsilon_no_right, sigma_no_right = _remove_mask(
        epsilon_true,
        sigma_true,
        right_mask,
        epsilon_background=epsilon_background,
        sigma_background=sigma_background,
    )

    data_true = bridge.forward(
        epsilon_true,
        sigma_true,
        sources,
        receivers,
        settings,
        save_wavefield=False,
    )
    data_no_left = bridge.forward(
        epsilon_no_left,
        sigma_no_left,
        sources,
        receivers,
        settings,
        save_wavefield=False,
    )
    data_no_right = bridge.forward(
        epsilon_no_right,
        sigma_no_right,
        sources,
        receivers,
        settings,
        save_wavefield=False,
    )

    left_delta = np.asarray(data_true - data_no_left)
    right_delta = np.asarray(data_true - data_no_right)
    summary = {
        "config": str(Path(args.config)),
        "source_count": len(sources),
        "receiver_count": len(receivers),
        "steps": int(settings.steps),
        "left_cell_count": int(left_mask.sum()),
        "right_cell_count": int(right_mask.sum()),
        "left_total_energy": _energy(left_delta),
        "right_total_energy": _energy(right_delta),
        "right_to_left_energy_ratio": _safe_ratio(_energy(right_delta), _energy(left_delta)),
        "left_per_shot_energy": _energy(left_delta, axis=(1, 2)).tolist(),
        "right_per_shot_energy": _energy(right_delta, axis=(1, 2)).tolist(),
        "left_per_receiver_energy": _energy(left_delta, axis=(0, 2)).tolist(),
        "right_per_receiver_energy": _energy(right_delta, axis=(0, 2)).tolist(),
        "left_shot_receiver_energy": _energy(left_delta, axis=2).tolist(),
        "right_shot_receiver_energy": _energy(right_delta, axis=2).tolist(),
        "sources": [list(item) for item in sources],
        "receivers": [list(item) for item in receivers],
    }

    stem = "cross_shape_anomaly_illumination"
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure_path = output_dir / f"{stem}.png"
    _plot_summary(summary, figure_path)
    print(json_path)
    print(figure_path)


def _remove_mask(
    epsilon: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray,
    *,
    epsilon_background: float,
    sigma_background: float,
) -> tuple[np.ndarray, np.ndarray]:
    epsilon_removed = np.asarray(epsilon).copy()
    sigma_removed = np.asarray(sigma).copy()
    epsilon_removed[mask] = epsilon_background
    sigma_removed[mask] = sigma_background
    return epsilon_removed, sigma_removed


def _energy(array: np.ndarray, axis: Any = None) -> Any:
    values = np.asarray(array, dtype=np.float64)
    return np.mean(values * values, axis=axis)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-30:
        return float("inf")
    return float(numerator / denominator)


def _plot_summary(summary: dict[str, Any], figure_path: Path) -> None:
    left_shot = np.asarray(summary["left_per_shot_energy"], dtype=np.float64)
    right_shot = np.asarray(summary["right_per_shot_energy"], dtype=np.float64)
    left_receiver = np.asarray(summary["left_per_receiver_energy"], dtype=np.float64)
    right_receiver = np.asarray(summary["right_per_receiver_energy"], dtype=np.float64)
    left_matrix = np.asarray(summary["left_shot_receiver_energy"], dtype=np.float64)
    right_matrix = np.asarray(summary["right_shot_receiver_energy"], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(left_shot, label="left anomaly")
    axes[0, 0].plot(right_shot, label="right anomaly")
    axes[0, 0].set_title("Per-shot waveform contribution")
    axes[0, 0].set_xlabel("source index")
    axes[0, 0].set_ylabel("mean squared delta")
    axes[0, 0].legend()

    axes[0, 1].plot(left_receiver, label="left anomaly")
    axes[0, 1].plot(right_receiver, label="right anomaly")
    axes[0, 1].set_title("Per-receiver waveform contribution")
    axes[0, 1].set_xlabel("receiver index")
    axes[0, 1].set_ylabel("mean squared delta")
    axes[0, 1].legend()

    vmax = max(float(left_matrix.max()), float(right_matrix.max()), 1e-30)
    im_left = axes[1, 0].imshow(left_matrix, aspect="auto", origin="lower", vmax=vmax)
    axes[1, 0].set_title("Left anomaly shot-receiver energy")
    axes[1, 0].set_xlabel("receiver index")
    axes[1, 0].set_ylabel("source index")
    fig.colorbar(im_left, ax=axes[1, 0], shrink=0.8)

    im_right = axes[1, 1].imshow(right_matrix, aspect="auto", origin="lower", vmax=vmax)
    axes[1, 1].set_title("Right anomaly shot-receiver energy")
    axes[1, 1].set_xlabel("receiver index")
    axes[1, 1].set_ylabel("source index")
    fig.colorbar(im_right, ax=axes[1, 1], shrink=0.8)

    fig.suptitle(
        "Cross-shape anomaly visibility: "
        f"right/left total energy={summary['right_to_left_energy_ratio']:.3f}"
    )
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
