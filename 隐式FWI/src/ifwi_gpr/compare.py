"""Compare IFWI and dropout-IFWI runs with paper-style visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ifwi_gpr.io import write_json
from ifwi_gpr.metrics import parameter_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare IFWI run directories.")
    parser.add_argument("--ifwi-run", required=True, help="Run directory for IFWI.")
    parser.add_argument(
        "--dropout-run",
        required=True,
        help="Run directory for dropout-IFWI.",
    )
    parser.add_argument(
        "--output",
        default="runs/comparisons/cross_ifwi_vs_dropout",
        help="Output directory for comparison artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = compare_ifwi_runs(
        Path(args.ifwi_run),
        Path(args.dropout_run),
        Path(args.output),
    )
    print(f"Comparison complete: {output_dir}")


def compare_ifwi_runs(ifwi_run: Path, dropout_run: Path, output_dir: Path) -> Path:
    """Create side-by-side figures and metrics for two run directories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ifwi = _load_run(ifwi_run)
    dropout = _load_run(dropout_run)
    _validate_shared_truth(ifwi, dropout)

    metrics = {
        "ifwi_run": str(ifwi_run),
        "dropout_run": str(dropout_run),
        "epsilon": {
            "initial": parameter_metrics(ifwi["initial_epsilon"], ifwi["true_epsilon"]),
            "ifwi": parameter_metrics(ifwi["final_epsilon"], ifwi["true_epsilon"]),
            "dropout_ifwi": parameter_metrics(
                dropout["final_epsilon"],
                dropout["true_epsilon"],
            ),
        },
        "sigma": {
            "initial": parameter_metrics(ifwi["initial_sigma"], ifwi["true_sigma"]),
            "ifwi": parameter_metrics(ifwi["final_sigma"], ifwi["true_sigma"]),
            "dropout_ifwi": parameter_metrics(
                dropout["final_sigma"],
                dropout["true_sigma"],
            ),
        },
        "data_misfit": {
            "ifwi": _load_data_misfit(ifwi_run),
            "dropout_ifwi": _load_data_misfit(dropout_run),
        },
    }
    write_json(output_dir / "comparison_metrics.json", metrics)

    _save_parameter_comparison(output_dir, ifwi, dropout)
    _save_error_comparison(output_dir, ifwi, dropout)
    _save_loss_comparison(output_dir, ifwi_run, dropout_run)
    return output_dir


def _load_run(run_dir: Path) -> dict[str, np.ndarray]:
    names = [
        "true_epsilon",
        "true_sigma",
        "initial_epsilon",
        "initial_sigma",
        "final_epsilon",
        "final_sigma",
    ]
    arrays: dict[str, np.ndarray] = {}
    for name in names:
        path = run_dir / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing required run artifact: {path}")
        arrays[name] = np.load(path)
    return arrays


def _validate_shared_truth(ifwi: dict[str, np.ndarray], dropout: dict[str, np.ndarray]) -> None:
    for name in ("true_epsilon", "true_sigma"):
        if ifwi[name].shape != dropout[name].shape:
            raise ValueError(f"{name} shape mismatch")
        if not np.allclose(ifwi[name], dropout[name]):
            raise ValueError(f"{name} differs between compared runs")


def _load_data_misfit(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return metrics.get("data_misfit", {})


def _save_parameter_comparison(
    output_dir: Path,
    ifwi: dict[str, np.ndarray],
    dropout: dict[str, np.ndarray],
) -> None:
    panels = [
        ("True epsilon_r", ifwi["true_epsilon"]),
        ("Initial epsilon_r", ifwi["initial_epsilon"]),
        ("IFWI epsilon_r", ifwi["final_epsilon"]),
        ("dropout-IFWI epsilon_r", dropout["final_epsilon"]),
        ("True sigma (S/m)", ifwi["true_sigma"]),
        ("Initial sigma (S/m)", ifwi["initial_sigma"]),
        ("IFWI sigma (S/m)", ifwi["final_sigma"]),
        ("dropout-IFWI sigma (S/m)", dropout["final_sigma"]),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), dpi=220, constrained_layout=True)
    eps_values = [data for title, data in panels[:4]]
    sig_values = [data for title, data in panels[4:]]
    eps_vmin = float(min(np.min(data) for data in eps_values))
    eps_vmax = float(max(np.max(data) for data in eps_values))
    sig_vmin = float(min(np.min(data) for data in sig_values))
    sig_vmax = float(max(np.max(data) for data in sig_values))

    eps_artist = None
    sig_artist = None
    for ax, (title, data) in zip(axes[0], panels[:4]):
        eps_artist = ax.imshow(data, cmap="viridis", vmin=eps_vmin, vmax=eps_vmax)
        _style_map_axis(ax, title)
    for ax, (title, data) in zip(axes[1], panels[4:]):
        sig_artist = ax.imshow(data, cmap="magma", vmin=sig_vmin, vmax=sig_vmax)
        _style_map_axis(ax, title)
    fig.colorbar(eps_artist, ax=axes[0], shrink=0.78, label="epsilon_r")
    fig.colorbar(sig_artist, ax=axes[1], shrink=0.78, label="sigma (S/m)")
    _save_figure(fig, output_dir / "parameter_comparison")


def _save_error_comparison(
    output_dir: Path,
    ifwi: dict[str, np.ndarray],
    dropout: dict[str, np.ndarray],
) -> None:
    eps_errors = [
        ("Initial |epsilon error|", np.abs(ifwi["initial_epsilon"] - ifwi["true_epsilon"])),
        ("IFWI |epsilon error|", np.abs(ifwi["final_epsilon"] - ifwi["true_epsilon"])),
        (
            "dropout-IFWI |epsilon error|",
            np.abs(dropout["final_epsilon"] - dropout["true_epsilon"]),
        ),
    ]
    sig_errors = [
        ("Initial |sigma error|", np.abs(ifwi["initial_sigma"] - ifwi["true_sigma"])),
        ("IFWI |sigma error|", np.abs(ifwi["final_sigma"] - ifwi["true_sigma"])),
        (
            "dropout-IFWI |sigma error|",
            np.abs(dropout["final_sigma"] - dropout["true_sigma"]),
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=220, constrained_layout=True)
    eps_vmax = float(max(np.max(data) for _, data in eps_errors))
    sig_vmax = float(max(np.max(data) for _, data in sig_errors))
    eps_artist = None
    sig_artist = None
    for ax, (title, data) in zip(axes[0], eps_errors):
        eps_artist = ax.imshow(data, cmap="inferno", vmin=0.0, vmax=eps_vmax)
        _style_map_axis(ax, title)
    for ax, (title, data) in zip(axes[1], sig_errors):
        sig_artist = ax.imshow(data, cmap="cividis", vmin=0.0, vmax=sig_vmax)
        _style_map_axis(ax, title)
    fig.colorbar(eps_artist, ax=axes[0], shrink=0.78, label="|delta epsilon_r|")
    fig.colorbar(sig_artist, ax=axes[1], shrink=0.78, label="|delta sigma| (S/m)")
    _save_figure(fig, output_dir / "error_comparison")


def _save_loss_comparison(output_dir: Path, ifwi_run: Path, dropout_run: Path) -> None:
    ifwi_history = _load_history(ifwi_run)
    dropout_history = _load_history(dropout_run)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5), dpi=220, constrained_layout=True)
    for label, history, color in (
        ("IFWI", ifwi_history, "#1f77b4"),
        ("dropout-IFWI", dropout_history, "#d62728"),
    ):
        epochs = [int(item["epoch"]) for item in history]
        losses = [float(item["loss"]) for item in history]
        if epochs:
            ax.semilogy(epochs, losses, marker="o", linewidth=1.5, markersize=3, label=label, color=color)
    ax.set_title("Waveform Loss Comparison", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend()
    _save_figure(fig, output_dir / "loss_comparison")


def _load_history(run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return []
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return list(metrics.get("history", []))


def _style_map_axis(ax: Any, title: str) -> None:
    ax.set_title(title, fontsize=9.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _save_figure(fig: Any, path_stem: Path) -> None:
    fig.savefig(path_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
