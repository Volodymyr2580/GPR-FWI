"""Utilities for exporting publication-style IFWI figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def generate_run_figures(
    run_dir: str | Path,
    *,
    epsilon_true: np.ndarray,
    sigma_true: np.ndarray,
    epsilon_initial: np.ndarray,
    sigma_initial: np.ndarray,
    epsilon_final: np.ndarray,
    sigma_final: np.ndarray,
    metrics: dict[str, Any],
    dx: float,
    dz: float,
    observed_data: np.ndarray | None = None,
    synthetic_data: np.ndarray | None = None,
    dt: float | None = None,
    shot_index: int | None = None,
) -> Path:
    """Create default figure set for one IFWI run."""
    figures_dir = Path(run_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    extent = _build_extent(epsilon_true.shape, dx=dx, dz=dz)

    _save_parameter_maps(
        figures_dir,
        epsilon_true=epsilon_true,
        sigma_true=sigma_true,
        epsilon_initial=epsilon_initial,
        sigma_initial=sigma_initial,
        epsilon_final=epsilon_final,
        sigma_final=sigma_final,
        extent=extent,
    )
    _save_error_maps(
        figures_dir,
        epsilon_true=epsilon_true,
        sigma_true=sigma_true,
        epsilon_initial=epsilon_initial,
        sigma_initial=sigma_initial,
        epsilon_final=epsilon_final,
        sigma_final=sigma_final,
        extent=extent,
    )
    _save_loss_curve(figures_dir, metrics)
    if observed_data is not None and synthetic_data is not None:
        _save_data_comparison(
            figures_dir,
            observed_data=observed_data,
            synthetic_data=synthetic_data,
            dt=dt,
            shot_index=shot_index,
        )
    return figures_dir


def _build_extent(shape: tuple[int, int], *, dx: float, dz: float) -> list[float]:
    nx, nz = shape
    return [0.0, dz * (nz - 1), dx * (nx - 1), 0.0]


def _save_parameter_maps(
    figures_dir: Path,
    *,
    epsilon_true: np.ndarray,
    sigma_true: np.ndarray,
    epsilon_initial: np.ndarray,
    sigma_initial: np.ndarray,
    epsilon_final: np.ndarray,
    sigma_final: np.ndarray,
    extent: list[float],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=200, constrained_layout=True)

    eps_vmin = float(min(np.min(epsilon_true), np.min(epsilon_initial), np.min(epsilon_final)))
    eps_vmax = float(max(np.max(epsilon_true), np.max(epsilon_initial), np.max(epsilon_final)))
    sig_vmin = float(min(np.min(sigma_true), np.min(sigma_initial), np.min(sigma_final)))
    sig_vmax = float(max(np.max(sigma_true), np.max(sigma_initial), np.max(sigma_final)))

    eps_maps = [
        ("True epsilon_r", epsilon_true),
        ("Initial epsilon_r", epsilon_initial),
        ("Final epsilon_r", epsilon_final),
    ]
    sig_maps = [
        ("True sigma (S/m)", sigma_true),
        ("Initial sigma (S/m)", sigma_initial),
        ("Final sigma (S/m)", sigma_final),
    ]

    eps_artist = None
    sig_artist = None
    for ax, (title, data) in zip(axes[0], eps_maps):
        eps_artist = ax.imshow(
            data,
            cmap="viridis",
            vmin=eps_vmin,
            vmax=eps_vmax,
            extent=extent,
            aspect="auto",
        )
        ax.set_title(title, fontsize=10)
        _style_axis(ax)

    for ax, (title, data) in zip(axes[1], sig_maps):
        sig_artist = ax.imshow(
            data,
            cmap="magma",
            vmin=sig_vmin,
            vmax=sig_vmax,
            extent=extent,
            aspect="auto",
        )
        ax.set_title(title, fontsize=10)
        _style_axis(ax)

    fig.colorbar(eps_artist, ax=axes[0], shrink=0.85, label="epsilon_r")
    fig.colorbar(sig_artist, ax=axes[1], shrink=0.85, label="sigma (S/m)")
    _save_figure(fig, figures_dir / "parameter_maps")


def _save_error_maps(
    figures_dir: Path,
    *,
    epsilon_true: np.ndarray,
    sigma_true: np.ndarray,
    epsilon_initial: np.ndarray,
    sigma_initial: np.ndarray,
    epsilon_final: np.ndarray,
    sigma_final: np.ndarray,
    extent: list[float],
) -> None:
    epsilon_initial_error = np.abs(epsilon_initial - epsilon_true)
    epsilon_final_error = np.abs(epsilon_final - epsilon_true)
    sigma_initial_error = np.abs(sigma_initial - sigma_true)
    sigma_final_error = np.abs(sigma_final - sigma_true)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=200, constrained_layout=True)

    eps_vmax = float(max(np.max(epsilon_initial_error), np.max(epsilon_final_error), 1e-12))
    sig_vmax = float(max(np.max(sigma_initial_error), np.max(sigma_final_error), 1e-12))

    eps_artist = None
    sig_artist = None
    eps_titles = [
        ("|Initial - True| epsilon_r", epsilon_initial_error),
        ("|Final - True| epsilon_r", epsilon_final_error),
    ]
    sig_titles = [
        ("|Initial - True| sigma", sigma_initial_error),
        ("|Final - True| sigma", sigma_final_error),
    ]

    for ax, (title, data) in zip(axes[0], eps_titles):
        eps_artist = ax.imshow(
            data,
            cmap="inferno",
            vmin=0.0,
            vmax=eps_vmax,
            extent=extent,
            aspect="auto",
        )
        ax.set_title(title, fontsize=10)
        _style_axis(ax)

    for ax, (title, data) in zip(axes[1], sig_titles):
        sig_artist = ax.imshow(
            data,
            cmap="cividis",
            vmin=0.0,
            vmax=sig_vmax,
            extent=extent,
            aspect="auto",
        )
        ax.set_title(title, fontsize=10)
        _style_axis(ax)

    fig.colorbar(eps_artist, ax=axes[0], shrink=0.85, label="|delta epsilon_r|")
    fig.colorbar(sig_artist, ax=axes[1], shrink=0.85, label="|delta sigma| (S/m)")
    _save_figure(fig, figures_dir / "error_maps")


def _save_loss_curve(figures_dir: Path, metrics: dict[str, Any]) -> None:
    pretrain_history = metrics.get("pretrain_history", [])
    inversion_history = metrics.get("history", [])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=200, constrained_layout=True)

    _plot_history(
        axes[0],
        pretrain_history,
        title="Pretrain Loss",
        color="#1f77b4",
    )
    _plot_history(
        axes[1],
        inversion_history,
        title="Inversion Loss",
        color="#d62728",
    )
    _save_figure(fig, figures_dir / "loss_curve")


def _save_data_comparison(
    figures_dir: Path,
    *,
    observed_data: np.ndarray,
    synthetic_data: np.ndarray,
    dt: float | None,
    shot_index: int | None,
) -> None:
    observed = np.asarray(observed_data)
    synthetic = np.asarray(synthetic_data)
    if observed.shape != synthetic.shape:
        raise ValueError(
            f"observed/synthetic data shape mismatch: {observed.shape} vs {synthetic.shape}"
        )
    if observed.ndim != 3:
        raise ValueError("data comparison expects [shots, receivers, time]")

    selected_shot = int(shot_index if shot_index is not None else observed.shape[0] // 2)
    selected_shot = max(0, min(observed.shape[0] - 1, selected_shot))
    obs_shot = observed[selected_shot]
    syn_shot = synthetic[selected_shot]
    residual = syn_shot - obs_shot
    vmax = float(max(np.max(np.abs(obs_shot)), np.max(np.abs(syn_shot)), 1e-12))
    residual_vmax = float(max(np.max(np.abs(residual)), 1e-12))
    time_extent = _time_extent(obs_shot.shape, dt)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=200, constrained_layout=True)
    panels = [
        ("Observed", obs_shot, vmax),
        ("Synthetic final", syn_shot, vmax),
        ("Residual", residual, residual_vmax),
    ]
    artists = []
    for ax, (title, data, panel_vmax) in zip(axes, panels):
        artist = ax.imshow(
            data.T,
            cmap="seismic",
            vmin=-panel_vmax,
            vmax=panel_vmax,
            extent=time_extent,
            aspect="auto",
            origin="upper",
        )
        artists.append(artist)
        ax.set_title(f"{title} shot {selected_shot}", fontsize=10)
        ax.set_xlabel("Receiver index", fontsize=9)
        ax.set_ylabel("Time (ns)" if dt else "Time step", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.colorbar(artists[0], ax=axes[:2], shrink=0.82, label="Amplitude")
    fig.colorbar(artists[2], ax=axes[2], shrink=0.82, label="Residual")
    _save_figure(fig, figures_dir / "shot_gather_comparison")


def _time_extent(shape: tuple[int, int], dt: float | None) -> list[float]:
    n_receivers, n_time = shape
    receiver_min = -0.5 if n_receivers <= 1 else 0.0
    receiver_max = 0.5 if n_receivers <= 1 else float(n_receivers - 1)
    if dt is None:
        time_max = 0.5 if n_time <= 1 else float(n_time - 1)
        time_min = -0.5 if n_time <= 1 else 0.0
        return [receiver_min, receiver_max, time_max, time_min]
    max_time_ns = dt * (n_time - 1) * 1e9
    if n_time <= 1:
        return [receiver_min, receiver_max, 0.5, -0.5]
    return [receiver_min, receiver_max, max_time_ns, 0.0]


def _plot_history(ax: Any, history: list[dict[str, Any]], *, title: str, color: str) -> None:
    epochs = [int(item["epoch"]) for item in history]
    losses = [float(item["loss"]) for item in history]
    if epochs and all(loss > 0.0 for loss in losses):
        ax.semilogy(epochs, losses, marker="o", linewidth=1.5, markersize=3.5, color=color)
    else:
        ax.plot(epochs, losses, marker="o", linewidth=1.5, markersize=3.5, color=color)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)


def _style_axis(ax: Any) -> None:
    ax.set_xlabel("z (m)", fontsize=9)
    ax.set_ylabel("x (m)", fontsize=9)
    ax.tick_params(labelsize=8)


def _save_figure(fig: Any, path_stem: Path) -> None:
    fig.savefig(path_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
