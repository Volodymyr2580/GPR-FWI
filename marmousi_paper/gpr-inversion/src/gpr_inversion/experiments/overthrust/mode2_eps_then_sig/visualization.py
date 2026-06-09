"""Plotting helpers for paper-ready model and diagnostic figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable


def normalize_gradient_for_visualization(grad):
    """Normalize a gradient to [-1, 1] for plotting and optional saving."""
    grad = np.asarray(grad, dtype=np.float32)
    max_abs = np.max(np.abs(grad))
    if max_abs > 0:
        grad_norm = grad / max_abs
    else:
        grad_norm = grad.copy()
    return grad_norm, max_abs


def save_gradient_figure(grad_data, file_path, title, dx=0.02, dz=0.02):
    """Save a normalized gradient map."""
    nz, nx = grad_data.shape
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    im = ax.imshow(
        grad_data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        extent=[0, nx * dx, nz * dz, 0],
    )
    ax.set_title(title)
    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Depth (m)")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Normalized Gradient")
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()


def save_model_figure(
    model_data,
    file_path,
    title,
    is_sigma=False,
    dx=0.02,
    dz=0.02,
    vmin=None,
    vmax=None,
):
    """Save epsilon or sigma model figures with consistent paper-style axes."""
    plt.rcParams["font.family"] = "serif"

    if isinstance(model_data, torch.Tensor):
        model_data = model_data.detach().cpu().numpy()

    nz, nx = model_data.shape
    x_length = nx * dx
    z_length = nz * dz

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    if vmin is None:
        vmin = np.min(model_data) if is_sigma else np.percentile(model_data, 2)
    if vmax is None:
        vmax = np.max(model_data) if is_sigma else np.percentile(model_data, 98)

    im = ax.imshow(
        model_data,
        aspect="auto",
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        extent=[0, x_length, z_length, 0],
    )

    ax.set_title(title)
    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Depth (m)")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

    if is_sigma:
        cbar.set_label(r"$\sigma$ (S/m)")
    else:
        cbar.set_label(r"$\epsilon_r$")

    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()


def save_residual_figure(
    residual_data,
    file_path,
    title,
    label,
    dx=0.02,
    dz=0.02,
):
    """Save a zero-centered residual map for inverted-minus-true models."""
    plt.rcParams["font.family"] = "serif"

    if isinstance(residual_data, torch.Tensor):
        residual_data = residual_data.detach().cpu().numpy()

    nz, nx = residual_data.shape
    x_length = nx * dx
    z_length = nz * dz
    max_abs = np.max(np.abs(residual_data))
    if max_abs == 0:
        max_abs = 1.0

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    im = ax.imshow(
        residual_data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
        extent=[0, x_length, z_length, 0],
    )
    ax.set_title(title)
    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Depth (m)")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label)

    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_loss_curves(loss_history, eps_loss_history, save_path, current_epoch):
    """Save the full, recent, and epsilon-model loss curves."""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(loss_history, "b-", linewidth=2)
    plt.title("Full Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    recent_epochs = min(100, len(loss_history))
    if recent_epochs > 0:
        plt.plot(
            range(len(loss_history) - recent_epochs, len(loss_history)),
            loss_history[-recent_epochs:],
            "r-",
            linewidth=2,
        )
        plt.title(f"Recent {recent_epochs} Epochs Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(eps_loss_history, "g-", linewidth=2)
    plt.title("Epsilon Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    output_file = Path(save_path) / f"loss_curves_epoch_{current_epoch}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
