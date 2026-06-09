"""ROI finite-difference audits for Cross-shape IFWI gradients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ifwi_gpr.acquisition import build_acquisition
from ifwi_gpr.config import load_config, require_section
from ifwi_gpr.models import build_cross_shape_model
from ifwi_gpr.solver_bridge import CpuMpiSolverBridge, SolverSettings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare epsilon/sigma ROI adjoint gradients against finite differences by shot."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--delta-epsilon", type=float, default=0.05)
    parser.add_argument("--delta-sigma", type=float, default=5e-5)
    args = parser.parse_args()

    run_dir = args.run_dir
    config = load_config(run_dir / "config.json")
    model_cfg = require_section(config, "model")
    solver_cfg = require_section(config, "solver")
    shape = tuple(int(v) for v in model_cfg["shape"])
    sources, receivers = build_acquisition(require_section(config, "acquisition"), shape)

    epsilon = np.load(run_dir / f"epoch_{args.epoch}_epsilon.npy")
    sigma = np.load(run_dir / f"epoch_{args.epoch}_sigma.npy")
    observed = np.load(run_dir / "observed_data.npy")
    true_epsilon, true_sigma = build_cross_shape_model(
        shape,
        epsilon_background=float(model_cfg.get("epsilon_background", 4.0)),
        sigma_background=float(model_cfg.get("sigma_background", 0.003)),
    )
    roi_masks = {
        "left_cross": (true_epsilon == 1.0) | (true_sigma == 0.0001),
        "right_cross": (true_epsilon == 8.0) | (true_sigma == 0.01),
    }

    settings = SolverSettings(
        dt=float(solver_cfg["dt"]),
        dx=float(model_cfg["dx"]),
        dz=float(model_cfg["dz"]),
        npml=int(solver_cfg["npml"]),
        freq=float(solver_cfg["freq"]),
        steps=int(solver_cfg["steps"]),
    )
    bridge = CpuMpiSolverBridge(solver_cfg.get("reference_project"))

    reports = []
    for shot_idx, source in enumerate(sources):
        reports.append(
            audit_shot_rois(
                bridge=bridge,
                settings=settings,
                epsilon=epsilon,
                sigma=sigma,
                observed=observed[shot_idx : shot_idx + 1],
                source=source,
                receivers=receivers,
                shot_idx=shot_idx,
                roi_masks=roi_masks,
                delta_epsilon=args.delta_epsilon,
                delta_sigma=args.delta_sigma,
            )
        )

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "epoch": args.epoch,
        "delta_epsilon": args.delta_epsilon,
        "delta_sigma": args.delta_sigma,
        "shots": reports,
    }
    json_path = run_dir / f"epoch_{args.epoch}_roi_gradient_fd_by_shot.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig_path = run_dir / "figures" / f"epoch{args.epoch}_roi_gradient_fd_by_shot.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_roi_reports(reports, fig_path)
    print(json.dumps({"json": str(json_path), "figure": str(fig_path)}, indent=2))


def audit_shot_rois(
    *,
    bridge: CpuMpiSolverBridge,
    settings: SolverSettings,
    epsilon: np.ndarray,
    sigma: np.ndarray,
    observed: np.ndarray,
    source: tuple[int, int],
    receivers: list[tuple[int, int]],
    shot_idx: int,
    roi_masks: dict[str, np.ndarray],
    delta_epsilon: float,
    delta_sigma: float,
) -> dict[str, Any]:
    synthetic, wavefield_data = bridge.forward(
        epsilon,
        sigma,
        [source],
        receivers,
        settings,
        save_wavefield=True,
    )
    residual = (2.0 / synthetic.size) * (synthetic - observed)
    grad_eps, grad_sig = bridge.gradient(
        epsilon,
        sigma,
        residual,
        [source],
        receivers,
        settings,
        wavefield_data=wavefield_data,
        sigma_required_gradient=True,
    )

    roi_reports: dict[str, Any] = {}
    for roi_name, roi_mask in roi_masks.items():
        eps_fd = central_roi_fd(
            bridge=bridge,
            settings=settings,
            epsilon=epsilon,
            sigma=sigma,
            observed=observed,
            source=source,
            receivers=receivers,
            roi_mask=roi_mask,
            parameter="epsilon",
            delta=delta_epsilon,
        )
        sig_fd = central_roi_fd(
            bridge=bridge,
            settings=settings,
            epsilon=epsilon,
            sigma=sigma,
            observed=observed,
            source=source,
            receivers=receivers,
            roi_mask=roi_mask,
            parameter="sigma",
            delta=delta_sigma,
        )
        roi_reports[roi_name] = {
            "epsilon_fd": eps_fd,
            "epsilon_adjoint_sum": float(np.sum(grad_eps[roi_mask])),
            "sigma_fd": sig_fd,
            "sigma_adjoint_sum": float(np.sum(grad_sig[roi_mask])),
            "cell_count": int(np.count_nonzero(roi_mask)),
        }
    return {
        "shot_idx": shot_idx,
        "source": list(source),
        "loss": float(np.mean((synthetic - observed) ** 2)),
        "rois": roi_reports,
    }


def central_roi_fd(
    *,
    bridge: CpuMpiSolverBridge,
    settings: SolverSettings,
    epsilon: np.ndarray,
    sigma: np.ndarray,
    observed: np.ndarray,
    source: tuple[int, int],
    receivers: list[tuple[int, int]],
    roi_mask: np.ndarray,
    parameter: str,
    delta: float,
) -> float:
    eps_plus = epsilon.copy()
    eps_minus = epsilon.copy()
    sig_plus = sigma.copy()
    sig_minus = sigma.copy()
    if parameter == "epsilon":
        eps_plus[roi_mask] += delta
        eps_minus[roi_mask] -= delta
    elif parameter == "sigma":
        sig_plus[roi_mask] += delta
        sig_minus[roi_mask] -= delta
    else:
        raise ValueError(f"Unsupported parameter: {parameter}")
    plus = bridge.forward(eps_plus, sig_plus, [source], receivers, settings, save_wavefield=False)
    minus = bridge.forward(eps_minus, sig_minus, [source], receivers, settings, save_wavefield=False)
    loss_plus = float(np.mean((plus - observed) ** 2))
    loss_minus = float(np.mean((minus - observed) ** 2))
    return (loss_plus - loss_minus) / (2.0 * delta)


def plot_roi_reports(reports: list[dict[str, Any]], output_path: Path) -> None:
    shot_indices = np.array([item["shot_idx"] for item in reports], dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for row, roi_name in enumerate(["left_cross", "right_cross"]):
        for col, parameter in enumerate(["epsilon", "sigma"]):
            fd = np.array([item["rois"][roi_name][f"{parameter}_fd"] for item in reports])
            adj = np.array(
                [item["rois"][roi_name][f"{parameter}_adjoint_sum"] for item in reports]
            )
            ax = axes[row, col]
            ax.axhline(0.0, color="0.4", linewidth=0.8)
            ax.plot(shot_indices, fd, marker="o", label="finite difference")
            ax.plot(shot_indices, adj, marker="x", label="adjoint sum")
            matches = np.sign(fd) == np.sign(adj)
            ax.scatter(
                shot_indices[~matches],
                fd[~matches],
                s=55,
                facecolors="none",
                edgecolors="red",
                label="sign mismatch" if row == 0 and col == 0 else None,
            )
            ax.set_title(f"{roi_name} {parameter}")
            ax.set_ylabel("d loss / d ROI")
            ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("shot index")
    axes[-1, 1].set_xlabel("shot index")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
