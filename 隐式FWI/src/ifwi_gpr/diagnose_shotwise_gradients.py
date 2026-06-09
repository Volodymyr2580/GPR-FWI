"""Shot-wise finite-difference audits for the GPR IFWI epsilon gradient."""

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
from ifwi_gpr.solver_bridge import CpuMpiSolverBridge, SolverSettings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare shot-wise epsilon adjoint gradients against block finite differences."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--shots", nargs="+", type=int, required=True)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args()

    run_dir = args.run_dir
    config = load_config(run_dir / "config.json")
    model_cfg = require_section(config, "model")
    solver_cfg = require_section(config, "solver")
    shape = tuple(int(v) for v in model_cfg["shape"])

    epsilon = np.load(run_dir / f"epoch_{args.epoch}_epsilon.npy")
    sigma = np.load(run_dir / f"epoch_{args.epoch}_sigma.npy")
    observed = np.load(run_dir / "observed_data.npy")
    sources, receivers = build_acquisition(require_section(config, "acquisition"), shape)
    settings = SolverSettings(
        dt=float(solver_cfg["dt"]),
        dx=float(model_cfg["dx"]),
        dz=float(model_cfg["dz"]),
        npml=int(solver_cfg["npml"]),
        freq=float(solver_cfg["freq"]),
        steps=int(solver_cfg["steps"]),
    )
    bridge = CpuMpiSolverBridge(solver_cfg.get("reference_project"))

    shot_reports: list[dict[str, Any]] = []
    out_prefix = run_dir / f"epoch_{args.epoch}_shotwise_epsilon_fd"
    for shot_idx in args.shots:
        report = audit_one_shot(
            bridge=bridge,
            settings=settings,
            epsilon=epsilon,
            sigma=sigma,
            observed=observed[shot_idx : shot_idx + 1],
            source=sources[shot_idx],
            receivers=receivers,
            shot_idx=shot_idx,
            blocks=args.blocks,
            delta=args.delta,
        )
        shot_reports.append(report)
        np.save(
            run_dir / f"epoch_{args.epoch}_shot_{shot_idx}_epsilon_block_fd.npy",
            report["fd_grid"],
        )
        np.save(
            run_dir / f"epoch_{args.epoch}_shot_{shot_idx}_epsilon_block_adjoint.npy",
            report["adjoint_grid"],
        )

    serializable = []
    for report in shot_reports:
        serializable.append(
            {
                key: value
                for key, value in report.items()
                if key not in {"fd_grid", "adjoint_grid"}
            }
        )
    summary = {
        "run_dir": str(run_dir),
        "epoch": args.epoch,
        "blocks": args.blocks,
        "delta": args.delta,
        "shots": serializable,
    }
    json_path = out_prefix.with_name(f"{out_prefix.name}_shots_{'_'.join(map(str, args.shots))}.json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig_path = run_dir / "figures" / (
        f"epoch{args.epoch}_shotwise_epsilon_fd_vs_adjoint_"
        f"{'_'.join(map(str, args.shots))}.png"
    )
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_shot_reports(shot_reports, fig_path)
    print(json.dumps({"json": str(json_path), "figure": str(fig_path)}, indent=2))


def audit_one_shot(
    *,
    bridge: CpuMpiSolverBridge,
    settings: SolverSettings,
    epsilon: np.ndarray,
    sigma: np.ndarray,
    observed: np.ndarray,
    source: tuple[int, int],
    receivers: list[tuple[int, int]],
    shot_idx: int,
    blocks: int,
    delta: float,
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
    grad_eps, _ = bridge.gradient(
        epsilon,
        sigma,
        residual,
        [source],
        receivers,
        settings,
        wavefield_data=wavefield_data,
        sigma_required_gradient=False,
    )

    slices = make_block_slices(epsilon.shape, blocks)
    fd_grid = np.zeros((blocks, blocks), dtype=np.float64)
    adjoint_grid = np.zeros((blocks, blocks), dtype=np.float64)
    for ix, x_slice in enumerate(slices[0]):
        for iz, z_slice in enumerate(slices[1]):
            eps_plus = epsilon.copy()
            eps_minus = epsilon.copy()
            eps_plus[x_slice, z_slice] += delta
            eps_minus[x_slice, z_slice] -= delta
            plus = bridge.forward(
                eps_plus,
                sigma,
                [source],
                receivers,
                settings,
                save_wavefield=False,
            )
            minus = bridge.forward(
                eps_minus,
                sigma,
                [source],
                receivers,
                settings,
                save_wavefield=False,
            )
            loss_plus = float(np.mean((plus - observed) ** 2))
            loss_minus = float(np.mean((minus - observed) ** 2))
            fd_grid[ix, iz] = (loss_plus - loss_minus) / (2.0 * delta)
            adjoint_grid[ix, iz] = float(np.sum(grad_eps[x_slice, z_slice]))

    same_sign = np.sign(fd_grid) == np.sign(adjoint_grid)
    nonzero = (np.abs(fd_grid) > 0.0) | (np.abs(adjoint_grid) > 0.0)
    corrcoef = safe_corrcoef(fd_grid, adjoint_grid)
    return {
        "shot_idx": shot_idx,
        "source": list(source),
        "loss": float(np.mean((synthetic - observed) ** 2)),
        "same_sign_count": int(np.count_nonzero(same_sign & nonzero)),
        "nonzero_count": int(np.count_nonzero(nonzero)),
        "same_sign_fraction": float(np.count_nonzero(same_sign & nonzero) / max(np.count_nonzero(nonzero), 1)),
        "corrcoef": corrcoef,
        "fd_grid": fd_grid,
        "adjoint_grid": adjoint_grid,
    }


def make_block_slices(shape: tuple[int, int], blocks: int) -> tuple[list[slice], list[slice]]:
    x_edges = np.linspace(0, shape[0], blocks + 1, dtype=int)
    z_edges = np.linspace(0, shape[1], blocks + 1, dtype=int)
    x_slices = [slice(int(x_edges[i]), int(x_edges[i + 1])) for i in range(blocks)]
    z_slices = [slice(int(z_edges[i]), int(z_edges[i + 1])) for i in range(blocks)]
    return x_slices, z_slices


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = np.asarray(a, dtype=np.float64).ravel()
    b_flat = np.asarray(b, dtype=np.float64).ravel()
    if np.std(a_flat) <= 0.0 or np.std(b_flat) <= 0.0:
        return float("nan")
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def plot_shot_reports(reports: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(len(reports), 3, figsize=(10, 3.1 * len(reports)), squeeze=False)
    for row, report in enumerate(reports):
        fd_grid = report["fd_grid"]
        adjoint_grid = report["adjoint_grid"]
        sign_match = np.sign(fd_grid) == np.sign(adjoint_grid)
        vmax = max(float(np.max(np.abs(fd_grid))), float(np.max(np.abs(adjoint_grid))), 1e-20)
        for col, (title, values, cmap, vmin, vmax_use) in enumerate(
            [
                ("FD", fd_grid, "coolwarm", -vmax, vmax),
                ("Adjoint", adjoint_grid, "coolwarm", -vmax, vmax),
                ("Sign match", sign_match.astype(float), "gray", 0.0, 1.0),
            ]
        ):
            ax = axes[row, col]
            im = ax.imshow(values.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax_use)
            ax.set_title(f"shot {report['shot_idx']} {title}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes[row, 0].set_ylabel(
            f"src={tuple(report['source'])}\n"
            f"same={report['same_sign_count']}/{report['nonzero_count']}, "
            f"r={report['corrcoef']:.2f}"
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
