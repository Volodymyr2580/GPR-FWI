"""Compare adjoint reverse-loop variants against ROI finite differences."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


REFERENCE_SRC = Path("E:/sci_research/GPR/marmousi_paper/gpr-inversion/src")


@dataclass(frozen=True)
class AdjointVariant:
    name: str
    ca_mode: str
    injection: str
    gradient_mode: str = "continuous"
    adjoint_shift: int = 0
    forward_shift: int = 0


VARIANTS = [
    AdjointVariant("current_ca_r_raw", "ca_r", "raw"),
    AdjointVariant("ca_forward_raw", "ca", "raw"),
    AdjointVariant("ca_one_raw", "one", "raw"),
    AdjointVariant("ca_r_cb", "ca_r", "cb"),
    AdjointVariant("ca_forward_cb", "ca", "cb"),
    AdjointVariant("ca_r_cb_dt_dxdz", "ca_r", "cb_dt_dxdz"),
    AdjointVariant("ca_forward_cb_dt_dxdz", "ca", "cb_dt_dxdz"),
    AdjointVariant("ca_r_raw_adj_minus1", "ca_r", "raw", adjoint_shift=-1),
    AdjointVariant("ca_r_raw_adj_plus1", "ca_r", "raw", adjoint_shift=1),
    AdjointVariant("ca_forward_raw_adj_minus1", "ca", "raw", adjoint_shift=-1),
    AdjointVariant("ca_forward_raw_adj_plus1", "ca", "raw", adjoint_shift=1),
    AdjointVariant("ca_r_raw_fwd_minus1", "ca_r", "raw", forward_shift=-1),
    AdjointVariant("ca_r_raw_fwd_plus1", "ca_r", "raw", forward_shift=1),
    AdjointVariant("ca_r_discrete", "ca_r", "raw", gradient_mode="discrete_ca_cb"),
    AdjointVariant(
        "ca_r_discrete_adj_plus1",
        "ca_r",
        "raw",
        gradient_mode="discrete_ca_cb",
        adjoint_shift=1,
    ),
    AdjointVariant("ca_forward_discrete", "ca", "raw", gradient_mode="discrete_ca_cb"),
    AdjointVariant(
        "ca_forward_discrete_adj_plus1",
        "ca",
        "raw",
        gradient_mode="discrete_ca_cb",
        adjoint_shift=1,
    ),
    AdjointVariant(
        "neg_ca_r_discrete_adj_plus1",
        "ca_r",
        "raw",
        gradient_mode="negative_discrete_ca_cb",
        adjoint_shift=1,
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit reverse-loop variants for the right-cross epsilon gradient."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--roi", default="right_cross", choices=["left_cross", "right_cross"])
    args = parser.parse_args()

    import sys

    reference_src_text = str(REFERENCE_SRC)
    if reference_src_text not in sys.path:
        sys.path.insert(0, reference_src_text)

    from gpr_inversion.common.Add_CPML import Add_CPML
    from gpr_inversion.experiments.overthrust.mode2_eps_then_sig.Time_loop import (
        update_E,
        update_H,
    )

    run_dir = args.run_dir
    config = load_config(run_dir / "config.json")
    model_cfg = require_section(config, "model")
    solver_cfg = require_section(config, "solver")
    shape = tuple(int(v) for v in model_cfg["shape"])
    sources, receivers = build_acquisition(require_section(config, "acquisition"), shape)
    settings = SolverSettings(
        dt=float(solver_cfg["dt"]),
        dx=float(model_cfg["dx"]),
        dz=float(model_cfg["dz"]),
        npml=int(solver_cfg["npml"]),
        freq=float(solver_cfg["freq"]),
        steps=int(solver_cfg["steps"]),
    )

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
    roi_mask = roi_masks[args.roi]
    fd_by_shot = load_roi_fd_by_shot(run_dir, args.epoch, args.roi, "epsilon")

    bridge = CpuMpiSolverBridge(solver_cfg.get("reference_project"))
    epsilon_extended = np.pad(epsilon, ((settings.npml, settings.npml), (settings.npml, settings.npml)), "edge")
    sigma_extended = np.pad(sigma, ((settings.npml, settings.npml), (settings.npml, settings.npml)), "edge")
    mu_extended = np.ones((shape[0] + 2 * settings.npml, shape[1] + 2 * settings.npml))
    cpml_params = Add_CPML(
        shape[0],
        shape[1],
        sigma_extended.copy(),
        epsilon_extended.copy(),
        mu_extended.copy(),
        settings.dx,
        settings.dz,
        settings.dt,
    )

    variant_values: dict[str, list[float]] = {variant.name: [] for variant in VARIANTS}
    losses: list[float] = []
    for shot_idx, source in enumerate(sources):
        synthetic, wavefield_data = bridge.forward(
            epsilon,
            sigma,
            [source],
            receivers,
            settings,
            save_wavefield=True,
        )
        residual = (2.0 / synthetic.size) * (synthetic - observed[shot_idx : shot_idx + 1])
        losses.append(float(np.mean((synthetic - observed[shot_idx : shot_idx + 1]) ** 2)))
        forward_wavefield = np.asarray(wavefield_data[0])
        dt_s = settings.dt
        forward_diff = (forward_wavefield[2:] - forward_wavefield[:-2]) / (2.0 * dt_s)
        shot_residual = residual[0]

        for variant in VARIANTS:
            adjoint_list = reverse_time_loop_variant(
                update_H=update_H,
                update_E=update_E,
                shape=shape,
                settings=settings,
                sigma_extended=sigma_extended,
                epsilon_extended=epsilon_extended,
                mu_extended=mu_extended,
                cpml_params=cpml_params,
                receiver_list=receivers,
                residual_data=shot_residual,
                variant=variant,
            )
            grad_eps = accumulate_epsilon_gradient(
                adjoint_list=adjoint_list,
                forward_wavefield=forward_wavefield,
                forward_diff=forward_diff,
                epsilon_extended=epsilon_extended,
                sigma_extended=sigma_extended,
                cpml_params=cpml_params,
                settings=settings,
                shape=shape,
                gradient_mode=variant.gradient_mode,
                adjoint_shift=variant.adjoint_shift,
                forward_shift=variant.forward_shift,
            )
            variant_values[variant.name].append(float(np.sum(grad_eps[roi_mask])))

    reports = []
    fd = np.array([fd_by_shot[idx] for idx in range(len(sources))], dtype=np.float64)
    for variant in VARIANTS:
        adj = np.array(variant_values[variant.name], dtype=np.float64)
        nonzero = (np.abs(fd) > 0.0) | (np.abs(adj) > 0.0)
        same = (np.sign(fd) == np.sign(adj)) & nonzero
        reports.append(
            {
                "name": variant.name,
                "ca_mode": variant.ca_mode,
                "injection": variant.injection,
                "gradient_mode": variant.gradient_mode,
                "adjoint_shift": variant.adjoint_shift,
                "forward_shift": variant.forward_shift,
                "fd_sum": float(np.sum(fd)),
                "adjoint_sum": float(np.sum(adj)),
                "same_sign_count": int(np.count_nonzero(same)),
                "nonzero_count": int(np.count_nonzero(nonzero)),
                "same_sign_fraction": float(np.count_nonzero(same) / max(np.count_nonzero(nonzero), 1)),
                "corrcoef": safe_corrcoef(fd, adj),
                "mismatch_shots": [
                    int(idx)
                    for idx, (fd_value, adj_value) in enumerate(zip(fd, adj))
                    if fd_value != 0.0 and adj_value != 0.0 and np.sign(fd_value) != np.sign(adj_value)
                ],
            }
        )

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "epoch": args.epoch,
        "roi": args.roi,
        "shot_losses": losses,
        "variants": reports,
    }
    json_path = run_dir / f"epoch_{args.epoch}_{args.roi}_epsilon_adjoint_variant_audit.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig_path = run_dir / "figures" / f"epoch{args.epoch}_{args.roi}_epsilon_adjoint_variant_audit.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_variant_summary(fd, variant_values, reports, fig_path)
    print(json.dumps({"json": str(json_path), "figure": str(fig_path)}, indent=2))


def load_roi_fd_by_shot(run_dir: Path, epoch: int, roi_name: str, parameter: str) -> dict[int, float]:
    path = run_dir / f"epoch_{epoch}_roi_gradient_fd_by_shot.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    values = {}
    for shot in data["shots"]:
        values[int(shot["shot_idx"])] = float(shot["rois"][roi_name][f"{parameter}_fd"])
    return values


def reverse_time_loop_variant(
    *,
    update_H,
    update_E,
    shape: tuple[int, int],
    settings: SolverSettings,
    sigma_extended: np.ndarray,
    epsilon_extended: np.ndarray,
    mu_extended: np.ndarray,
    cpml_params,
    receiver_list: list[tuple[int, int]],
    residual_data: np.ndarray,
    variant: AdjointVariant,
) -> list[np.ndarray]:
    ep0 = 8.841941282883074e-12
    mu0 = 1.2566370614359173e-06
    xl, zl = shape
    npml = settings.npml
    epsilon_abs = epsilon_extended.copy() * ep0
    mu_abs = mu_extended.copy() * mu0
    ey = np.zeros((xl + 2 * npml, zl + 2 * npml))
    hz = np.zeros_like(ey)
    hx = np.zeros_like(ey)
    memory_dey_dx = np.zeros((2 * npml, zl + 2 * npml))
    memory_dey_dz = np.zeros((xl + 2 * npml, 2 * npml))
    memory_dhz_dx = np.zeros((2 * npml, zl + 2 * npml))
    memory_dhx_dz = np.zeros((xl + 2 * npml, 2 * npml))

    if variant.ca_mode == "ca_r":
        ca = cpml_params.ca_r
    elif variant.ca_mode == "ca":
        ca = cpml_params.ca
    elif variant.ca_mode == "one":
        ca = np.ones_like(cpml_params.ca)
    else:
        raise ValueError(f"Unsupported ca_mode: {variant.ca_mode}")
    cb = cpml_params.cb
    receiver_list_extended = [(rec[0] + npml, rec[1] + npml) for rec in receiver_list]
    adjoint_sampled = []
    for tt in range(settings.steps):
        time_index = settings.steps - tt - 1
        for rec_idx, rec_pos in enumerate(receiver_list_extended):
            value = residual_data[rec_idx, time_index]
            if variant.injection == "raw":
                scale = 1.0
            elif variant.injection == "cb":
                scale = cb[rec_pos[0], rec_pos[1]]
            elif variant.injection == "cb_dt_dxdz":
                scale = cb[rec_pos[0], rec_pos[1]] * settings.dt / settings.dx / settings.dz
            else:
                raise ValueError(f"Unsupported injection: {variant.injection}")
            ey[rec_pos[0], rec_pos[1]] -= scale * value

        hz, hx = update_H(
            xl,
            zl,
            settings.dx,
            settings.dz,
            settings.dt,
            sigma_extended.copy(),
            epsilon_abs,
            mu_abs,
            npml,
            cpml_params.a_x_half,
            cpml_params.a_z_half,
            cpml_params.b_x_half,
            cpml_params.b_z_half,
            cpml_params.k_x_half,
            cpml_params.k_z_half,
            hz,
            hx,
            ey,
            memory_dey_dx,
            memory_dey_dz,
        )
        ey = update_E(
            xl,
            zl,
            settings.dx,
            settings.dz,
            settings.dt,
            ca,
            cb,
            npml,
            cpml_params.a_x,
            cpml_params.a_z,
            cpml_params.b_x,
            cpml_params.b_z,
            cpml_params.k_x,
            cpml_params.k_z,
            hz,
            hx,
            ey,
            memory_dhz_dx,
            memory_dhx_dz,
        )
        adjoint_sampled.append(ey.copy())
    return adjoint_sampled[::-1]


def accumulate_epsilon_gradient(
    *,
    adjoint_list: list[np.ndarray],
    forward_wavefield: np.ndarray,
    forward_diff: np.ndarray,
    epsilon_extended: np.ndarray,
    sigma_extended: np.ndarray,
    cpml_params,
    settings: SolverSettings,
    shape: tuple[int, int],
    gradient_mode: str,
    adjoint_shift: int,
    forward_shift: int,
) -> np.ndarray:
    ep0 = 8.841941282883074e-12
    xl, zl = shape
    npml = settings.npml
    grad_eps = np.zeros((xl, zl), dtype=np.float64)
    epsilon_abs = epsilon_extended * ep0
    denominator = epsilon_abs + sigma_extended * settings.dt / 2.0
    dca_deps = ep0 * sigma_extended * settings.dt / (denominator * denominator)
    dcb_deps = -ep0 / (denominator * denominator)
    for k in range(1, settings.steps - 1):
        adj_idx = k + adjoint_shift
        fwd_idx = k - 1 + forward_shift
        if adj_idx < 0 or adj_idx >= len(adjoint_list):
            continue
        if fwd_idx < 0 or fwd_idx >= len(forward_diff):
            continue
        adjoint_field = adjoint_list[adj_idx]
        if gradient_mode == "continuous":
            contribution = ep0 * adjoint_field * forward_diff[fwd_idx]
        elif gradient_mode in {"discrete_ca_cb", "negative_discrete_ca_cb"}:
            wavefield_index = fwd_idx + 1
            if wavefield_index <= 0 or wavefield_index >= len(forward_wavefield):
                continue
            e_old = forward_wavefield[wavefield_index - 1]
            e_new = forward_wavefield[wavefield_index]
            curl_h_dt = (e_new - cpml_params.ca * e_old) / cpml_params.cb
            local_update_derivative = dca_deps * e_old + dcb_deps * curl_h_dt
            contribution = adjoint_field * local_update_derivative
            if gradient_mode == "negative_discrete_ca_cb":
                contribution = -contribution
        else:
            raise ValueError(f"Unsupported gradient_mode: {gradient_mode}")
        grad_eps += contribution[npml : npml + xl, npml : npml + zl]
    mask = np.ones_like(grad_eps)
    mask[:10, :] = 0.0
    return grad_eps * mask


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 0.0 or np.std(b) <= 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def plot_variant_summary(
    fd: np.ndarray,
    variant_values: dict[str, list[float]],
    reports: list[dict[str, Any]],
    output_path: Path,
) -> None:
    best_by_corr = sorted(reports, key=lambda item: item["corrcoef"], reverse=True)[:4]
    best_by_sign = sorted(
        reports,
        key=lambda item: (item["same_sign_count"], item["corrcoef"]),
        reverse=True,
    )[:4]
    selected_names = []
    for item in best_by_sign + best_by_corr:
        if item["name"] not in selected_names:
            selected_names.append(item["name"])

    shot_indices = np.arange(fd.size)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.plot(shot_indices, fd, color="black", marker="o", label="finite difference")
    for name in selected_names[:5]:
        ax.plot(shot_indices, variant_values[name], marker="x", linewidth=1.2, label=name)
    ax.set_title("ROI epsilon derivative by shot")
    ax.set_xlabel("shot index")
    ax.set_ylabel("d loss / d ROI")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    names = [item["name"] for item in reports]
    same = [item["same_sign_count"] for item in reports]
    corr = [item["corrcoef"] for item in reports]
    y = np.arange(len(names))
    ax.barh(y, same, color="tab:blue", alpha=0.75, label="same sign count")
    ax2 = ax.twiny()
    ax2.plot(corr, y, color="tab:red", marker="o", linestyle="", label="corrcoef")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("same sign count / 32")
    ax2.set_xlabel("corrcoef")
    ax.set_title("Variant ranking")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
