"""Config-driven runner for traditional Marmousi epsilon-only inversion."""

from __future__ import annotations

import argparse
import importlib
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gpr_inversion.io import ensure_unique_dir, save_model_snapshot
from gpr_inversion.models import build_initial_model, resize_model
from gpr_inversion.visualization import save_gradient_figure, save_model_figure

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_DATA_PATH = PROJECT_ROOT.parent / "traditional" / "mode1_LBFGS" / "modelv1.bin"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "marmousi" / "traditional_eps_only"

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except Exception:

    class _DummyComm:
        def Barrier(self):
            pass

        def bcast(self, value, root=0):
            return value

        def gather(self, value, root=0):
            return [value]

        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

    comm = _DummyComm()
    rank, size = 0, 1


@dataclass(frozen=True)
class NumericSettings:
    xl: int = 100
    zl: int = 200
    dx: float = 0.02
    dz: float = 0.02
    dt: float = 4e-11
    npml: int = 10
    steps: int = 1000
    freq: float = 4e8
    eps_max: float = 6.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run traditional Marmousi epsilon-only inversion.")
    parser.add_argument("--acquisition-mode", choices=("mode1", "mode2"), default="mode1")
    parser.add_argument("--optimizer", choices=("lbfgs", "rmsprop"), default="lbfgs")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--learning-rate-eps", type=float, default=0.02)
    parser.add_argument("--alpha-tikhonov-eps", type=float, default=0.01)
    parser.add_argument("--num-epochs-stage1", type=int, default=2001)
    parser.add_argument("--fixed-top-rows", type=int, default=10)
    parser.add_argument("--snapshot-interval", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    settings = NumericSettings()
    numeric = _load_numeric_modules(args.acquisition_mode, args.optimizer)
    source_list, receiver_list = _build_geometry(args.acquisition_mode, settings.zl)

    if rank == 0:
        output_dir = ensure_unique_dir(args.output_dir)
        print(f"Results will be saved to: {output_dir}")
    else:
        output_dir = None
    output_dir = comm.bcast(str(output_dir) if rank == 0 else None, root=0)
    comm.Barrier()

    epsilon_true = _load_marmousi_epsilon(args.data_path, settings, args.fixed_top_rows)
    sigma_true = np.zeros_like(epsilon_true)
    epsilon0 = _build_initial_epsilon(epsilon_true, args.acquisition_mode, args.optimizer)
    sigma0 = np.zeros_like(epsilon0)

    if rank == 0:
        save_model_figure(epsilon_true, Path(output_dir) / "epsilon_true.png", "Epsilon True", vmin=1, vmax=6)
        save_model_figure(epsilon0, Path(output_dir) / "epsilon0.png", "Epsilon0", vmin=1, vmax=6)

    if rank == 0:
        print("Generating observed data...")
    local_d_obs = numeric.forward_model(
        epsilon_true,
        sigma_true,
        source_list,
        receiver_list,
        settings.dt,
        settings.dx,
        settings.dz,
        settings.npml,
        settings.freq,
        settings.steps,
        save_wavefield=False,
    )
    d_obs = _gather_shot_data(local_d_obs, source_list, receiver_list, settings.steps)

    if args.optimizer == "lbfgs":
        _run_lbfgs(args, settings, numeric, source_list, receiver_list, d_obs, epsilon0, sigma0, output_dir)
    else:
        _run_rmsprop(args, settings, numeric, source_list, receiver_list, d_obs, epsilon0, sigma0, output_dir)

    if rank == 0:
        print(f"Traditional Marmousi run finished. Results saved in {output_dir}.")


def _load_numeric_modules(acquisition_mode: str, optimizer: str):
    if acquisition_mode == "mode1":
        package = "gpr_inversion.experiments.marmousi.traditional_eps_only.mode1"
    elif optimizer == "lbfgs":
        package = "gpr_inversion.experiments.marmousi.traditional_eps_only.mode2_lbfgs"
    else:
        package = "gpr_inversion.experiments.marmousi.traditional_eps_only.mode2_rmsprop"
    forward = importlib.import_module(f"{package}.forward")
    gradient = importlib.import_module(f"{package}.gradient")

    class Numeric:
        forward_model = staticmethod(forward.forward_model)
        compute_gradient = staticmethod(gradient.compute_gradient)
        compute_tikhonov_gradient = staticmethod(gradient.compute_tikhonov_gradient)

    return Numeric


def _build_geometry(acquisition_mode: str, lateral_size: int):
    if acquisition_mode == "mode1":
        sources = [(0, i) for i in range(0, lateral_size, 2)]
        receivers = sources.copy()
    else:
        sources = [(0, i) for i in range(0, lateral_size, 5)]
        receivers = [(0, i) for i in range(0, lateral_size, 2)]
    return sources, receivers


def _load_marmousi_epsilon(data_path: str, settings: NumericSettings, fixed_top_rows: int) -> np.ndarray:
    raw = np.fromfile(data_path, dtype=np.float32)
    expected = 150 * 460
    if raw.size != expected:
        raise ValueError(f"Expected Marmousi binary with {expected} float32 values, got {raw.size}: {data_path}")
    model = raw.reshape(460, 150).T
    model = resize_model(model, (settings.xl, settings.zl), order=1)
    epsilon = model / np.max(model) * settings.eps_max
    if fixed_top_rows > 0:
        top_rows = np.tile(epsilon[0, :].copy(), (fixed_top_rows, 1))
        epsilon = np.vstack([top_rows, epsilon])
    if rank == 0:
        print(f"epsilon_true shape: {epsilon.shape}")
    return epsilon.astype(np.float32)


def _build_initial_epsilon(epsilon_true: np.ndarray, acquisition_mode: str, optimizer: str) -> np.ndarray:
    smooth_sigma = {
        ("mode1", "lbfgs"): 2.0,
        ("mode1", "rmsprop"): 5.0,
        ("mode2", "lbfgs"): 20.0,
        ("mode2", "rmsprop"): 20.0,
    }[(acquisition_mode, optimizer)]
    return build_initial_model(epsilon_true, "inverse_smooth", smooth_sigma=smooth_sigma).astype(np.float32)


def _gather_shot_data(local_data, source_list, receiver_list, steps: int):
    gathered = comm.gather(local_data, root=0)
    if rank == 0:
        if len(receiver_list) == len(source_list):
            data = np.zeros((len(source_list), steps), dtype=np.float32)
        else:
            data = np.zeros((len(source_list), len(receiver_list), steps), dtype=np.float32)
        cursor = 0
        for proc_data in gathered:
            if proc_data is not None and proc_data.shape[0] > 0:
                n_local = proc_data.shape[0]
                data[cursor : cursor + n_local] = proc_data
                cursor += n_local
    else:
        data = None
    return comm.bcast(data, root=0)


def _run_rmsprop(args, settings, numeric, source_list, receiver_list, d_obs, epsilon0, sigma0, output_dir):
    model_eps = epsilon0.copy()
    model_sig = sigma0.copy()
    velocity = np.zeros_like(model_eps.flatten())
    loss_history: list[float] = []
    beta_eps = 0.9

    for epoch in range(args.num_epochs_stage1):
        epoch_start = time.time()
        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.num_epochs_stage1}")
        local_d_syn, local_wavefield = numeric.forward_model(
            model_eps,
            model_sig,
            source_list,
            receiver_list,
            settings.dt,
            settings.dx,
            settings.dz,
            settings.npml,
            settings.freq,
            settings.steps,
            save_wavefield=True,
        )
        d_syn = _gather_shot_data(local_d_syn, source_list, receiver_list, settings.steps)
        residual = d_syn - d_obs
        grad_eps, _ = numeric.compute_gradient(
            model_eps,
            model_sig,
            residual,
            source_list,
            receiver_list,
            settings.dt,
            settings.dx,
            settings.dz,
            settings.npml,
            settings.freq,
            settings.steps,
            sigma_required_gradient=True,
            wavefield_data=local_wavefield,
        )
        tikh_grad_eps = numeric.compute_tikhonov_gradient(model_eps.copy(), settings.dx, settings.dz, args.alpha_tikhonov_eps)
        tikh_grad_eps[: args.fixed_top_rows, :] = 0
        grad_flat = grad_eps.flatten()
        velocity = beta_eps * velocity + (1 - beta_eps) * grad_flat**2
        update_eps = grad_flat / np.sqrt(velocity + 1e-8)
        max_update = max(np.max(np.abs(update_eps)), 1e-12)
        delta_eps = -args.learning_rate_eps * (update_eps / max_update) - tikh_grad_eps.flatten()
        model_eps = np.maximum(model_eps + delta_eps.reshape(model_eps.shape), 1.0)

        loss = float(np.linalg.norm(residual))
        _save_epoch_outputs(epoch, args, output_dir, model_eps, grad_eps, loss, loss_history, epoch_start, settings)


def _run_lbfgs(args, settings, numeric, source_list, receiver_list, d_obs, epsilon0, sigma0, output_dir):
    model_eps = epsilon0.copy()
    model_sig = sigma0.copy()
    s_history: list[np.ndarray] = []
    y_history: list[np.ndarray] = []
    rho_history: list[float] = []
    loss_history: list[float] = []
    lbfgs_m = 10
    c1 = 1e-4
    c2 = 0.9

    for epoch in range(args.num_epochs_stage1):
        epoch_start = time.time()
        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.num_epochs_stage1}")
        residual, local_wavefield, current_loss = _forward_residual(
            numeric, model_eps, model_sig, source_list, receiver_list, d_obs, settings
        )
        grad_eps, _ = numeric.compute_gradient(
            model_eps,
            model_sig,
            residual,
            source_list,
            receiver_list,
            settings.dt,
            settings.dx,
            settings.dz,
            settings.npml,
            settings.freq,
            settings.steps,
            sigma_required_gradient=True,
            wavefield_data=local_wavefield,
        )
        tikh_grad_eps = numeric.compute_tikhonov_gradient(model_eps.copy(), settings.dx, settings.dz, args.alpha_tikhonov_eps)
        tikh_grad_eps[: args.fixed_top_rows, :] = 0
        g_total = grad_eps.flatten() + tikh_grad_eps.flatten()
        prev_model_eps = model_eps.copy()
        prev_g_total = g_total.copy()
        direction = _lbfgs_direction(g_total, s_history, y_history, rho_history)
        gd = float(np.dot(prev_g_total, direction))
        if gd >= 0:
            direction = -prev_g_total
            gd = -float(np.dot(prev_g_total, prev_g_total))

        model_eps, residual, loss, new_grad_eps, new_tikh_grad_eps = _line_search_or_step(
            args,
            settings,
            numeric,
            source_list,
            receiver_list,
            d_obs,
            model_eps,
            model_sig,
            direction,
            gd,
            current_loss,
            c1,
            c2,
        )
        g_total_new = new_grad_eps.flatten() + new_tikh_grad_eps.flatten()
        s_vec = model_eps.flatten() - prev_model_eps.flatten()
        y_vec = g_total_new - prev_g_total
        den = float(np.dot(y_vec, s_vec))
        if den > 1e-12:
            s_history.append(s_vec)
            y_history.append(y_vec)
            rho_history.append(1.0 / den)
        if len(s_history) > lbfgs_m:
            s_history.pop(0)
            y_history.pop(0)
            rho_history.pop(0)
        model_eps = comm.bcast(model_eps if rank == 0 else None, root=0)
        _save_epoch_outputs(epoch, args, output_dir, model_eps, new_grad_eps, loss, loss_history, epoch_start, settings)


def _forward_residual(numeric, model_eps, model_sig, source_list, receiver_list, d_obs, settings):
    local_d_syn, local_wavefield = numeric.forward_model(
        model_eps,
        model_sig,
        source_list,
        receiver_list,
        settings.dt,
        settings.dx,
        settings.dz,
        settings.npml,
        settings.freq,
        settings.steps,
        save_wavefield=True,
    )
    d_syn = _gather_shot_data(local_d_syn, source_list, receiver_list, settings.steps)
    residual = d_syn - d_obs
    loss = float(np.linalg.norm(residual))
    return residual, local_wavefield, loss


def _line_search_or_step(args, settings, numeric, source_list, receiver_list, d_obs, model_eps, model_sig, direction, gd, current_loss, c1, c2):
    alpha = args.learning_rate_eps
    accepted = False
    for _ in range(10):
        candidate = np.maximum(model_eps + alpha * direction.reshape(model_eps.shape), 1.0)
        cand_residual, cand_wavefield, cand_loss = _forward_residual(
            numeric, candidate, model_sig, source_list, receiver_list, d_obs, settings
        )
        cand_grad_eps, _ = numeric.compute_gradient(
            candidate,
            model_sig,
            cand_residual,
            source_list,
            receiver_list,
            settings.dt,
            settings.dx,
            settings.dz,
            settings.npml,
            settings.freq,
            settings.steps,
            sigma_required_gradient=True,
            wavefield_data=cand_wavefield,
        )
        cand_tikh = numeric.compute_tikhonov_gradient(candidate.copy(), settings.dx, settings.dz, args.alpha_tikhonov_eps)
        cand_tikh[: args.fixed_top_rows, :] = 0
        gtd_new = float(np.dot(cand_grad_eps.flatten() + cand_tikh.flatten(), direction))
        accepted = cand_loss <= current_loss + c1 * alpha * gd and abs(gtd_new) <= c2 * abs(gd)
        if accepted:
            return candidate, cand_residual, cand_loss, cand_grad_eps, cand_tikh
        alpha *= 0.5

    norm_direction = direction / max(np.linalg.norm(direction), 1e-12)
    candidate = np.maximum(model_eps + args.learning_rate_eps * norm_direction.reshape(model_eps.shape), 1.0)
    residual, wavefield, loss = _forward_residual(numeric, candidate, model_sig, source_list, receiver_list, d_obs, settings)
    grad_eps, _ = numeric.compute_gradient(
        candidate,
        model_sig,
        residual,
        source_list,
        receiver_list,
        settings.dt,
        settings.dx,
        settings.dz,
        settings.npml,
        settings.freq,
        settings.steps,
        sigma_required_gradient=True,
        wavefield_data=wavefield,
    )
    tikh = numeric.compute_tikhonov_gradient(candidate.copy(), settings.dx, settings.dz, args.alpha_tikhonov_eps)
    tikh[: args.fixed_top_rows, :] = 0
    return candidate, residual, loss, grad_eps, tikh


def _lbfgs_direction(g_total, s_history, y_history, rho_history):
    if not s_history:
        return -g_total
    q = g_total.copy()
    alpha_list = []
    for idx in range(len(s_history) - 1, -1, -1):
        alpha = rho_history[idx] * np.dot(s_history[idx], q)
        q = q - alpha * y_history[idx]
        alpha_list.append(alpha)
    ys = np.dot(y_history[-1], s_history[-1])
    yy = np.dot(y_history[-1], y_history[-1])
    gamma = ys / yy if yy > 0 else 1.0
    r = gamma * q
    for idx, alpha in enumerate(reversed(alpha_list)):
        beta = rho_history[idx] * np.dot(y_history[idx], r)
        r = r + s_history[idx] * (alpha - beta)
    return -r


def _save_epoch_outputs(epoch, args, output_dir, model_eps, grad_eps, loss, loss_history, epoch_start, settings):
    loss = comm.bcast(loss if rank == 0 else None, root=0)
    if rank != 0:
        return
    loss_history.append(loss)
    if _should_save_epoch(epoch):
        save_model_figure(model_eps, Path(output_dir) / f"model_eps_epoch{epoch}.png", f"epsilon model (epoch {epoch})", vmin=1, vmax=6)
        save_gradient_figure(grad_eps, Path(output_dir) / f"grad_eps_epoch{epoch}.png", f"epsilon gradient (epoch {epoch})", settings.dx, settings.dz)
    if epoch % args.snapshot_interval == 0:
        save_model_snapshot(output_dir, epoch, model_eps)
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("FWI Loss Curve")
    plt.savefig(Path(output_dir) / "loss_curve.png")
    plt.close()
    print(f"  Epoch {epoch + 1}/{args.num_epochs_stage1} Loss: {loss:.6f} elapsed: {time.time() - epoch_start:.2f}s")


def _should_save_epoch(epoch: int) -> bool:
    return epoch in {0, 1, 2, 5, 10, 20, 50} or (epoch >= 50 and epoch % 50 == 0)


if __name__ == "__main__":
    main()
