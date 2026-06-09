#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Unet重参数化的MPI版本全波形反演程序
基于gpr_train.py的主体框架，将数据生成、正演模拟和梯度计算并行化
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Function
import torch.utils.data as data_utils
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, uniform_filter, zoom

from .dataset import ShotDataset
from .forward import forward_model
from .gradient import compute_gradient
from .regularization import (
    combine_data_and_tv_gradient,
    compute_tv_gradient,
    normalize_torch_gradient,
    tikhonov_reg,
)
from .runtime import (
    ensure_unique_dir,
    get_parameter_number,
    same_seeds,
    sanitize_inversion_gradient,
)
from .unet import UNet
from .visualization import (
    normalize_gradient_for_visualization,
    plot_loss_curves,
    save_gradient_figure,
    save_model_figure,
    save_residual_figure,
)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "overthrust" / "OverThrust.npy"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "overthrust" / "mode2_eps_then_sig"

# ==== MPI 初始化 ====
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    MPI_AVAILABLE = True
    if rank == 0:
        print(f"MPI初始化成功，进程数: {size}")
except Exception:
    # 兼容无 MPI 环境（串行）
    class _DummyComm:
        def Barrier(self):
            pass
        def bcast(self, x, root=0):
            return x
        def gather(self, x, root=0):
            return [x]
        def Get_rank(self):
            return 0
        def Get_size(self):
            return 1
    comm = _DummyComm()
    rank, size = 0, 1
    MPI_AVAILABLE = False
    print("MPI不可用，运行串行模式")

# 设置PyTorch默认数据类型
torch.set_default_dtype(torch.float32)
L_SIZE = 10

def _local_shot_slice(n_shots):
    if MPI_AVAILABLE and 'comm' in globals() and comm.Get_size() > 1:
        size_local = comm.Get_size()
        rank_local = comm.Get_rank()
    else:
        size_local = 1
        rank_local = 0
    shots_per_proc = n_shots // size_local
    remainder = n_shots % size_local
    if rank_local < remainder:
        start_idx = rank_local * (shots_per_proc + 1)
        end_idx = start_idx + shots_per_proc + 1
    else:
        start_idx = rank_local * shots_per_proc + remainder
        end_idx = start_idx + shots_per_proc
    return start_idx, end_idx


def _make_jax_case(epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps):
    from jax_fdtd_lab.cases import TinyFdtdCase

    return TinyFdtdCase(
        epsilon=np.asarray(epsilon, dtype=np.float32),
        sigma=np.asarray(sigma, dtype=np.float32),
        sources=tuple((int(x), int(z)) for x, z in source_list),
        receivers=tuple((int(x), int(z)) for x, z in receiver_list),
        dt=float(dt),
        dx=float(dx),
        dz=float(dz),
        npml=int(npml),
        freq=float(freq),
        steps=int(steps),
    )


def _jax_shot_batch_size():
    """Return the JAX source batch size used for memory-heavy wavefield paths."""
    value = os.environ.get("JAX_FDTD_SHOT_BATCH_SIZE", "5")
    try:
        return int(value)
    except ValueError:
        if rank == 0:
            print(f"忽略非法的 JAX_FDTD_SHOT_BATCH_SIZE={value!r}，使用默认值 5")
        return 5


def _resolve_rank_device(requested_device):
    if requested_device != "auto":
        return requested_device
    if not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{rank % max(torch.cuda.device_count(), 1)}"


def _estimate_jax_shot_batch_size(device, max_batch_size):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return 1
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(torch.device(device))
    except Exception:
        return max(1, min(max_batch_size, 2))

    free_gb = free_bytes / (1024**3)
    reserve_gb = 4.0
    estimated_gb_per_shot = 2.0
    estimated = int(max(1.0, (free_gb - reserve_gb) / estimated_gb_per_shot))
    return max(1, min(max_batch_size, estimated))


def _jax_forward_numpy(epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps):
    if len(source_list) == 0:
        return np.zeros((0, len(receiver_list), steps), dtype=np.float32)
    from jax_fdtd_lab.jax_forward import run_jax_forward

    case = _make_jax_case(epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps)
    return np.asarray(run_jax_forward(case), dtype=np.float32)


def _jax_forward_with_illumination_numpy(epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps):
    if len(source_list) == 0:
        return (
            np.zeros((0, len(receiver_list), steps), dtype=np.float32),
            np.zeros_like(epsilon, dtype=np.float32),
        )
    from jax_fdtd_lab.jax_forward import run_jax_forward_with_illumination_batched

    case = _make_jax_case(epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps)
    data, illumination = run_jax_forward_with_illumination_batched(
        case,
        shot_batch_size=_jax_shot_batch_size(),
    )
    return np.asarray(data, dtype=np.float32), np.asarray(illumination, dtype=np.float32)


def _jax_adjoint_gradient_numpy(
    epsilon,
    sigma,
    residual,
    source_list,
    receiver_list,
    dt,
    dx,
    dz,
    npml,
    freq,
    steps,
    sigma_required_gradient,
):
    if len(source_list) == 0:
        grad_eps = np.zeros_like(epsilon, dtype=np.float32)
        grad_sig = np.zeros_like(sigma, dtype=np.float32) if sigma_required_gradient else None
        return grad_eps, grad_sig
    from jax_fdtd_lab.jax_adjoint import run_jax_adjoint_gradient_batched

    case = _make_jax_case(epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps)
    return run_jax_adjoint_gradient_batched(
        case,
        np.asarray(residual, dtype=np.float32),
        sigma_required_gradient=sigma_required_gradient,
        shot_batch_size=_jax_shot_batch_size(),
    )


class ForwardModelFunction(Function):
    """
    自定义自动微分函数，包装FDTD正演模拟
    """
    @staticmethod
    def forward(ctx, epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps, alpha_eps, alpha_sig, alpha_tv_eps, alpha_tv_sig, sigma_beta, sigma_tv_domain, sigma_required_gradient, illumination_enabled, fdtd_backend):
        """
        前向传播：执行FDTD正演模拟
        """
        # 将torch张量转换为numpy数组用于FDTD计算
        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()
        active_source_list = source_list
        wave_field = None
        illumination = None
        if fdtd_backend == "jax":
            start_idx, end_idx = _local_shot_slice(len(source_list))
            active_source_list = source_list[start_idx:end_idx]
            if illumination_enabled:
                data, local_illumination = _jax_forward_with_illumination_numpy(
                    epsilon_np,
                    sigma_np,
                    active_source_list,
                    receiver_list,
                    dt,
                    dx,
                    dz,
                    npml,
                    freq,
                    steps,
                )
                illumination = local_illumination
                if MPI_AVAILABLE and 'comm' in globals() and comm.Get_size() > 1:
                    global_illumination = np.empty_like(local_illumination)
                    comm.Allreduce(local_illumination, global_illumination, op=MPI.SUM)
                    illumination = global_illumination
            else:
                data = _jax_forward_numpy(
                    epsilon_np,
                    sigma_np,
                    active_source_list,
                    receiver_list,
                    dt,
                    dx,
                    dz,
                    npml,
                    freq,
                    steps,
                )
        elif illumination_enabled:
            data, wave_field, illumination = forward_model(
                epsilon_np,
                sigma_np,
                source_list,
                receiver_list,
                dt,
                dx,
                dz,
                npml,
                freq,
                steps,
                save_wavefield=True,
                return_illumination=True,
            )
        else:
            data, wave_field = forward_model(
                epsilon_np,
                sigma_np,
                source_list,
                receiver_list,
                dt,
                dx,
                dz,
                npml,
                freq,
                steps,
                save_wavefield=True,
            )

        data_tensor = torch.from_numpy(np.asarray(data, dtype=np.float32).copy()).to(epsilon.device).float()
        
        # 保存用于反向传播的信息
        ctx.save_for_backward(epsilon, sigma)
        ctx.source_list = active_source_list
        ctx.receiver_list = receiver_list
        ctx.dt = dt
        ctx.dx = dx
        ctx.dz = dz
        ctx.npml = npml
        ctx.freq = freq
        ctx.steps = steps
        ctx.wave_field = wave_field
        ctx.alpha_eps = alpha_eps
        ctx.alpha_sig = alpha_sig
        ctx.alpha_tv_eps = alpha_tv_eps
        ctx.alpha_tv_sig = alpha_tv_sig
        ctx.sigma_beta = sigma_beta
        ctx.sigma_tv_domain = sigma_tv_domain
        ctx.sigma_required_gradient = sigma_required_gradient
        ctx.illumination_enabled = illumination_enabled
        ctx.illumination = illumination
        ctx.fdtd_backend = fdtd_backend

        return data_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：计算梯度
        """
        epsilon, sigma = ctx.saved_tensors
        source_list = ctx.source_list
        receiver_list = ctx.receiver_list
        dt = ctx.dt
        dx = ctx.dx
        dz = ctx.dz
        npml = ctx.npml
        freq = ctx.freq
        steps = ctx.steps
        wave_field = ctx.wave_field
        alpha_eps = ctx.alpha_eps
        alpha_sig = ctx.alpha_sig
        alpha_tv_eps = ctx.alpha_tv_eps
        alpha_tv_sig = ctx.alpha_tv_sig
        sigma_beta = ctx.sigma_beta
        sigma_tv_domain = ctx.sigma_tv_domain
        
        # 将torch张量转换为numpy数组
        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()
        grad_output_np = grad_output.detach().cpu().numpy()
        # 计算本地梯度。JAX backend 使用手写伴随梯度，不通过 JAX 计算图反传。
        if getattr(ctx, 'fdtd_backend', 'numpy') == "jax":
            grad_eps, grad_sig = _jax_adjoint_gradient_numpy(
                epsilon_np,
                sigma_np,
                grad_output_np,
                source_list,
                receiver_list,
                dt,
                dx,
                dz,
                npml,
                freq,
                steps,
                sigma_required_gradient=ctx.sigma_required_gradient,
            )
        else:
            grad_eps, grad_sig = compute_gradient(
                epsilon_np,
                sigma_np,
                grad_output_np,
                source_list,
                receiver_list,
                dt,
                dx,
                dz,
                npml,
                freq,
                steps,
                sigma_required_gradient=ctx.sigma_required_gradient,
                wavefield_data=wave_field,
            )

        if getattr(ctx, 'illumination_enabled', False) and ctx.illumination is not None:
            H = ctx.illumination
            H_max = np.max(H)
            denom = H + (1e-8 * H_max if H_max > 0 else 1e-8)
            grad_eps = grad_eps / denom
            if grad_sig is not None:
                grad_sig = grad_sig / denom

        # 转换回torch张量
        grad_eps_tensor = torch.from_numpy(np.asarray(grad_eps, dtype=np.float32).copy()).to(epsilon.device)
        grad_sig_tensor = torch.from_numpy(np.asarray(grad_sig, dtype=np.float32).copy()).to(sigma.device) if grad_sig is not None else None

        # 在MPI环境下对梯度做Allreduce求和，汇总所有进程的贡献
        try:
            if MPI_AVAILABLE and 'comm' in globals() and comm.Get_size() > 1:
                # eps梯度
                grad_eps_np_local = grad_eps_tensor.detach().cpu().numpy()
                grad_eps_np_global = np.empty_like(grad_eps_np_local)
                comm.Allreduce(grad_eps_np_local, grad_eps_np_global, op=MPI.SUM)
                grad_eps_np_global = grad_eps_np_global / float(comm.Get_size())
                grad_eps_tensor = torch.from_numpy(grad_eps_np_global).to(epsilon.device)

                # sig梯度（若存在）
                if grad_sig_tensor is not None:
                    grad_sig_np_local = grad_sig_tensor.detach().cpu().numpy()
                    grad_sig_np_global = np.empty_like(grad_sig_np_local)
                    comm.Allreduce(grad_sig_np_local, grad_sig_np_global, op=MPI.SUM)
                    grad_sig_np_global = grad_sig_np_global / float(comm.Get_size())
                    grad_sig_tensor = torch.from_numpy(grad_sig_np_global).to(sigma.device)
        except Exception:
            # 若无MPI或出错，则退化为本地梯度
            pass

        grad_eps = combine_data_and_tv_gradient(
            grad_eps_tensor,
            epsilon_np,
            dx,
            dz,
            alpha_data=alpha_eps,
            alpha_tv=alpha_tv_eps,
            device=epsilon.device,
        )
        
        if grad_sig_tensor is not None:
            grad_sig = combine_sigma_data_and_tv_gradient(
                grad_sig_tensor,
                sigma_np,
                dx,
                dz,
                alpha_data=alpha_sig,
                alpha_tv=alpha_tv_sig,
                beta=sigma_beta,
                tv_domain=sigma_tv_domain,
                device=sigma.device,
            )
        else:
            grad_sig = None

        return grad_eps, grad_sig, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

# ========== 辅助函数 ==========
def grad_hook(grad):
    """梯度处理钩子函数：固定浅层、归一化并裁剪异常梯度。"""
    return sanitize_inversion_gradient(grad, fixed_top_rows=L_SIZE)


def combine_sigma_data_and_tv_gradient(
    data_gradient,
    sigma_np,
    dx,
    dz,
    alpha_data,
    alpha_tv,
    beta,
    tv_domain,
    device,
):
    """Combine sigma data and TV gradients with optional log-domain TV."""
    combined = normalize_torch_gradient(data_gradient)
    if alpha_tv != 0.0:
        sigma_safe = np.maximum(np.asarray(sigma_np, dtype=np.float32), 1e-12)
        if tv_domain == "log":
            tv_gradient = compute_tv_gradient(np.log(sigma_safe), dx, dz) / sigma_safe
        else:
            tv_gradient = compute_tv_gradient(sigma_np, dx, dz)
        tv_gradient_tensor = torch.from_numpy(tv_gradient).to(device).float()
        combined = combined + alpha_tv * normalize_torch_gradient(tv_gradient_tensor)
    return normalize_torch_gradient(combined) * alpha_data * beta


def write_metrics_files(save_path, metrics_history):
    """Write machine-readable metrics for external sweep tools such as Optuna."""
    metrics_path = Path(save_path)
    metrics_path.mkdir(parents=True, exist_ok=True)
    final_metrics = metrics_history[-1] if metrics_history else {}
    payload = {
        "final": final_metrics,
        "history": metrics_history,
    }
    with (metrics_path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if not metrics_history:
        return
    fieldnames = list(metrics_history[0].keys())
    with (metrics_path / "metrics_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_history)


def prepare_overthrust_model(overthrust_model, target_shape=(100, 200)):
    """Return a float32 OverThrust model resized to the training grid."""
    if overthrust_model.ndim != 2:
        raise ValueError(f"OverThrust model must be 2D, got shape {overthrust_model.shape}")
    if overthrust_model.shape == target_shape:
        return overthrust_model.astype(np.float32, copy=False), False
    zoom_factors = (
        target_shape[0] / overthrust_model.shape[0],
        target_shape[1] / overthrust_model.shape[1],
    )
    return zoom(overthrust_model, zoom_factors, order=1).astype(np.float32, copy=False), True


def resolve_overthrust_data_path(data_path):
    """Resolve the requested OverThrust data file and provide a helpful fallback."""
    requested = Path(data_path)
    if requested.exists():
        return requested

    candidates = []
    if requested.name != "OverThrust.npy":
        candidates.append(requested.with_name("OverThrust.npy"))
    candidates.append(DEFAULT_DATA_PATH)

    for candidate in candidates:
        if candidate.exists():
            if rank == 0:
                print(f"未找到数据文件 {requested}，改用 {candidate}")
            return candidate

    tried = [str(requested)] + [str(candidate) for candidate in candidates]
    raise FileNotFoundError(
        "找不到 OverThrust 数据文件。已尝试: " + "; ".join(tried)
    )


def compute_ssim_2d(pred, target, data_range=None, window_size=7):
    """Compute a lightweight 2D SSIM score with a uniform local window."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"SSIM shapes must match, got {pred.shape} and {target.shape}")
    if pred.ndim != 2:
        raise ValueError(f"SSIM expects 2D arrays, got ndim={pred.ndim}")
    if min(pred.shape) < 2:
        return float("nan")
    if np.array_equal(pred, target):
        return 1.0

    if data_range is None:
        target_range = float(np.max(target) - np.min(target))
        pred_range = float(np.max(pred) - np.min(pred))
        value_scale = float(max(np.max(np.abs(target)), np.max(np.abs(pred)), 1.0))
        data_range = max(target_range, pred_range, value_scale * 1e-6)
    else:
        data_range = max(float(data_range), 1e-12)

    window_size = int(max(2, min(window_size, pred.shape[0], pred.shape[1])))
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_pred = uniform_filter(pred, size=window_size, mode="reflect")
    mu_target = uniform_filter(target, size=window_size, mode="reflect")
    mu_pred_sq = mu_pred * mu_pred
    mu_target_sq = mu_target * mu_target
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = uniform_filter(pred * pred, size=window_size, mode="reflect") - mu_pred_sq
    sigma_target_sq = uniform_filter(target * target, size=window_size, mode="reflect") - mu_target_sq
    sigma_cross = uniform_filter(pred * target, size=window_size, mode="reflect") - mu_cross

    numerator = (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-30)
    return float(np.mean(ssim_map))


def compute_model_quality_metrics(eps_pred, eps_true, sig_pred, sig_true, fixed_top_rows=0):
    """Return full-model and active-region MSE/NMSE/SSIM metrics."""
    eps_pred = np.asarray(eps_pred, dtype=np.float64)
    eps_true = np.asarray(eps_true, dtype=np.float64)
    sig_pred = np.asarray(sig_pred, dtype=np.float64)
    sig_true = np.asarray(sig_true, dtype=np.float64)

    def block(prefix, pred, target):
        mse = float(np.mean((pred - target) ** 2))
        denom = float(np.mean(target**2))
        nmse = mse / max(denom, 1e-30)
        ssim = compute_ssim_2d(pred, target)
        return {
            f"{prefix}_mse": mse,
            f"{prefix}_nmse": nmse,
            f"{prefix}_ssim": ssim,
        }

    metrics = {}
    metrics.update(block("eps_model", eps_pred, eps_true))
    metrics.update(block("sig_model", sig_pred, sig_true))

    start = int(max(0, min(fixed_top_rows, eps_pred.shape[0] - 1)))
    metrics.update(block("eps_model_active", eps_pred[start:, :], eps_true[start:, :]))
    metrics.update(block("sig_model_active", sig_pred[start:, :], sig_true[start:, :]))
    metrics["model_quality_objective_active"] = (
        metrics["eps_model_active_nmse"]
        + metrics["sig_model_active_nmse"]
        + (1.0 - metrics["eps_model_active_ssim"])
        + (1.0 - metrics["sig_model_active_ssim"])
    )
    metrics["model_quality_objective_full"] = (
        metrics["eps_model_nmse"]
        + metrics["sig_model_nmse"]
        + (1.0 - metrics["eps_model_ssim"])
        + (1.0 - metrics["sig_model_ssim"])
    )
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate-eps", type=float, default=1e-4)
    parser.add_argument("--learning-rate-sig", type=float, default=1e-5)
    parser.add_argument("--lr-eps-stage2", "--lr_eps_stage2", dest="lr_eps_stage2", type=float, default=1e-6)
    parser.add_argument("--alpha-eps", type=float, default=1.0)
    parser.add_argument("--alpha-sig", type=float, default=1.0)
    parser.add_argument("--alpha-tv-eps", type=float, default=0.0)
    parser.add_argument("--alpha-tv-sig", type=float, default=5e-4)
    parser.add_argument("--sigma-beta", type=float, default=1.0)
    parser.add_argument("--sigma-param", choices=("linear", "log"), default="linear")
    parser.add_argument("--sigma-tv-domain", choices=("physical", "log"), default="physical")
    parser.add_argument("--alpha-tikhonov-eps", type=float, default=0.0)
    parser.add_argument("--alpha-tikhonov-sig", type=float, default=0.0)
    parser.add_argument("--alpha-l1-data", type=float, default=0.0)
    parser.add_argument("--stage1-data-loss", choices=("l1", "l2", "mixed"), default="mixed")
    parser.add_argument("--stage2-data-loss", choices=("l1", "l2", "mixed"), default="mixed")
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--num-epochs-stage1", type=int, default=2500)
    parser.add_argument("--num-epochs-stage2", type=int, default=12501)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage1-checkpoint-path", type=str, default="")
    parser.add_argument("--illumination", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument(
        "--jax-device",
        type=str,
        default="",
        help="Device for the JAX FDTD backend, for example cuda:1 or auto. Defaults to --device.",
    )
    parser.add_argument(
        "--jax-shot-batch-size",
        type=int,
        default=5,
        help="Number of shots per JAX FDTD batch. Lower this to 1 or 2 if CUDA runs out of memory.",
    )
    parser.add_argument(
        "--auto-jax-shot-batch-size",
        action="store_true",
        help="Estimate the initial JAX shot batch size from free CUDA memory.",
    )
    parser.add_argument(
        "--max-jax-shot-batch-size",
        type=int,
        default=5,
        help="Upper bound used with --auto-jax-shot-batch-size.",
    )
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fixed-top-rows", type=int, default=10)
    parser.add_argument(
        "--sigma-shallow-skip",
        action="store_true",
        help="Use the newer shallow skip connections in the sigma UNet. Off by default to match the original mode2 program.",
    )
    parser.add_argument("--snapshot-interval", type=int, default=1000)
    parser.add_argument(
        "--fdtd-backend",
        choices=("numpy", "jax"),
        default="numpy",
        help="FDTD backend. numpy keeps the original CPU/MPI path; jax uses the JAX GPU forward and hand-written adjoint.",
    )
    return parser.parse_args(argv)


def resolve_fixed_top_rows_from_checkpoint(args):
    """Keep stage2's fixed shallow layer consistent with the saved stage1 run."""
    fixed_top_rows = int(args.fixed_top_rows)
    if not (args.resume and args.stage1_checkpoint_path):
        return fixed_top_rows

    ckpt_path = args.stage1_checkpoint_path.strip()
    if not ckpt_path:
        return fixed_top_rows

    checkpoint_fixed_top_rows = None
    if rank == 0 and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            checkpoint_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
            if isinstance(checkpoint_args, dict) and "fixed_top_rows" in checkpoint_args:
                checkpoint_fixed_top_rows = int(checkpoint_args["fixed_top_rows"])
        except Exception as exc:
            print(f"警告：无法从checkpoint读取fixed_top_rows，将使用命令行值 {fixed_top_rows}: {exc}")

    if MPI_AVAILABLE and size > 1:
        checkpoint_fixed_top_rows = comm.bcast(checkpoint_fixed_top_rows, root=0)

    if checkpoint_fixed_top_rows is not None:
        if checkpoint_fixed_top_rows != fixed_top_rows and rank == 0:
            print(
                "检测到stage1 checkpoint固定浅层行数与命令行不一致，"
                f"使用checkpoint值 fixed_top_rows={checkpoint_fixed_top_rows} "
                f"(命令行为 {fixed_top_rows})"
            )
        fixed_top_rows = checkpoint_fixed_top_rows

    return fixed_top_rows


# ========== 主程序 ==========
def main(argv=None):
    args = parse_args(argv)
    global L_SIZE
    args.fixed_top_rows = resolve_fixed_top_rows_from_checkpoint(args)
    L_SIZE = args.fixed_top_rows
    # 设置随机种子
    same_seeds(2025)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Create the base output directory on rank 0, then broadcast it.
    if rank == 0:
        base_output_dir = os.path.abspath(args.output_dir)
        try:
            os.makedirs(base_output_dir, exist_ok=True)
        except PermissionError:
            home_dir = os.path.expanduser('~')
            base_output_dir = home_dir
            os.makedirs(base_output_dir, exist_ok=True)
        print(f"结果将保存到: {base_output_dir}")
    else:
        base_output_dir = None

    # 广播基础输出目录给所有进程
    base_output_dir = comm.bcast(base_output_dir if rank == 0 else None, root=0)
    
    comm.Barrier()  # 等待所有进程
    
    # 设置设备
    requested_device = _resolve_rank_device(args.device)
    if requested_device.startswith('cuda'):
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            if requested_device == 'cuda':
                device = torch.device('cuda')
            else:
                parts = requested_device.split(':', 1)
                if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) >= torch.cuda.device_count():
                    device = torch.device('cuda:0')
                else:
                    device = torch.device(requested_device)
    else:
        device = torch.device(requested_device)

    if args.fdtd_backend == "jax":
        jax_device = args.jax_device or requested_device
        jax_device = _resolve_rank_device(jax_device)
        jax_shot_batch_size = (
            _estimate_jax_shot_batch_size(str(device), args.max_jax_shot_batch_size)
            if args.auto_jax_shot_batch_size
            else args.jax_shot_batch_size
        )
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ["JAX_FDTD_SHOT_BATCH_SIZE"] = str(jax_shot_batch_size)
        os.environ["JAX_FDTD_DEVICE"] = jax_device
    if rank == 0:
        print(f"使用设备: {device}")
        print(f"FDTD backend: {args.fdtd_backend}")
        if args.fdtd_backend == "jax":
            print(f"JAX FDTD device: {os.environ.get('JAX_FDTD_DEVICE')}")
            print(f"JAX shot batch size: {os.environ.get('JAX_FDTD_SHOT_BATCH_SIZE')}")
            print(f"XLA_PYTHON_CLIENT_PREALLOCATE: {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')}")
        print(f"MPI进程数: {size}")
    
    # ========== 训练设置 ==========
    learning_rate_eps = args.learning_rate_eps
    learning_rate_sig = args.learning_rate_sig
    learning_rate_eps_stage2 = args.lr_eps_stage2 if args.lr_eps_stage2 is not None else learning_rate_eps
    alpha_eps = args.alpha_eps
    alpha_sig = args.alpha_sig
    alpha_tv_eps = args.alpha_tv_eps
    alpha_tv_sig = args.alpha_tv_sig
    sigma_beta = args.sigma_beta
    sigma_param = args.sigma_param
    sigma_tv_domain = args.sigma_tv_domain
    if sigma_beta <= 0.0:
        raise ValueError("--sigma-beta must be positive")
    alpha_tikhonov_eps = args.alpha_tikhonov_eps
    alpha_tikhonov_sig = args.alpha_tikhonov_sig
    alpha_l1_data = args.alpha_l1_data
    stage1_data_loss = args.stage1_data_loss
    stage2_data_loss = args.stage2_data_loss
    offset = args.offset
    l_size = args.fixed_top_rows
    
    # 构造结果文件夹名
    if rank == 0:
        result_dir = os.path.join(
            base_output_dir,
            f'Eps_First_lrE1{learning_rate_eps:.0e}_lrE2{learning_rate_eps_stage2:.0e}_lrS{learning_rate_sig:.0e}_tvEps{alpha_tv_eps:.3g}_tvSig{alpha_tv_sig:.3g}_sigParam{sigma_param}_beta{sigma_beta:.3g}_tvD{sigma_tv_domain}_S1{stage1_data_loss}_S2{stage2_data_loss}_l1D{alpha_l1_data:.3g}_TikSig{alpha_tikhonov_sig:.3g}_E2{args.num_epochs_stage2}',
        )
        result_dir = ensure_unique_dir(result_dir)
        path = result_dir if result_dir.endswith(os.sep) else result_dir + os.sep
        print(f"结果保存路径: {path}")
    
    comm.Barrier()
    
    # ========== 参数设置 ==========
    # 网格参数
    xl, zl = 100, 200  # 深度x横向长度
    dx, dz = 0.02, 0.02
    dt = 4e-11
    npml = 10
    steps = 1000
    
    # 震源参数
    freq = 4e8
    
    #真实模型参数。服务器上可能已经提前把原始 200x400 OverThrust 下采样到 100x200。
    data_path = resolve_overthrust_data_path(args.data_path)
    overthrust_model = np.load(data_path)
    epsilon_true, resized_model = prepare_overthrust_model(overthrust_model, target_shape=(xl, zl))
    if rank == 0:
        if resized_model:
            print(f"将数据模型从 {overthrust_model.shape} resize 到 {(xl, zl)}")
        else:
            print(f"使用已匹配网格的数据模型: {overthrust_model.shape}")
    top_layer = torch.tensor(epsilon_true[:l_size,:], dtype=torch.float32, device=device)
    
    # Sigma模型处理
    sigma_true_raw = epsilon_true.copy()*1e-3
    top_layer_sig = torch.tensor(sigma_true_raw[:l_size,:], dtype=torch.float32, device=device)

    # 转换为torch张量
    epsilon_true = torch.tensor(epsilon_true, dtype=torch.float32, device=device)
    sigma_true = torch.tensor(sigma_true_raw, dtype=torch.float32, device=device)
    torch.set_default_dtype(torch.float32)
    # 参数范围
    min_eps, max_eps = 3.0, 8.5
    min_sig,max_sig = 3.0*1e-3, 8.5*1e-3
    # 源点和接收点设置
    source_list = [(0, i) for i in range(0, zl, 5)]  
    receiver_list = [(0, i) for i in range(0, zl, 2)]
    
    if rank == 0:
        print(f"源点数量: {len(source_list)}")
        print(f"接收点数量: {len(receiver_list)}")
        print(f"源点位置: {source_list}")
        print(f"接收点位置: {receiver_list}")

        # 额外保存：真实模型 epsilon_true 图像
        save_model_figure(epsilon_true, f'{path}epsilon_true.png', 'True Epsilon Model', 
                          is_sigma=False, dx=dx, dz=dz, vmin=min_eps, vmax=max_eps)

        save_model_figure(sigma_true, f'{path}sigma_true.png', 'True Sigma Model', 
                          is_sigma=True, dx=dx, dz=dz, vmin=min_sig, vmax=max_sig)
    
    # ========== 生成观测数据（MPI并行化） ==========
    if rank == 0:
        print("生成观测数据...")
    
    # 使用MPI并行化生成观测数据
    local_d_obs = forward_model(
        epsilon_true.cpu().numpy(),
        sigma_true.cpu().numpy(),
        source_list,
        receiver_list,
        dt, dx, dz, npml, freq, steps, save_wavefield=False
    )  # 这一步中返回的d_obs是各进程本地的，需要合并
    d_obs_list = comm.gather(local_d_obs, root=0)
    if rank == 0:
        # 将各进程的观测数据在shot维拼接为三维数组 [n_shots, n_receivers, n_steps]
        d_obs_np = np.concatenate(d_obs_list, axis=0) if len(d_obs_list) > 0 else np.zeros((0, len(receiver_list), steps), dtype=np.float32)
    else:
        d_obs_np = None
    # 广播拼接好的三维观测数据到所有进程
    d_obs_np = comm.bcast(d_obs_np, root=0)
    
    d_obs = torch.tensor(d_obs_np, dtype=torch.float32, device=device)
    if rank == 0:
        print(f"观测数据形状: {d_obs.shape}")
        print(f"观测数据类型: {d_obs.dtype}")

        # 额外保存：观测数据 d_obs 图像 - 处理3D数据结构
        if d_obs.dim() == 3:  # (n_shots, n_receivers, n_steps)
            d_obs_np = d_obs.detach().cpu().numpy()
            print(f"观测数据统计信息:")
            print(f"  形状: {d_obs_np.shape}")
            print(f"  最小值: {np.min(d_obs_np):.6f}")
            print(f"  最大值: {np.max(d_obs_np):.6f}")
            print(f"  平均值: {np.mean(d_obs_np):.6f}")
            print(f"  标准差: {np.std(d_obs_np):.6f}")
            shot_idx = d_obs_np.shape[0] // 2
            shot_data = d_obs_np[shot_idx]
            plt.figure(figsize=(12, 4))
            plt.imshow(shot_data.T, aspect='auto', cmap='seismic')
            plt.title(f'Observed Data (d_obs) - Shot {shot_idx}')
            plt.xlabel('Receiver Index')
            plt.ylabel('Time Step')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"{path}d_obs.png")
            plt.close()
        else:
            # 兼容旧的2D数据格式
            plt.figure(figsize=(12, 4))
            plt.imshow(d_obs.detach().cpu().numpy().T, aspect='auto', cmap='seismic')
            plt.title('Observed Data (d_obs) - 2D Format')
            plt.colorbar(); plt.tight_layout()
            plt.savefig(f"{path}d_obs.png"); plt.close()

    
    
    # ========== 数据加载器设置 ==========
    # 创建数据集
    if d_obs.dim() == 4:
        d_obs = d_obs.squeeze(0)  # 移除可能的batch维度
    
    dataset = ShotDataset([d_obs], [source_list], [receiver_list])
    train_loader = data_utils.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # ========== UNet网络设置 ==========
    net_1 = None
    net_2 = None
    if rank == 0:
        print("初始化UNet网络...")
        model_train_size = (xl - l_size, zl)
        net_1 = UNet(in_channels=1, output_size=model_train_size).to(device)  # 介电常数网络
        print(
            f"固定浅层行数: {l_size}; UNet输出尺寸: {model_train_size}; "
            f"sigma_shallow_skip={args.sigma_shallow_skip}"
        )
        net_2 = UNet(
            in_channels=1,
            shallow_skip_connections=args.sigma_shallow_skip,
            output_size=model_train_size,
        ).to(device)  # 电导率网络
        print(f"Net1(Eps)参数数量: {get_parameter_number(net_1)}")
        print(f"Net2(Sig)参数数量: {get_parameter_number(net_2)}")
    
    # ========== 网络输入处理 ==========
    # 对真实模型进行高斯滤波处理
    eps_input = torch.tensor(1/gaussian_filter(1/epsilon_true.cpu().numpy(), 5), dtype=torch.float32)
    sig_input = torch.tensor(1/gaussian_filter(1/sigma_true.cpu().numpy(), 5), dtype=torch.float32)
    
    # 对sig_input进行归一化，以便网络更好处理
    sig_input_min = sig_input.min()
    sig_input_max = sig_input.max()
    sig_input = (sig_input - sig_input_min) / (sig_input_max - sig_input_min) # 归一化到0-1

    # 计算padding，按输入尺寸对齐到32的倍数
    h0, w0 = eps_input.shape
    required_height = ((h0 + 31) // 32) * 32
    required_width = ((w0 + 31) // 32) * 32
    pad_height = required_height - h0
    pad_width = required_width - w0
    
    if pad_height % 2 == 1:
        pad_height += 1
    if pad_width % 2 == 1:
        pad_width += 1
    
    if rank == 0:
        print(f"padding: height={pad_height}, width={pad_width}")
    
    # 对称padding
    eps_input_padded = F.pad(eps_input, (pad_width//2, pad_width//2, pad_height//2, pad_height//2))
    sig_input_padded = F.pad(sig_input, (pad_width//2, pad_width//2, pad_height//2, pad_height//2))
    
    orig_height, orig_width = h0, w0
    # 扩展维度
    net1_input = eps_input_padded.unsqueeze(0).unsqueeze(0).to(device)
    net2_input = sig_input_padded.unsqueeze(0).unsqueeze(0).to(device)
    net2_input_fixed = net2_input.detach()
    
    # 广播网络输入到所有进程
    if MPI_AVAILABLE and size > 1:
        net1_input = comm.bcast(net1_input, root=0)
        net2_input = comm.bcast(net2_input, root=0)
  
    # ========== 训练设置 ==========
    optimizer = None
    optimizer_both = None
    if rank == 0:
        optimizer = torch.optim.Adam(net_1.parameters(), lr=learning_rate_eps, betas=(0.9, 0.95))
    loss_fn = torch.nn.MSELoss()
    num_epochs_stage1 = args.num_epochs_stage1
    num_epochs_stage2 = args.num_epochs_stage2
    num_epochs = num_epochs_stage1 + num_epochs_stage2

    stage1_checkpoint_save_path = None
    if rank == 0:
        stage1_checkpoint_save_path = args.stage1_checkpoint_path.strip() if args.stage1_checkpoint_path else ""
        if not stage1_checkpoint_save_path:
            stage1_checkpoint_save_path = os.path.join(path, "stage1_eps_checkpoint.pt")
    stage1_checkpoint_save_path = comm.bcast(stage1_checkpoint_save_path if rank == 0 else None, root=0)

    sigma_uniform_value = float(top_layer_sig.mean().item())
    sigma_bottom = torch.full(
        (sigma_true.shape[0] - top_layer_sig.shape[0], sigma_true.shape[1]),
        sigma_uniform_value,
        dtype=torch.float32,
        device=device,
    )
    sigma_fixed_root = torch.cat((top_layer_sig, sigma_bottom), dim=0)
    sigma_fixed_root = torch.clamp(sigma_fixed_root, min=min_sig, max=max_sig)

    eps_model_for_sig = None

    if args.resume:
        ckpt_path = args.stage1_checkpoint_path.strip() if args.stage1_checkpoint_path else ""
        if not ckpt_path:
            candidates = []
            try:
                import glob
                candidates = glob.glob(os.path.join(base_output_dir, "**", "stage1_eps_checkpoint.pt"), recursive=True)
            except Exception:
                candidates = []
            if candidates:
                candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                ckpt_path = candidates[0]
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError("resume需要提供 --stage1-checkpoint-path 或确保 result_overthrust_twopara 下存在 stage1_eps_checkpoint.pt")
        if rank == 0:
            ckpt = torch.load(ckpt_path, map_location=device)
            if "net1_state_dict" in ckpt and ckpt["net1_state_dict"] is not None and net_1 is not None:
                net_1.load_state_dict(ckpt["net1_state_dict"])
            if "net2_input" not in ckpt:
                raise KeyError("checkpoint缺少 net2_input")
            net2_input_fixed = ckpt["net2_input"].to(device).detach()
            optimizer_both = torch.optim.Adam([
                {'params': net_1.parameters(), 'lr': learning_rate_eps_stage2},
                {'params': net_2.parameters(), 'lr': learning_rate_sig},
            ], betas=(0.9, 0.99))
            optimizer = optimizer_both
        if MPI_AVAILABLE and size > 1:
            net2_input_fixed = comm.bcast(net2_input_fixed if rank == 0 else None, root=0)
        num_epochs_stage1 = 0
        num_epochs = num_epochs_stage2
    
    # 损失记录
    loss_history = []
    eps_loss_history = []
    metrics_history = []
    
    # 训练循环
    if rank == 0:
        print("开始训练...")
    for epoch in range(num_epochs):
        train_sigma = epoch >= num_epochs_stage1
        if rank == 0:
            if (not args.resume) and epoch == num_epochs_stage1:
                if eps_model_for_sig is None:
                    raise RuntimeError("eps_model_for_sig is None at stage switch")
                eps_min = eps_model_for_sig.min()
                eps_max = eps_model_for_sig.max()
                eps_norm = (eps_model_for_sig - eps_min) / (eps_max - eps_min + 1e-8)
                eps_norm_padded = F.pad(eps_norm, (pad_width//2, pad_width//2, pad_height//2, pad_height//2))
                net2_input_fixed = eps_norm_padded.unsqueeze(0).unsqueeze(0).to(device).detach()
                torch.save(
                    {
                        "net1_state_dict": net_1.state_dict() if net_1 is not None else None,
                        "net2_input": net2_input_fixed.detach().cpu(),
                        "args": vars(args),
                        "num_epochs_stage1": num_epochs_stage1,
                    },
                    stage1_checkpoint_save_path,
                )
                optimizer_both = torch.optim.Adam([
                    {'params': net_1.parameters(), 'lr': learning_rate_eps_stage2},
                    {'params': net_2.parameters(), 'lr': learning_rate_sig},
                ], betas=(0.9, 0.99))
                optimizer = optimizer_both
            if net_1 is not None: net_1.train()
            if net_2 is not None:
                if train_sigma:
                    net_2.train()
                else:
                    net_2.eval()
        if MPI_AVAILABLE and size > 1 and (not args.resume) and epoch == num_epochs_stage1:
            net2_input_fixed = comm.bcast(net2_input_fixed if rank == 0 else None, root=0)
        epoch_loss = 0
        epoch_data_misfit_sum = 0.0
        epoch_l1_data_sum = 0.0
        epoch_metric_samples = 0.0
        total_samples = 0
        selected_indices = range(len(source_list))
        selected_source_list = source_list
        
        selected_receiver_list = receiver_list
        # 处理3D数据结构 (n_shots, n_receivers, n_steps)
        if d_obs.dim() == 3:
            selected_d_obs = d_obs[selected_indices, :, :]
        else:
            # 兼容旧的2D数据格式
            selected_d_obs = d_obs[selected_indices, :]
        
        # 计算与forward.py一致的本进程shot切片区间
        n_shots_total = len(selected_source_list)
        shots_per_proc = n_shots_total // size
        remainder = n_shots_total % size
        if rank < remainder:
            start_idx = rank * (shots_per_proc + 1)
            end_idx = start_idx + shots_per_proc + 1
        else:
            start_idx = rank * shots_per_proc + remainder
            end_idx = start_idx + shots_per_proc
        
        # 根据数据维度选择正确的切片
        if d_obs.dim() == 3:
            selected_d_obs_local = selected_d_obs[start_idx:end_idx, :, :]
        else:
            selected_d_obs_local = selected_d_obs[start_idx:end_idx, :]
        
        for i, (s_d_i, x_s_i, x_r_i) in enumerate(train_loader):
            batch_size_now = s_d_i.size(0)
            if rank == 0 and optimizer is not None:
                optimizer.zero_grad()
            
            # 准备各进程使用的模型参数
            if rank == 0 and net_1 is not None:
                # Epsilon网络前向
                batch_net1_input = net1_input[:, i:i+1, :, :].to(device)
                _, _, net1_output_model = net_1(batch_net1_input)

                # Epsilon后处理
                net1_ = torch.sigmoid(net1_output_model + offset)
                net1_o = net1_ * (max_eps - min_eps) + min_eps
                epsilon = torch.cat((top_layer, net1_o[:, :]), dim=0)
                outputs_eps_root = torch.clamp(epsilon, min=min_eps, max=max_eps)
                outputs_eps_root.retain_grad()
                outputs_eps_np = outputs_eps_root.detach().cpu().numpy()
                eps_model_for_sig = outputs_eps_root.detach()

                if train_sigma and net_2 is not None:
                    # Sigma网络前向
                    batch_net2_input = net2_input_fixed[:, i:i+1, :, :].to(device)
                    _, _, net2_output_model = net_2(batch_net2_input)

                    # Sigma后处理
                    net2_ = torch.sigmoid(net2_output_model + offset)
                    if sigma_param == "log":
                        log_min_sig = torch.log(torch.tensor(min_sig, dtype=net2_.dtype, device=net2_.device))
                        log_max_sig = torch.log(torch.tensor(max_sig, dtype=net2_.dtype, device=net2_.device))
                        net2_o = torch.exp(net2_ * (log_max_sig - log_min_sig) + log_min_sig)
                    else:
                        net2_o = net2_ * (max_sig - min_sig) + min_sig
                    sigma = torch.cat((top_layer_sig, net2_o[:, :]), dim=0)
                    outputs_sig_root = torch.clamp(sigma, min=min_sig, max=max_sig)
                    outputs_sig_root.retain_grad()
                    outputs_sig_np = outputs_sig_root.detach().cpu().numpy()

                    if i == 0 and epoch % 50 == 0:
                        print(f"net1(eps) out stats: min={net1_output_model.min():.4f}, max={net1_output_model.max():.4f}")
                        print(f"net2(sig) out stats: min={net2_output_model.min():.4f}, max={net2_output_model.max():.4f}")
                else:
                    outputs_sig_root = sigma_fixed_root
                    outputs_sig_np = outputs_sig_root.detach().cpu().numpy()
            else:
                outputs_eps_np = None
                outputs_sig_np = None

            # 广播outputs到所有进程
            outputs_eps_np = comm.bcast(outputs_eps_np, root=0)
            outputs_sig_np = comm.bcast(outputs_sig_np, root=0)

            # 各进程构建本地可求导副本；root使用原始变量以保持计算图
            if rank == 0 and net_1 is not None:
                outputs_eps_use = outputs_eps_root
                outputs_sig_use = outputs_sig_root
            else:
                outputs_eps_use = torch.tensor(outputs_eps_np, dtype=torch.float32, device=device, requires_grad=True)
                outputs_eps_use.retain_grad()
                outputs_sig_use = torch.tensor(outputs_sig_np, dtype=torch.float32, device=device, requires_grad=train_sigma)
                if outputs_sig_use.requires_grad:
                    outputs_sig_use.retain_grad()

            if outputs_eps_use.requires_grad:
                outputs_eps_use.register_hook(grad_hook)
            if outputs_sig_use.requires_grad:
                outputs_sig_use.register_hook(grad_hook)

            # 使用MPI并行正演，保存波场
            if rank == 0:
                print(f"Epoch {epoch}, Batch {i}: 开始正演模拟...")

            d_syn = ForwardModelFunction.apply(
                outputs_eps_use, outputs_sig_use, selected_source_list, selected_receiver_list, 
                dt, dx, dz, npml, freq, steps, alpha_eps, alpha_sig, alpha_tv_eps, alpha_tv_sig, sigma_beta, sigma_tv_domain, train_sigma, args.illumination, args.fdtd_backend
            )
            # 确保数据类型一致 - 都转换为float32
            d_syn = d_syn.float()
            selected_d_obs_local = selected_d_obs_local.float()

            if rank == 0:
                print(f"合成数据形状(本进程): {d_syn.shape}")
                print(f"合成数据类型: {d_syn.dtype}")
                print(f"观测数据形状(本进程): {selected_d_obs_local.shape}")
                print(f"观测数据类型: {selected_d_obs_local.dtype}")
                
                # 打印数据维度信息
                print(f"Epoch {epoch}: 数据维度信息:")
                print(f"  d_syn shape: {d_syn.shape}")
                print(f"  selected_d_obs_local shape: {selected_d_obs_local.shape}")
                print(f"  selected_source_list length: {len(selected_source_list)}")
                print(f"  receiver_list length: {len(receiver_list)}")
                
                # 在训练初期可视化数据对比
                if epoch < 5 and i == 0:
                    print(f"Epoch {epoch}: 可视化数据对比...")
                    
                    # 数据统计信息
                    d_syn_np = d_syn.detach().cpu().numpy()
                    d_obs_np = selected_d_obs_local.detach().cpu().numpy()
                    print(f"  合成数据 - 形状: {d_syn_np.shape}, 范围: [{np.min(d_syn_np):.6f}, {np.max(d_syn_np):.6f}]")
                    print(f"  观测数据 - 形状: {d_obs_np.shape}, 范围: [{np.min(d_obs_np):.6f}, {np.max(d_obs_np):.6f}]")
                    if d_syn_np.shape == d_obs_np.shape:
                        residual_np = d_syn_np - d_obs_np
                        print(f"  残差范围: [{np.min(residual_np):.6f}, {np.max(residual_np):.6f}]")
                    else:
                        print(f"  警告: 合成数据和观测数据形状不匹配!")
                    
            
            # 计算数据损失。MSE/L1都记录下来；真正用于反传的loss按阶段选择。
            mse_data_loss = loss_fn(d_syn, selected_d_obs_local)
            l1_data_loss = F.l1_loss(d_syn, selected_d_obs_local)
            current_data_loss_mode = stage2_data_loss if train_sigma else stage1_data_loss
            if current_data_loss_mode == "l1":
                data_loss = l1_data_loss
            elif current_data_loss_mode == "l2":
                data_loss = mse_data_loss
            else:
                data_loss = mse_data_loss + alpha_l1_data * l1_data_loss

            # 总损失：数据拟合项 + 可选Tikhonov正则项。
            total_loss = data_loss
            if alpha_tikhonov_eps != 0.0 or alpha_tikhonov_sig != 0.0:
                total_loss = total_loss + tikhonov_reg(
                    outputs_eps_use,
                    outputs_sig_use,
                    alpha_tikhonov_eps,
                    alpha_tikhonov_sig,
                    dx=dx,
                    dz=dz,
                )
            
            if np.isnan(float(total_loss.item())):
                raise ValueError('loss is nan while training')
            
            # 反向传播
            total_loss.backward(retain_graph=True)

            # 参数更新
            if rank == 0 and optimizer is not None:
                optimizer.step()
                #scheduler.step()

            local_metric_samples = selected_d_obs_local.shape[0] if selected_d_obs_local.dim() > 0 else 1
            local_metrics = np.array(
                [
                    float(mse_data_loss.item()) * local_metric_samples,
                    float(l1_data_loss.item()) * local_metric_samples,
                    float(local_metric_samples),
                ],
                dtype=np.float64,
            )
            if MPI_AVAILABLE and size > 1:
                global_metrics = np.empty_like(local_metrics)
                comm.Allreduce(local_metrics, global_metrics, op=MPI.SUM)
            else:
                global_metrics = local_metrics

            # 统计与可视化（仅root）
            if rank == 0:
                epoch_loss += total_loss.item() * batch_size_now
                epoch_data_misfit_sum += global_metrics[0]
                epoch_l1_data_sum += global_metrics[1]
                epoch_metric_samples += global_metrics[2]
                total_samples += batch_size_now

                if epoch % 20 == 0 or epoch in [1, 2, 3, 5, 10]:
                    # 可视化结果
                    save_model_figure(outputs_eps_use, f'{path}epoch_{epoch}_epsilon.png', 
                                      f'Epsilon (epoch {epoch})', is_sigma=False, dx=dx, dz=dz, vmin=min_eps, vmax=max_eps)
                    print(f"Epoch {epoch}: 已保存epsilon图")

                    save_model_figure(outputs_sig_use, f'{path}epoch_{epoch}_sigma.png', 
                                      f'Sigma (epoch {epoch})', is_sigma=True, dx=dx, dz=dz)
                    print(f"Epoch {epoch}: 已保存sigma图")

                    eps_residual = outputs_eps_use.detach() - epsilon_true
                    sig_residual = outputs_sig_use.detach() - sigma_true
                    save_residual_figure(
                        eps_residual,
                        f'{path}epoch_{epoch}_epsilon_residual.png',
                        f'Epsilon Residual (inverted - true, epoch {epoch})',
                        r'$\Delta\epsilon_r$',
                        dx=dx,
                        dz=dz,
                    )
                    save_residual_figure(
                        sig_residual,
                        f'{path}epoch_{epoch}_sigma_residual.png',
                        f'Sigma Residual (inverted - true, epoch {epoch})',
                        r'$\Delta\sigma$ (S/m)',
                        dx=dx,
                        dz=dz,
                    )
                    print(f"Epoch {epoch}: 已保存epsilon/sigma残差图")

                    if d_syn.dim() == 3 and selected_d_obs_local.dim() == 3:
                        d_syn_np_vis = d_syn.detach().cpu().numpy()
                        d_obs_np_vis = selected_d_obs_local.detach().cpu().numpy()
                        n_local_shots = d_syn_np_vis.shape[0]
                        if n_local_shots > 0:
                            shot_idx_local = n_local_shots // 2
                            global_shot_idx = start_idx + shot_idx_local
                            shot_syn = d_syn_np_vis[shot_idx_local]
                            shot_obs = d_obs_np_vis[shot_idx_local]
                            vmin = min(shot_syn.min(), shot_obs.min())
                            vmax = max(shot_syn.max(), shot_obs.max())
                            plt.figure()
                            plt.subplot(1, 2, 1)
                            plt.imshow(shot_obs.T, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax)
                            plt.title(f'd_obs shot {global_shot_idx}')
                            plt.subplot(1, 2, 2)
                            plt.imshow(shot_syn.T, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax)
                            plt.title(f'd_syn shot {global_shot_idx}')
                            plt.tight_layout()
                            plt.savefig(f'{path}epoch_{epoch}_shot_{global_shot_idx}_gather_compare.png')
                            plt.close()

                            n_receivers = shot_syn.shape[0]
                            n_steps = shot_syn.shape[1]
                            if n_receivers >= 3:
                                rec_indices = [0, n_receivers // 2, n_receivers - 1]
                            else:
                                rec_indices = list(range(n_receivers))
                            t = np.arange(n_steps) * dt
                            fig, axes = plt.subplots(len(rec_indices), 1, figsize=(6, 2 * len(rec_indices)), sharex=True)
                            if len(rec_indices) == 1:
                                axes = [axes]
                            for idx_plot, rec_idx in enumerate(rec_indices):
                                axes[idx_plot].plot(t, shot_obs[rec_idx], 'k-', label='d_obs')
                                axes[idx_plot].plot(t, shot_syn[rec_idx], 'r--', label='d_syn')
                                axes[idx_plot].set_ylabel(f'Rec {rec_idx}')
                                axes[idx_plot].grid(True, alpha=0.3)
                            axes[-1].set_xlabel('Time')
                            axes[0].legend(loc='upper right')
                            fig.tight_layout()
                            fig.savefig(f'{path}epoch_{epoch}_shot_{global_shot_idx}_waveform_compare.png')
                            plt.close(fig)
                    
                    # 检查梯度状态
                    print(f"Epoch {epoch}: outputs_eps.grad is None: {outputs_eps_use.grad is None}")
                    print(f"Epoch {epoch}: outputs_eps.requires_grad: {outputs_eps_use.requires_grad}")
                    
                    # 可视化梯度
                    if outputs_eps_use.grad is not None:
                        print(f"Epoch {epoch}: 开始绘制梯度...")
                        grad_np = outputs_eps_use.grad.detach().cpu().numpy()
                        print(f"Epoch {epoch}: 梯度形状: {grad_np.shape}")
                        print(f"Epoch {epoch}: 梯度范围: [{np.min(grad_np):.6f}, {np.max(grad_np):.6f}]")

                        
                        save_gradient_figure(
                            grad_np,
                            f'{path}epoch_{epoch}_epsilon_grad.png',
                            f'Epsilon Gradient (Normed, epoch {epoch})',
                            dx=dx,
                            dz=dz,
                        )
                        print(f"Epoch {epoch}: 已保存归一化梯度图")
                        
                        # 保存归一化梯度数据
                        #np.save(f'{path}epoch_{epoch}_epsilon_grad.npy', grad_np_norm)
                        print(f"Epoch {epoch}: 已保存归一化梯度数据")
                    else:
                        print(f"Epoch {epoch}: 梯度为None，无法绘制")

                    # 可视化 Sigma 梯度与 TV 梯度
                    if outputs_sig_use.grad is not None:
                        print(f"Epoch {epoch}: 开始绘制Sigma梯度...")
                        grad_sig_np = outputs_sig_use.grad.detach().cpu().numpy()
                        print(f"Epoch {epoch}: Sigma梯度形状: {grad_sig_np.shape}")
                        print(f"Epoch {epoch}: Sigma梯度范围: [{np.min(grad_sig_np):.6f}, {np.max(grad_sig_np):.6f}]")

                        save_gradient_figure(
                            grad_sig_np,
                            f'{path}epoch_{epoch}_sigma_grad.png',
                            f'Sigma Gradient (Normed, epoch {epoch})',
                            dx=dx,
                            dz=dz,
                        )
                        print(f"Epoch {epoch}: 已保存Sigma归一化梯度图")

                        if train_sigma:
                            sigma_tv_grad_np = compute_tv_gradient(outputs_sig_use.detach().cpu().numpy(), dx, dz)
                            sigma_tv_grad_norm, sigma_tv_grad_max = normalize_gradient_for_visualization(sigma_tv_grad_np)
                            save_gradient_figure(
                                sigma_tv_grad_norm,
                                f'{path}epoch_{epoch}_sigma_tv_grad_norm.png',
                                f'Sigma TV Gradient (Normed, epoch {epoch})',
                                dx=dx,
                                dz=dz,
                            )
                            np.save(f'{path}epoch_{epoch}_sigma_tv_grad_norm.npy', sigma_tv_grad_norm)
                            print(f"Epoch {epoch}: 已保存Sigma TV梯度图与数据, max_abs={sigma_tv_grad_max:.6e}")
                    
                    # 保存数据
                    #np.save(f'{path}epoch_{epoch}_epsilon.npy', outputs_eps_use.detach().cpu().numpy())
                    print(f"Epoch {epoch}: 已保存epsilon数据")

                if i == 0 and (epoch % 200 == 0) and d_syn.dim() == 3 and selected_d_obs_local.dim() == 3:
                    d_syn_np_vis = d_syn.detach().cpu().numpy()
                    d_obs_np_vis = selected_d_obs_local.detach().cpu().numpy()
                    n_local_shots = d_syn_np_vis.shape[0]
                    n_receivers = d_syn_np_vis.shape[1]
                    n_steps = d_syn_np_vis.shape[2]
                    if n_local_shots > 0 and n_receivers > 0:
                        t = np.arange(n_steps) * dt
                        k_shots = min(10, n_local_shots)
                        shot_indices = np.unique(np.linspace(0, n_local_shots - 1, k_shots, dtype=int))
                        rec_mid = n_receivers // 2
                        fig, axes = plt.subplots(len(shot_indices), 1, figsize=(8, 2 * len(shot_indices)), sharex=True)
                        if len(shot_indices) == 1:
                            axes = [axes]
                        for ax, s_idx in zip(axes, shot_indices):
                            g_shot = start_idx + int(s_idx)
                            ax.plot(t, d_obs_np_vis[s_idx, rec_mid, :], 'k-', linewidth=0.8, label='d_obs')
                            ax.plot(t, d_syn_np_vis[s_idx, rec_mid, :], 'r--', linewidth=0.8, label='d_syn')
                            ax.set_ylabel(f'Shot {g_shot}')
                            ax.grid(True, alpha=0.3)
                        axes[-1].set_xlabel('Time')
                        axes[0].legend(loc='upper right')
                        fig.tight_layout()
                        fig.savefig(f'{path}epoch_{epoch}_trace_compare_10shots_rec{rec_mid}.png')
                        plt.close(fig)

                        k_recs = min(10, n_receivers)
                        rec_indices = np.unique(np.linspace(0, n_receivers - 1, k_recs, dtype=int))
                        shot_mid = n_local_shots // 2
                        g_shot = start_idx + int(shot_mid)
                        fig, axes = plt.subplots(len(rec_indices), 1, figsize=(8, 2 * len(rec_indices)), sharex=True)
                        if len(rec_indices) == 1:
                            axes = [axes]
                        for ax, r_idx in zip(axes, rec_indices):
                            ax.plot(t, d_obs_np_vis[shot_mid, r_idx, :], 'k-', linewidth=0.8, label='d_obs')
                            ax.plot(t, d_syn_np_vis[shot_mid, r_idx, :], 'r--', linewidth=0.8, label='d_syn')
                            ax.set_ylabel(f'Rec {int(r_idx)}')
                            ax.grid(True, alpha=0.3)
                        axes[-1].set_xlabel('Time')
                        axes[0].legend(loc='upper right')
                        fig.suptitle(f'Shot {g_shot} waveform compare')
                        fig.tight_layout()
                        fig.savefig(f'{path}epoch_{epoch}_shot_{g_shot}_trace_compare_10recs.png')
                        plt.close(fig)

                if i == 0 and (epoch % 100 == 0):
                    sig_est = outputs_sig_use.detach().cpu().numpy()
                    sig_true_np = sigma_true.detach().cpu().numpy()
                    nz, nx = sig_est.shape
                    kx = min(20, nx)
                    x_indices = np.unique(np.linspace(0, nx - 1, kx, dtype=int))
                    z = np.arange(nz) * dz
                    nrows = 4
                    ncols = int(np.ceil(len(x_indices) / nrows))
                    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 10), sharey=True)
                    axes = np.array(axes).reshape(-1)
                    for ax in axes[len(x_indices):]:
                        ax.axis('off')
                    for ax, x_idx in zip(axes, x_indices):
                        ax.plot(sig_true_np[:, x_idx], z, 'k-', linewidth=0.7, label='true')
                        ax.plot(sig_est[:, x_idx], z, 'r--', linewidth=0.7, label='inv')
                        ax.set_title(f'x={int(x_idx)}')
                        ax.grid(True, alpha=0.3)
                        ax.invert_yaxis()
                    axes[0].legend(loc='upper right')
                    fig.suptitle(f'Sigma vertical profiles (epoch {epoch})')
                    fig.tight_layout()
                    fig.savefig(f'{path}epoch_{epoch}_sigma_vertical_profiles_20x.png')
                    plt.close(fig)

                if i == 0 and (epoch % args.snapshot_interval == 0):
                    np.save(f'{path}epoch_{epoch}_epsilon.npy', outputs_eps_use.detach().cpu().numpy())
                    np.save(f'{path}epoch_{epoch}_sigma.npy', outputs_sig_use.detach().cpu().numpy())

        # 记录损失（仅root）
        if rank == 0 and total_samples > 0:
            epoch_loss = epoch_loss / total_samples
            epoch_data_misfit = (
                epoch_data_misfit_sum / epoch_metric_samples
                if epoch_metric_samples > 0
                else float("nan")
            )
            epoch_l1_data = (
                epoch_l1_data_sum / epoch_metric_samples
                if epoch_metric_samples > 0
                else float("nan")
            )
            loss_model_eps = loss_fn(outputs_eps_use, epsilon_true) / loss_fn(epsilon_true, torch.zeros_like(epsilon_true))
            loss_model_sig = loss_fn(outputs_sig_use, sigma_true) / loss_fn(sigma_true, torch.zeros_like(sigma_true))
            eps_residual_mae = F.l1_loss(outputs_eps_use, epsilon_true)
            sig_residual_mae = F.l1_loss(outputs_sig_use, sigma_true)
            model_residual_score = loss_model_eps + loss_model_sig
            model_quality_metrics = compute_model_quality_metrics(
                outputs_eps_use.detach().cpu().numpy(),
                epsilon_true.detach().cpu().numpy(),
                outputs_sig_use.detach().cpu().numpy(),
                sigma_true.detach().cpu().numpy(),
                fixed_top_rows=l_size,
            )

            loss_history.append(epoch_loss)
            eps_loss_history.append(loss_model_eps.item())
            metrics_row = {
                "epoch": int(epoch),
                "total_loss": float(epoch_loss),
                "data_misfit_mse": float(epoch_data_misfit),
                "data_l1": float(epoch_l1_data),
                "eps_model_loss": float(loss_model_eps.item()),
                "sig_model_loss": float(loss_model_sig.item()),
                "eps_residual_mae": float(eps_residual_mae.item()),
                "sig_residual_mae": float(sig_residual_mae.item()),
                "model_residual_score": float(model_residual_score.item()),
                "learning_rate_eps": float(learning_rate_eps),
                "lr_eps_stage2": float(learning_rate_eps_stage2),
                "learning_rate_sig": float(learning_rate_sig),
                "alpha_tv_eps": float(alpha_tv_eps),
                "alpha_tv_sig": float(alpha_tv_sig),
                "sigma_beta": float(sigma_beta),
                "sigma_param": sigma_param,
                "sigma_tv_domain": sigma_tv_domain,
                "sigma_shallow_skip": bool(args.sigma_shallow_skip),
                "alpha_l1_data": float(alpha_l1_data),
                "stage1_data_loss": stage1_data_loss,
                "stage2_data_loss": stage2_data_loss,
                "current_data_loss_mode": current_data_loss_mode,
                "alpha_tikhonov_eps": float(alpha_tikhonov_eps),
                "alpha_tikhonov_sig": float(alpha_tikhonov_sig),
            }
            metrics_row.update(model_quality_metrics)
            metrics_history.append(metrics_row)
            write_metrics_files(path, metrics_history)

            print(f'Epoch {epoch}: Total Loss = {epoch_loss:.6f}, Data Mode = {current_data_loss_mode}, Data MSE = {epoch_data_misfit:.6f}, Data L1 = {epoch_l1_data:.6f}, Eps Loss = {loss_model_eps:.6f}, Sig Loss = {loss_model_sig:.6f}, Active Objective = {model_quality_metrics["model_quality_objective_active"]:.6f}, Active SSIM eps/sig = {model_quality_metrics["eps_model_active_ssim"]:.4f}/{model_quality_metrics["sig_model_active_ssim"]:.4f}')

            if epoch % 100 == 0 and epoch > 0:
                plot_loss_curves(loss_history, eps_loss_history, path, epoch)
        
        # 同步所有进程
        comm.Barrier()
        
        # 保持仅在rank 0更新网络参数，其他进程不持有网络
        # 无需在此同步网络权重

    # 训练结束后绘制最终的loss曲线（仅root）
    if rank == 0:
        if (not args.resume) and num_epochs_stage2 == 0:
            if eps_model_for_sig is None:
                raise RuntimeError("eps_model_for_sig is None after stage1")
            eps_min = eps_model_for_sig.min()
            eps_max = eps_model_for_sig.max()
            eps_norm = (eps_model_for_sig - eps_min) / (eps_max - eps_min + 1e-8)
            eps_norm_padded = F.pad(eps_norm, (pad_width//2, pad_width//2, pad_height//2, pad_height//2))
            net2_input_fixed = eps_norm_padded.unsqueeze(0).unsqueeze(0).to(device).detach()
            torch.save(
                {
                    "net1_state_dict": net_1.state_dict() if net_1 is not None else None,
                    "net2_input": net2_input_fixed.detach().cpu(),
                    "args": vars(args),
                    "num_epochs_stage1": num_epochs_stage1,
                },
                stage1_checkpoint_save_path,
            )
            print(f"Stage1 checkpoint saved to: {stage1_checkpoint_save_path}")
        print("绘制最终loss曲线...")
        plot_loss_curves(loss_history, eps_loss_history, path, num_epochs - 1)
        write_metrics_files(path, metrics_history)
        print("训练完成！最终loss曲线已保存。")

if __name__ == "__main__":
    main()
    
    
