#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量化的全波形反演验证实验脚本

与 main.py 结构保持一致，但将模型尺寸缩减到 100 x 500，
并构造了一个简单的合成模型用于 CPU/GPU 结果交叉验证。
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Function
import torch.utils.data as data_utils
import os
import shutil
import sys
import random
import time
from scipy.ndimage import gaussian_filter

from utils.unet_test import UNetTest

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

    class _DummyComm:
        def Barrier(self):
            pass

        def bcast(self, x, root=0):
            return x

        def gather(self, x, root=0):
            return [x]

        def allreduce(self, x, op=None):
            return x

        def Allreduce(self, sendbuf, recvbuf, op=None):
            np.copyto(recvbuf, sendbuf)

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

# ========== 常量设定 ==========
BACKGROUND_VALUE = 6.0
SQUARE_VALUE = 3.0
BOTTOM_VALUE = 1.5
SQUARE_SIZE = 20
BOTTOM_THICKNESS = 10

MODEL_DEPTH = 100
MODEL_WIDTH = 500
WINDOW_DEPTH = 100
WINDOW_WIDTH = 200
PADDING_WIDTH = 100
TOTAL_SHOTS = MODEL_WIDTH  # 每个水平位置一个炮点

DT = 1.4e-10
DX = 0.06
DZ = 0.06
NPML = 10
FREQ = 1.5e8

C0 = 2.99792458e8
BACKGROUND_SPEED = C0 / np.sqrt(BACKGROUND_VALUE)
MAX_TRAVEL_DISTANCE = 2.0 * MODEL_DEPTH * DZ
TIME_WINDOW = 1.3 * MAX_TRAVEL_DISTANCE / BACKGROUND_SPEED
STEPS = int(np.ceil(TIME_WINDOW / DT))
STEPS = max(STEPS, 1000)

MIN_EPS = 1.0
MAX_EPS = 9.0



from utils.forward import forward_model
from utils.gradient import (
    compute_gradient,
    compute_tikhonov_gradient,
    compute_laplacian_2d as compute_laplacian_2d_gpu,
)

# ========== 全局变量：子波缓存 ==========
_global_wavelet = None


# ========== 辅助函数 ==========
def extract_window(model_padded, source_x_global, window_depth=WINDOW_DEPTH, window_width=WINDOW_WIDTH):
    source_x_padded = source_x_global + PADDING_WIDTH
    window_center_x = source_x_padded
    offset_x = window_center_x - window_width // 2
    offset_x = max(0, min(offset_x, model_padded.shape[1] - window_width))
    window = model_padded[:window_depth, offset_x : offset_x + window_width]
    return window, offset_x


def sync_padding_with_edge(model, padding_width=PADDING_WIDTH):
    model_synced = model.copy()
    model_synced[:, :padding_width] = model[:, padding_width : padding_width + 1]
    model_synced[:, -padding_width:] = model[:, -padding_width - 1 : -padding_width]
    return model_synced


class ForwardModelFunction(Function):
    @staticmethod
    def forward(ctx, epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps):
        if epsilon.is_cuda:
            epsilon_input = epsilon
            sigma_input = sigma
        else:
            epsilon_input = epsilon.detach().cpu().numpy()
            sigma_input = sigma.detach().cpu().numpy()

        # 执行正演模拟
        current_wavelet = _global_wavelet if _global_wavelet is not None else None
        data, wave_field = forward_model(
            epsilon_input,
            sigma_input,
            source_list,
            receiver_list,
            dt,
            dx,
            dz,
            npml,
            freq,
            steps,
            save_wavefield=True,
            wavelet=current_wavelet,
        )

        data_tensor = torch.from_numpy(data).to(epsilon.device).float()

        ctx.save_for_backward(epsilon, sigma)
        ctx.source_list = source_list
        ctx.receiver_list = receiver_list
        ctx.dt = dt
        ctx.dx = dx
        ctx.dz = dz
        ctx.npml = npml
        ctx.freq = freq
        ctx.steps = steps
        ctx.wave_field = wave_field

        return data_tensor

    @staticmethod
    def backward(ctx, grad_output):
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

        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()
        grad_output_np = grad_output.detach().cpu().numpy()

        grad_eps, _ = compute_gradient(
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
            sigma_required_gradient=False,
            wavefield_data=wave_field,
        )

        grad_eps_tensor = torch.from_numpy(grad_eps).to(epsilon.device)

        try:
            if MPI_AVAILABLE and comm.Get_size() > 1:
                grad_local = grad_eps_tensor.detach().cpu().numpy()
                grad_global = np.empty_like(grad_local)
                comm.Allreduce(grad_local, grad_global, op=MPI.SUM)
                grad_global = grad_global / float(comm.Get_size())
                grad_eps_tensor = torch.from_numpy(grad_global).to(epsilon.device)
        except Exception:
            pass

        return grad_eps_tensor, None, None, None, None, None, None, None, None, None


class TikhonovRegularizationFunction(Function):
    @staticmethod
    def forward(ctx, epsilon, sigma, alpha_tikhonov_eps, alpha_tikhonov_sig, dx, dz):
        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()

        laplacian_eps_np = compute_laplacian_2d_gpu(epsilon_np, dx, dz)
        laplacian_sig_np = compute_laplacian_2d_gpu(sigma_np, dx, dz)

        laplacian_eps = torch.from_numpy(laplacian_eps_np).to(epsilon.device)
        laplacian_sig = torch.from_numpy(laplacian_sig_np).to(sigma.device)

        reg_loss_eps = 1e-10 * alpha_tikhonov_eps * torch.sum(laplacian_eps**2)
        reg_loss_sig = 1e-10 * alpha_tikhonov_sig * torch.sum(laplacian_sig**2)

        reg_loss = reg_loss_eps + reg_loss_sig

        ctx.save_for_backward(epsilon, sigma, laplacian_eps, laplacian_sig)
        ctx.alpha_tikhonov_eps = alpha_tikhonov_eps
        ctx.alpha_tikhonov_sig = alpha_tikhonov_sig
        ctx.dx = dx
        ctx.dz = dz

        return reg_loss

    @staticmethod
    def backward(ctx, grad_output):
        epsilon, sigma, laplacian_eps, laplacian_sig = ctx.saved_tensors
        alpha_tikhonov_eps = ctx.alpha_tikhonov_eps
        alpha_tikhonov_sig = ctx.alpha_tikhonov_sig
        dx = ctx.dx
        dz = ctx.dz

        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()

        tikhonov_grad_eps = compute_tikhonov_gradient(epsilon_np, dx, dz, alpha_tikhonov_eps)
        tikhonov_grad_sig = compute_tikhonov_gradient(sigma_np, dx, dz, alpha_tikhonov_sig)

        grad_eps = torch.from_numpy(tikhonov_grad_eps).to(epsilon.device) * grad_output
        grad_sig = torch.from_numpy(tikhonov_grad_sig).to(sigma.device) * grad_output

        return grad_eps, grad_sig, None, None, None, None


def tikhonov_reg(epsilon, sigma, alpha_tikhonov_eps, alpha_tikhonov_sig, dx=DX, dz=DZ):
    return TikhonovRegularizationFunction.apply(epsilon, sigma, alpha_tikhonov_eps, alpha_tikhonov_sig, dx, dz)


def grad_hook(grad):
    if torch.isnan(grad).any() or torch.isinf(grad).any():
        print("警告：梯度包含nan或inf值！")
        grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)

    grad[:5, :] = 0.0

    max_abs_value = torch.max(torch.abs(grad)).item()
    if max_abs_value != 0:
        grad /= max_abs_value

    grad = torch.clamp(grad, -1.0, 1.0)
    return grad


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_clean_dir(target_dir):
    if os.path.exists(target_dir):
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
        else:
            os.remove(target_dir)
    os.makedirs(target_dir, exist_ok=True)


def generate_models():
    true_model = np.full((MODEL_DEPTH, MODEL_WIDTH), BACKGROUND_VALUE, dtype=np.float32)
    true_model[-BOTTOM_THICKNESS:, :] = BOTTOM_VALUE

    n_cols = 5
    anomaly_left_positions = np.linspace(0, MODEL_WIDTH - SQUARE_SIZE, num=n_cols, dtype=int)
    anomaly_top_layers = [40, 60]

    for top in anomaly_top_layers:
        bottom = top + SQUARE_SIZE
        for left in anomaly_left_positions:
            right = left + SQUARE_SIZE
            true_model[top:bottom, left:right] = SQUARE_VALUE

    initial_model = np.full((MODEL_DEPTH, MODEL_WIDTH), BACKGROUND_VALUE, dtype=np.float32)
    initial_model[-BOTTOM_THICKNESS:, :] = BOTTOM_VALUE

    return true_model, initial_model


def eps_to_sig(epsilon_r, freq=FREQ):
    ep0 = 8.841941282883074e-12
    epsilon = epsilon_r * ep0
    index = 0.440 * np.log(epsilon_r) / np.log(1.93) - 2.943
    sig = 2 * np.pi * freq * epsilon * 10 ** (index)
    return sig.astype(np.float32)


def select_minibatch_shots(n_total_shots=TOTAL_SHOTS, stride=10):
    z = random.randint(0, stride - 1)
    selected_indices = [i for i in range(n_total_shots) if i % stride == z]
    return selected_indices


def plot_model(model, path, title, vmin=MIN_EPS, vmax=MAX_EPS):
    plt.figure(figsize=(12, 4))
    plt.imshow(model, aspect="auto", cmap="jet", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_ricker_wavelet(frequency, dt, steps, peak_time=None):
    """Generate a Ricker wavelet (Mexican hat) with the given center frequency."""
    t = np.arange(steps, dtype=np.float64) * dt
    if peak_time is None:
        peak_time = 3.0 / frequency  # place the peak early to fit within the window
    t_shift = t - peak_time
    pi_f = np.pi * frequency
    wavelet = (1.0 - 2.0 * (pi_f ** 2) * (t_shift ** 2)) * np.exp(-(pi_f ** 2) * (t_shift ** 2))
    return wavelet.astype(np.float32)


def main():
    """
    主函数支持两种运行模式：
    - 模式1（generate_mode=True）：仅生成合成观测数据并保存
    - 模式2（generate_mode=False）：读取已有观测数据，进行全波形反演
    """
    import argparse

    parser = argparse.ArgumentParser(description="全波形反演验证程序 (100x500)")
    parser.add_argument(
        "--mode",
        type=str,
        default="inversion",
        choices=["generate", "inversion"],
        help="运行模式：generate=生成观测数据, inversion=执行反演",
    )
    parser.add_argument(
        "--d_obs_file",
        type=str,
        default="data_small/d_obs_small.npy",
        help="观测数据文件路径（inversion模式使用）",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="指定使用的GPU设备ID，如果不指定则自动分配",
    )
    args = parser.parse_args()

    generate_mode = args.mode == "generate"
    d_obs_file = args.d_obs_file
    gpu_device_id = args.gpu

    if rank == 0:
        print("=" * 70)
        if generate_mode:
            print("模式1：生成合成观测数据")
        else:
            print("模式2：执行全波形反演")
            print(f"读取观测数据: {d_obs_file}")
        print("=" * 70)

    same_seeds(2025)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if rank == 0:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_output_dir = os.path.join(script_dir, "result_gpr_mpi_test")
        data_dir = os.path.join(script_dir, "data_small")
        os.makedirs(data_dir, exist_ok=True)
        try:
            ensure_clean_dir(base_output_dir)
        except PermissionError:
            home_dir = os.path.expanduser("~")
            base_output_dir = os.path.join(home_dir, "result_gpr_mpi_test")
            ensure_clean_dir(base_output_dir)
        print(f"结果将保存到: {base_output_dir}")
    else:
        base_output_dir = None
        data_dir = None

    base_output_dir = comm.bcast(base_output_dir if rank == 0 else None, root=0)
    data_dir = comm.bcast(data_dir if rank == 0 else None, root=0)

    comm.Barrier()

    device = torch.device("cpu")
    if rank == 0:
        print("\n当前配置:")
        print(f"  MPI进程数: {size}")
        print("  GPU加速: ✗ 未启用 (CPU模式)")
        print("=" * 70)

    learning_rate = 1e-3
    alpha_tikhonov_eps = 0.01
    offset = 0.0

    result_dir = None
    if rank == 0:
        result_dir = os.path.join(base_output_dir, f"lr{learning_rate:.0e}_alpha{alpha_tikhonov_eps:.3g}")
        ensure_clean_dir(result_dir)
        if not result_dir.endswith(os.sep):
            result_dir += os.sep
        print(f"结果保存路径: {result_dir}")
    result_dir = comm.bcast(result_dir if rank == 0 else None, root=0)

    true_model, input_model = generate_models()
    true_model_pad = np.pad(true_model, ((0, 0), (PADDING_WIDTH, PADDING_WIDTH)), mode="edge")
    input_model_pad = np.pad(input_model, ((0, 0), (PADDING_WIDTH, PADDING_WIDTH)), mode="edge")

    # 只在需要时加载到GPU（正演模拟时），避免所有进程同时占用GPU内存
    epsilon_true_pad = torch.tensor(true_model_pad, dtype=torch.float32)
    sigma_true_pad = torch.full_like(epsilon_true_pad, 0.005)
    
    if rank == 0 and generate_mode:
        np.save(os.path.join(data_dir, "true_model.npy"), true_model)
        np.save(os.path.join(data_dir, "initial_model.npy"), input_model)

    if rank == 0:
        plot_model(true_model, os.path.join(result_dir, "epsilon_true.png"), "True Epsilon Model", vmin=MIN_EPS, vmax=MAX_EPS)
        plot_model(np.full_like(true_model, 0.005), os.path.join(result_dir, "sigma_true.png"), "True Sigma Model", vmin=0.0, vmax=0.01)
        
        # 保存UNet输入模型图像（使用相同的颜色映射以便比较）
        plot_model(input_model, os.path.join(result_dir, "unet_input_model.png"), "UNet Input Model (Background + Bottom Layer)", vmin=MIN_EPS, vmax=MAX_EPS)

    wavelet_to_use = None
    if rank == 0:
        print("生成1.5e8 Hz Ricker子波作为震源...")
    global _global_wavelet
    wavelet_to_use = generate_ricker_wavelet(FREQ, DT, STEPS)
    if rank == 0:
        wave_min, wave_max = float(wavelet_to_use.min()), float(wavelet_to_use.max())
        print(f"子波形状: {wavelet_to_use.shape}, 数据范围: [{wave_min:.6e}, {wave_max:.6e}]")
        plt.figure(figsize=(10, 4))
        plt.plot(wavelet_to_use, label="Ricker Wavelet", color="orange")
        plt.title("Generated Ricker Wavelet")
        plt.xlabel("Time Step")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "wavelet_ricker.png"), dpi=300)
        plt.close()
        np.save(os.path.join(result_dir, "wavelet_ricker.npy"), wavelet_to_use)
        np.save(os.path.join(result_dir, "wavelet_used.npy"), wavelet_to_use)

    wavelet = comm.bcast(wavelet_to_use if rank == 0 else None, root=0)
    _global_wavelet = wavelet
    if rank == 0:
        print(f"最终使用的子波范围: [{float(wavelet.min()):.6e}, {float(wavelet.max()):.6e}]，负值数量 {int(np.count_nonzero(wavelet < 0))}")

    source_list_full = [(0, i) for i in range(0, MODEL_WIDTH)]

    if generate_mode:
        if rank == 0:
            print("=" * 70)
            print("使用移动窗口策略生成合成观测数据...")
            print("=" * 70)

        n_shots_total = len(source_list_full)
        shots_per_proc = n_shots_total // size
        remainder = n_shots_total % size

        if rank < remainder:
            start_shot = rank * (shots_per_proc + 1)
            end_shot = start_shot + shots_per_proc + 1
        else:
            start_shot = rank * shots_per_proc + remainder
            end_shot = start_shot + shots_per_proc

        local_shots = list(range(start_shot, end_shot))

        if rank == 0:
            print(f"总炮点数: {n_shots_total}")
            print(f"MPI进程数: {size}")
            print(f"每个进程处理约 {shots_per_proc} 个炮点")

        print(f"Rank {rank}: 负责炮点 {start_shot} 到 {end_shot - 1} (共 {len(local_shots)} 个)")

        local_d_obs_list = []

        for shot_idx in local_shots:
            source_x = shot_idx

            epsilon_window_np, offset_x = extract_window(epsilon_true_pad.cpu().numpy(), source_x)
            sigma_window_np, _ = extract_window(sigma_true_pad.cpu().numpy(), source_x)

            window_source = [(0, WINDOW_WIDTH // 2)]
            window_receiver = window_source.copy()

            d_obs_shot = forward_model(
                epsilon_window_np,
                sigma_window_np,
                window_source,
                window_receiver,
                DT,
                DX,
                DZ,
                NPML,
                FREQ,
                STEPS,
                save_wavefield=False,
                wavelet=wavelet,
            )

            local_d_obs_list.append(d_obs_shot[0])

            if (shot_idx - start_shot) % 20 == 0:
                print(f"Rank {rank}: 已处理 {shot_idx - start_shot + 1}/{len(local_shots)} 个炮点")

        local_d_obs_array = np.stack(local_d_obs_list, axis=0)
        print(f"Rank {rank}: 完成正演，本地数据形状: {local_d_obs_array.shape}")

        comm.Barrier()

        all_d_obs_list = comm.gather(local_d_obs_array, root=0)

        if rank == 0:
            d_obs_np = np.concatenate(all_d_obs_list, axis=0)
            print(f"\n合成观测数据生成完成！形状: {d_obs_np.shape}")
            np.save(os.path.join(data_dir, "d_obs_small.npy"), d_obs_np)
            plt.figure(figsize=(12, 6))
            vmax = np.percentile(np.abs(d_obs_np), 98)
            plt.imshow(d_obs_np.T, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax)
            plt.xlabel("Shot Number")
            plt.ylabel("Time Step")
            plt.title("Synthetic Observed Data (100x500)")
            plt.colorbar(label="Amplitude")
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, "d_obs_synthetic.png"), dpi=300)
            plt.close()
            print(f"✓ 合成观测数据已保存到: {os.path.join(data_dir, 'd_obs_small.npy')}")
            print("如需执行反演，请使用: python main_test.py --mode inversion")
        comm.Barrier()
        return

    else:
        if not os.path.exists(d_obs_file):
            if rank == 0:
                print(f"错误：找不到观测数据文件 {d_obs_file}")
                print("请先运行模式1生成观测数据：python main_test.py --mode generate")
            comm.Barrier()
            return

        d_obs_np = np.load(d_obs_file)

        if rank == 0:
            print(f"观测数据形状: {d_obs_np.shape}")
            print(f"数据类型: {d_obs_np.dtype}")
            print(f"数据范围: [{d_obs_np.min():.6e}, {d_obs_np.max():.6e}]")
            if d_obs_np.shape != (MODEL_WIDTH, STEPS):
                print(f"警告：观测数据形状 {d_obs_np.shape} 与预期 ({MODEL_WIDTH}, {STEPS}) 不匹配")
                if d_obs_np.shape[1] != STEPS:
                    d_obs_np = d_obs_np[:, :STEPS]
            obs_img_path = os.path.join(result_dir, "d_obs_synthetic.png")
            if not os.path.exists(obs_img_path):
                plt.figure(figsize=(12, 6))
                vmax = np.percentile(np.abs(d_obs_np), 98)
                plt.imshow(d_obs_np.T, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax)
                plt.xlabel("Shot Number")
                plt.ylabel("Time Step")
                plt.title("Observed Data (100x500)")
                plt.colorbar(label="Amplitude")
                plt.tight_layout()
                plt.savefig(obs_img_path, dpi=300)
                plt.close()
                print(f"✓ 观测数据图像已保存: {obs_img_path}")
            print("=" * 70)

        if MPI_AVAILABLE and size > 1:
            if rank != 0:
                d_obs_np = None
            d_obs_np = comm.bcast(d_obs_np, root=0)

    d_obs_tensor = torch.tensor(d_obs_np, dtype=torch.float32, device=device)

    if rank == 0:
        print(f"观测数据张量形状: {d_obs_tensor.shape}")
        print(f"观测数据张量设备: {d_obs_tensor.device}")

    reduced_encoder_channels = [8, 16, 32, 64, 128]
    reduced_decoder_channels = [128, 64, 32, 16, 8]

    net_1 = None
    if rank == 0:
        print("初始化UNet网络 (100x500)...")
        net_1 = UNetTest(
            in_channels=1,
            out_channels=1,
            encoder_channels=reduced_encoder_channels,
            decoder_channels=reduced_decoder_channels,
            target_height=MODEL_DEPTH,
            target_width=MODEL_WIDTH,
        ).to(device)
        print(f"网络参数数量: {get_parameter_number(net_1)}")

    target_height, target_width = MODEL_DEPTH, MODEL_WIDTH
    required_height = ((target_height + 15) // 16) * 16
    required_width = ((target_width + 15) // 16) * 16

    pad_height = required_height - target_height
    pad_width = required_width - target_width
    if pad_height % 2 == 1:
        pad_height += 1
        required_height += 1
    if pad_width % 2 == 1:
        pad_width += 1
        required_width += 1

    if rank == 0:
        print(f"UNet输入padding: height={pad_height}, width={pad_width}")
        print(f"UNet输入尺寸: ({required_height}, {required_width})")

    input_model_padded = np.pad(
        input_model,
        ((pad_height // 2, pad_height // 2), (pad_width // 2, pad_width // 2)),
        mode="edge",
    )

    net1_input = torch.tensor(input_model_padded, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    if rank == 0:
        print(f"网络输入形状: {net1_input.shape}")

    optimizer = None
    if rank == 0 and net_1 is not None:
        optimizer = torch.optim.Adam(
            [
                {"params": net_1.parameters(), "lr": learning_rate},
            ]
        )

    loss_fn = torch.nn.MSELoss()
    num_epochs = 501

    loss_history = []

    if rank == 0:
        print("开始训练...")

    for epoch in range(num_epochs):
        if rank == 0 and net_1 is not None:
            net_1.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.no_grad():
                _, _, net1_output_vis = net_1(net1_input)
                epsilon_big = torch.sigmoid(net1_output_vis + offset) * (MAX_EPS - MIN_EPS) + MIN_EPS

            epsilon_big_pad = F.pad(epsilon_big, (PADDING_WIDTH, PADDING_WIDTH), mode="replicate")
            sigma_big_pad = torch.full_like(epsilon_big_pad, 0.005)
        else:
            epsilon_big_pad = None
            sigma_big_pad = None

        if MPI_AVAILABLE and size > 1:
            epsilon_big_pad_np = epsilon_big_pad.cpu().numpy() if rank == 0 else None
            sigma_big_pad_np = sigma_big_pad.cpu().numpy() if rank == 0 else None

            epsilon_big_pad_np = comm.bcast(epsilon_big_pad_np, root=0)
            sigma_big_pad_np = comm.bcast(sigma_big_pad_np, root=0)

            if rank != 0:
                epsilon_big_pad = torch.tensor(epsilon_big_pad_np, dtype=torch.float32, device=device)
                sigma_big_pad = torch.tensor(sigma_big_pad_np, dtype=torch.float32, device=device)

        if rank == 0:
            selected_shot_indices = select_minibatch_shots(n_total_shots=TOTAL_SHOTS, stride=10)
            print(f"Epoch {epoch}: 选择了 {len(selected_shot_indices)} 个炮点")
        else:
            selected_shot_indices = None

        selected_shot_indices = comm.bcast(selected_shot_indices, root=0)

        n_selected = len(selected_shot_indices)
        shots_per_proc_mini = n_selected // size
        remainder_mini = n_selected % size

        if rank < remainder_mini:
            mini_start = rank * (shots_per_proc_mini + 1)
            mini_end = mini_start + shots_per_proc_mini + 1
        else:
            mini_start = rank * shots_per_proc_mini + remainder_mini
            mini_end = mini_start + shots_per_proc_mini

        local_selected_shots = selected_shot_indices[mini_start:mini_end]

        if rank == 0:
            print(f"Epoch {epoch}: 每个进程处理约 {shots_per_proc_mini} 个炮点")
        print(f"Rank {rank}: 处理炮点 {mini_start} 到 {mini_end - 1} (共 {len(local_selected_shots)} 个)")

        accumulated_grad_eps = torch.zeros((MODEL_DEPTH, MODEL_WIDTH + 2 * PADDING_WIDTH), dtype=torch.float32, device=device)

        epoch_loss = 0.0
        n_shots_processed = 0

        if rank == 0 and optimizer is not None:
            optimizer.zero_grad()

        # 初始化单炮梯度存储（仅rank 0）
        individual_grads = []
        individual_shot_indices = []
        
        # 定义中央炮点位置
        central_shot_idx = MODEL_WIDTH // 2
        
        for shot_idx in local_selected_shots:
            source_x = shot_idx

            source_x_padded = source_x + PADDING_WIDTH
            window_center_x = source_x_padded
            offset_x = window_center_x - WINDOW_WIDTH // 2
            offset_x = max(0, min(offset_x, epsilon_big_pad.shape[1] - WINDOW_WIDTH))

            epsilon_window = epsilon_big_pad[:WINDOW_DEPTH, offset_x : offset_x + WINDOW_WIDTH].clone().detach()
            sigma_window = sigma_big_pad[:WINDOW_DEPTH, offset_x : offset_x + WINDOW_WIDTH].clone().detach()
            epsilon_window.requires_grad_(True)

            window_source = [(0, WINDOW_WIDTH // 2)]
            window_receiver = window_source.copy()

            d_obs_shot = d_obs_tensor[shot_idx : shot_idx + 1, :]

            d_syn_shot = ForwardModelFunction.apply(
                epsilon_window,
                sigma_window,
                window_source,
                window_receiver,
                DT,
                DX,
                DZ,
                NPML,
                FREQ,
                STEPS,
            )

            if rank == 0 and (shot_idx - local_selected_shots[0]) < 2:
                dmin = d_syn_shot.min().item()
                dmax = d_syn_shot.max().item()
                print(f"[调试 epoch {epoch}] Shot {shot_idx} d_syn范围: [{dmin:.3e}, {dmax:.3e}]")

            data_loss = loss_fn(d_syn_shot, d_obs_shot)
            data_loss.backward()

            if epsilon_window.grad is not None:
                # 将窗口梯度映射到完整模型的对应位置
                accumulated_grad_eps[:WINDOW_DEPTH, offset_x : offset_x + WINDOW_WIDTH] += epsilon_window.grad.detach()
                
                # 保存单炮梯度（rank 0 only）
                if rank == 0:
                    # 创建完整模型大小的单炮梯度图
                    shot_grad_full = np.zeros((MODEL_DEPTH, MODEL_WIDTH + 2 * PADDING_WIDTH), dtype=np.float32)
                    shot_grad_full[:WINDOW_DEPTH, offset_x:offset_x + WINDOW_WIDTH] = epsilon_window.grad.detach().cpu().numpy()
                    individual_grads.append(shot_grad_full)
                    individual_shot_indices.append(shot_idx)
                
                # 保存中央炮点的梯度用于可视化
                if shot_idx == central_shot_idx and rank == 0:
                    central_shot_grad = epsilon_window.grad.detach().cpu().numpy()
                    # 创建完整模型大小的梯度图来显示窗口位置
                    full_grad_display = np.zeros((MODEL_DEPTH, MODEL_WIDTH + 2 * PADDING_WIDTH), dtype=np.float32)
                    full_grad_display[:WINDOW_DEPTH, offset_x:offset_x + WINDOW_WIDTH] = central_shot_grad
                    
                    plt.figure(figsize=(12, 4))
                    plt.imshow(full_grad_display[:, PADDING_WIDTH:-PADDING_WIDTH], aspect='auto', cmap='RdBu_r')
                    plt.title(f'Central Shot {central_shot_idx} Gradient (Epoch {epoch})')
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(f'{result_dir}epoch_{epoch}_central_shot_gradient.png')
                    plt.close()
                    
                    # 同时保存窗口区域的放大图
                    plt.figure(figsize=(8, 4))
                    plt.imshow(central_shot_grad, aspect='auto', cmap='RdBu_r')
                    plt.title(f'Central Shot Window Gradient (Epoch {epoch})')
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(f'{result_dir}epoch_{epoch}_central_shot_window_gradient.png')
                    plt.close()
                
                epsilon_window.grad.zero_()

            epoch_loss += data_loss.item()
            n_shots_processed += 1

            
        # 保存所有单炮梯度（rank 0 only）
        if rank == 0 and individual_grads:
            # 创建单炮梯度目录
            shot_grad_dir = os.path.join(result_dir, f"epoch_{epoch}_shot_gradients")
            os.makedirs(shot_grad_dir, exist_ok=True)
            
            for i, (shot_idx, shot_grad) in enumerate(zip(individual_shot_indices, individual_grads)):
                plt.figure(figsize=(12, 4))
                plt.imshow(shot_grad[:, PADDING_WIDTH:-PADDING_WIDTH], aspect='auto', cmap='RdBu_r')
                plt.title(f'Shot {shot_idx} Gradient (Epoch {epoch})')
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(f'{shot_grad_dir}/shot_{shot_idx}_gradient.png')
                plt.close()
            
            print(f"Epoch {epoch}: 已保存 {len(individual_grads)} 个单炮梯度图像到 {shot_grad_dir}")

        if MPI_AVAILABLE and size > 1:
            grad_tensor_cpu = accumulated_grad_eps.detach().cpu()
            grad_np = np.ascontiguousarray(grad_tensor_cpu.numpy())
            grad_global = np.empty_like(grad_np)
            comm.Allreduce(grad_np, grad_global, op=MPI.SUM)
            accumulated_grad_eps = torch.tensor(grad_global, dtype=torch.float32, device=device)

            if hasattr(comm, "allreduce"):
                epoch_loss = comm.allreduce(epoch_loss, op=MPI.SUM)
                n_shots_processed = comm.allreduce(n_shots_processed, op=MPI.SUM)

        accumulated_grad_eps[:, :PADDING_WIDTH] = 0.0
        accumulated_grad_eps[:, -PADDING_WIDTH:] = 0.0

        grad_eps_no_pad = accumulated_grad_eps[:, PADDING_WIDTH:-PADDING_WIDTH]
        grad_eps_no_pad = grad_hook(grad_eps_no_pad)

        if rank == 0 and net_1 is not None:
            net_1.train()
            optimizer.zero_grad()
            
            _, _, net1_output = net_1(net1_input)
            epsilon_big_temp = torch.sigmoid(net1_output + offset) * (MAX_EPS - MIN_EPS) + MIN_EPS

            epsilon_big_temp.backward(gradient=grad_eps_no_pad)
            optimizer.step()
            optimizer.zero_grad()

            avg_loss = epoch_loss / n_shots_processed if n_shots_processed > 0 else 0.0
            loss_history.append(avg_loss)
            print(f"Epoch {epoch}: 平均损失 = {avg_loss:.6f}")

            if epoch % 10 == 0 or epoch in [1, 2, 5, 10]:
                plt.figure(figsize=(12, 4))
                plt.imshow(epsilon_big.detach().cpu().numpy(), aspect="auto", cmap="jet", vmin=MIN_EPS, vmax=MAX_EPS)
                plt.title(f"Epsilon Model (epoch {epoch})")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(f"{result_dir}epoch_{epoch}_epsilon.png")
                plt.close()

                plt.figure(figsize=(12, 4))
                plt.imshow(grad_eps_no_pad.detach().cpu().numpy(), aspect="auto", cmap="RdBu_r")
                plt.title(f"Accumulated Gradient (epoch {epoch})")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(f"{result_dir}epoch_{epoch}_gradient.png")
                plt.close()

        comm.Barrier()

        if rank == 0 and epoch % 100 == 0 and epoch > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(loss_history, "b-", linewidth=2)
            plt.title("Training Loss History")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{result_dir}loss_history_epoch_{epoch}.png")
            plt.close()

    if rank == 0 and net_1 is not None:
        print("训练完成！绘制最终loss曲线...")
        plt.figure(figsize=(8, 4))
        plt.plot(loss_history, "b-", linewidth=2)
        plt.title("Final Training Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{result_dir}loss_history_final.png")
        plt.close()

        with torch.no_grad():
            _, _, final_output = net_1(net1_input)
            final_epsilon = torch.sigmoid(final_output + offset) * (MAX_EPS - MIN_EPS) + MIN_EPS

        np.save(f"{result_dir}final_epsilon_model.npy", final_epsilon.cpu().numpy())
        plt.figure(figsize=(12, 4))
        plt.imshow(final_epsilon.cpu().numpy(), aspect="auto", cmap="jet", vmin=MIN_EPS, vmax=MAX_EPS)
        plt.title("Final Epsilon Model")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{result_dir}final_epsilon_model.png")
        plt.close()

        print(f"训练完成！结果已保存到: {result_dir}")


if __name__ == "__main__":
    main()

