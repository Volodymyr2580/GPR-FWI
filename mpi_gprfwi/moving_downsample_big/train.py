#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Unet重参数化的MPI版本全波形反演程序
基于gpr_train.py的主体框架，将数据生成、正演模拟和梯度计算并行化
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.utils.data as data_utils
import os
import argparse
import shutil
import sys
import random
import time
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from utils.forward import forward_model#, SOURCE_WAVELET_PATH
from utils.forward import _get_cpml_base, _CPMLParams
from utils.Time_loop import time_loop
from utils.gradient import compute_gradient, compute_tikhonov_gradient, compute_laplacian_2d
from utils.unet import UNet
from utils.Wavelet import ricker

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

# ========== 自动微分算子定义 ==========
class ForwardModelFunction(Function):
    """
    自定义自动微分函数，包装FDTD正演模拟
    """

    @staticmethod
    def forward(
        ctx,
        epsilon,
        sigma,
        source_list,
        receiver_list,
        dt,
        dx,
        dz,
        npml,
        freq,
        steps,
        wavelet=None,
        use_mpi=True,
    ):
        """
        前向传播：执行FDTD正演模拟
        """
        # 将torch张量转换为numpy数组用于FDTD计算
        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()

        # 执行正演模拟
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
            wavelet=wavelet,
            force_serial=not use_mpi,
        )

        # 将结果转换回torch张量，确保是float32类型
        data_tensor = torch.from_numpy(data).to(epsilon.device).float()

        # 保存用于反向传播的信息
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
        ctx.wavelet = wavelet
        ctx.use_mpi = use_mpi

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

        # 将torch张量转换为numpy数组
        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()
        grad_output_np = grad_output.detach().cpu().numpy()

        # 计算本地梯度
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
            sigma_required_gradient=False,
            wavefield_data=wave_field,
            force_serial=not ctx.use_mpi,
        )

        # 转换回torch张量
        grad_eps_tensor = torch.from_numpy(grad_eps).to(epsilon.device)
        grad_sig_tensor = (
            torch.from_numpy(grad_sig).to(sigma.device) if grad_sig is not None else None
        )

        # 在MPI环境下对梯度做Allreduce求和，汇总所有进程的贡献
        try:
            if ctx.use_mpi and MPI_AVAILABLE and 'comm' in globals() and comm.Get_size() > 1:
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
        return (
            grad_eps_tensor,
            grad_sig_tensor,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

class TikhonovRegularizationFunction(Function):
    """
    自定义Tikhonov正则化函数，利用gradient.py中的compute_tikhonov_gradient
    """
    @staticmethod
    def forward(ctx, epsilon, sigma, alpha_tikhonov_eps, alpha_tikhonov_sig, dx, dz):
        """
        前向传播：计算Tikhonov正则化损失
        """
        # 将torch张量转换为numpy数组以使用gradient.py中的函数
        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()
        
        # 使用gradient.py中的函数计算拉普拉斯算子
        laplacian_eps_np = compute_laplacian_2d(epsilon_np, dx, dz)
        laplacian_sig_np = compute_laplacian_2d(sigma_np, dx, dz)
        
        # 转换回torch张量
        laplacian_eps = torch.from_numpy(laplacian_eps_np).to(epsilon.device)
        laplacian_sig = torch.from_numpy(laplacian_sig_np).to(sigma.device)
        
        # 计算正则化损失
        reg_loss_eps = 1e-10 * alpha_tikhonov_eps * torch.sum(laplacian_eps**2)
        reg_loss_sig = 1e-10 * alpha_tikhonov_sig * torch.sum(laplacian_sig**2)
        
        # 总的正则化损失
        reg_loss = reg_loss_eps + reg_loss_sig
        
        # 保存用于反向传播的信息
        ctx.save_for_backward(epsilon, sigma, laplacian_eps, laplacian_sig)
        ctx.alpha_tikhonov_eps = alpha_tikhonov_eps
        ctx.alpha_tikhonov_sig = alpha_tikhonov_sig
        ctx.dx = dx
        ctx.dz = dz
        
        return reg_loss
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：使用gradient.py中的compute_tikhonov_gradient
        """
        epsilon, sigma, laplacian_eps, laplacian_sig = ctx.saved_tensors
        alpha_tikhonov_eps = ctx.alpha_tikhonov_eps
        alpha_tikhonov_sig = ctx.alpha_tikhonov_sig
        dx = ctx.dx
        dz = ctx.dz
        
        # 将torch张量转换为numpy数组
        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()
        
        # 使用gradient.py中的函数计算Tikhonov梯度
        tikhonov_grad_eps = compute_tikhonov_gradient(epsilon_np, dx, dz, alpha_tikhonov_eps)
        tikhonov_grad_sig = compute_tikhonov_gradient(sigma_np, dx, dz, alpha_tikhonov_sig)
        
        # 转换回torch张量
        grad_eps = torch.from_numpy(tikhonov_grad_eps).to(epsilon.device) * grad_output
        grad_sig = torch.from_numpy(tikhonov_grad_sig).to(sigma.device) * grad_output
        
        return grad_eps, grad_sig, None, None, None, None

def tikhonov_reg(epsilon, sigma, alpha_tikhonov_eps, alpha_tikhonov_sig, dx=0.02, dz=0.02):
    """
    计算Tikhonov正则化损失项，使用自定义自动微分函数
    """
    return TikhonovRegularizationFunction.apply(epsilon, sigma, alpha_tikhonov_eps, alpha_tikhonov_sig, dx, dz)

# ========== 辅助函数 ==========
def grad_hook(grad):
    """梯度处理钩子函数"""
    # 检查梯度是否包含nan或inf
    if torch.isnan(grad).any() or torch.isinf(grad).any():
        print("警告：梯度包含nan或inf值！")
        grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 浅层掩码：前10行梯度设为0
    grad[:10, :] = grad[:10, :] * 0
    
    # 梯度归一化
    max_abs_value = torch.max(torch.abs(grad)).item()
    if max_abs_value != 0:
        grad /= max_abs_value
    
    # 梯度裁剪：限制梯度范围
    grad = torch.clamp(grad, -1.0, 1.0)
    
    return grad

def get_parameter_number(net):
    """获取网络参数数量"""
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def plot_loss_curves(loss_history, eps_loss_history, save_path, current_epoch):
    """绘制损失曲线"""
    plt.figure(figsize=(15, 5))
    
    # 完整损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.title('Full Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 最近100个epoch的损失趋势
    plt.subplot(1, 3, 2)
    recent_epochs = min(100, len(loss_history))
    if recent_epochs > 0:
        plt.plot(range(len(loss_history)-recent_epochs, len(loss_history)), 
                loss_history[-recent_epochs:], 'r-', linewidth=2)
        plt.title(f'Recent {recent_epochs} Epochs Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    # Epsilon模型损失
    plt.subplot(1, 3, 3)
    plt.plot(eps_loss_history, 'g-', linewidth=2)
    plt.title('Epsilon Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}loss_curves_epoch_{current_epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

def same_seeds(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 统一的目录创建工具，确保可写且干净
def ensure_clean_dir(target_dir):
    """删除已存在的同名文件/目录并新建目录"""
    if os.path.exists(target_dir):
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
        else:
            os.remove(target_dir)
    os.makedirs(target_dir, exist_ok=True)

# ========== 全局常量 ==========
MODEL_DEPTH = 410
MODEL_WIDTH = 5000
WINDOW_DEPTH = 410
WINDOW_WIDTH = 820
PADDING_WIDTH = WINDOW_WIDTH // 2
fix_layer = 10

DT = 1.4e-10
DX = 0.06
DZ = 0.06
NPML = 10
FREQ = 1.5e8

MIN_EPS = 1.0
MAX_EPS = 9.0

SHOT_STRIDE = 2

C0 = 2.99792458e8
# BACKGROUND_EPS = 6.0
# BACKGROUND_SPEED = C0 / np.sqrt(BACKGROUND_EPS)
# MAX_TRAVEL_DISTANCE = 2.0 * MODEL_DEPTH * DZ
# TIME_WINDOW = 1.3 * MAX_TRAVEL_DISTANCE / BACKGROUND_SPEED
STEPS = 3500


# ========== 模型与子波辅助函数 ==========
def generate_models():
    """生成真实模型与初始模型"""
    true_model = np.full((MODEL_DEPTH, MODEL_WIDTH), 6.0, dtype=np.float32)
    true_model[-10:, :] = 1.5

    n_cols = 5
    square_size = 20
    block_width = 40  # 横向长度改为40
    anomaly_left_positions = np.linspace(5, MODEL_WIDTH - block_width - 5, num=n_cols, dtype=int)
    anomaly_top_layers = [40, 60]

    for top in anomaly_top_layers:
        bottom = top + square_size
        for left in anomaly_left_positions:
            right = left + block_width
            true_model[top:bottom, left:right] = 3.0

    initial_model = gaussian_filter(true_model, sigma=5).astype(np.float32)
    return true_model, initial_model


def eps_to_sig(epsilon_r, freq=FREQ):
    """介电常数转换为电导率"""
    ep0 = 8.841941282883074e-12
    epsilon = epsilon_r * ep0
    index = 0.440 * np.log(epsilon_r) / np.log(1.93) - 2.943
    sig = 2 * np.pi * freq * epsilon * 10 ** (index)
    return sig.astype(np.float32)


def generate_ricker_wavelet(frequency, dt, steps):
    """使用Wavelet.ricker生成指定长度的Ricker子波"""
    t = np.arange(steps, dtype=np.float64) * dt
    wavelet = ricker(t, frequency)
    return wavelet.astype(np.float64)


def extract_window(model_padded, source_x_global):
    """提取以炮点为中心的窗口模型，并返回窗口起始列索引"""
    source_x_padded = source_x_global + PADDING_WIDTH
    offset_x = source_x_padded - WINDOW_WIDTH // 2
    max_offset = max(0, model_padded.shape[1] - WINDOW_WIDTH)
    offset_x = max(0, min(offset_x, max_offset))
    window = model_padded[:WINDOW_DEPTH, offset_x:offset_x + WINDOW_WIDTH]
    return window, offset_x


def select_uniform_shots(epoch, total_shots, stride):
    """按照均匀间隔选择炮点索引"""
    stride = max(1, stride)
    offset = epoch % stride
    selected = list(range(offset, total_shots, stride))
    if not selected:
        selected = [epoch % total_shots]
    return selected


def pad_to_multiple(array, multiple=32):
    """对输入数组做上下左右补边，使其尺寸为multiple的倍数"""
    height, width = array.shape
    target_height = ((height + multiple - 1) // multiple) * multiple
    target_width = ((width + multiple - 1) // multiple) * multiple
    pad_height = target_height - height
    pad_width = target_width - width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    if pad_height == 0 and pad_width == 0:
        return array
    return np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")

# ========== 数据加载器 ==========
class MyDataset(data_utils.Dataset):
    def __init__(self, seismic_data, source_locations, receiver_locations):
        self.seismic_data = seismic_data
        self.source_locations = source_locations
        self.receiver_locations = receiver_locations
    
    def __getitem__(self, item):
        return self.seismic_data[item], self.source_locations[item], self.receiver_locations[item]
    
    def __len__(self):
        return len(self.seismic_data)


# ========== 主程序 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['generate', 'inversion'], default='inversion')
    args = parser.parse_args()
    same_seeds(2025)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    if rank == 0:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_output_dir = os.path.join(script_dir, 'result_gpr_mpi')
        data_dir = os.path.join(script_dir, 'data')
        try:
            ensure_clean_dir(base_output_dir)
        except PermissionError:
            home_dir = os.path.expanduser('~')
            base_output_dir = os.path.join(home_dir, 'result_gpr_mpi')
            ensure_clean_dir(base_output_dir)
        print(f"结果将保存到: {base_output_dir}")
        os.makedirs(data_dir, exist_ok=True)
    else:
        base_output_dir = None
        data_dir = None

    base_output_dir = comm.bcast(base_output_dir if rank == 0 else None, root=0)
    data_dir = comm.bcast(data_dir if rank == 0 else None, root=0)
    
    comm.Barrier()  # 等待所有进程
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    if rank == 0:
        print(f"使用设备: {device}")
        print(f"MPI进程数: {size}")
    base = _get_cpml_base(WINDOW_DEPTH, WINDOW_WIDTH, DX, DZ, DT, NPML)
    epsilon_tmp = np.ones((WINDOW_DEPTH + 2*NPML, WINDOW_WIDTH + 2*NPML), dtype=np.float32)
    sigma_tmp = np.ones_like(epsilon_tmp, dtype=np.float32) * 0
    mu_tmp = np.ones_like(epsilon_tmp, dtype=np.float32)
    cpml_params = _CPMLParams(base, np.ones_like(epsilon_tmp, dtype=np.float32), np.ones_like(epsilon_tmp, dtype=np.float32), np.ones_like(epsilon_tmp, dtype=np.float32))
    wavelet_warm = generate_ricker_wavelet(FREQ, DT, 1)
    src_ext = (NPML, NPML + WINDOW_WIDTH // 2)
    ref_ext = src_ext
    _gen = time_loop(WINDOW_DEPTH, WINDOW_WIDTH, DX, DZ, DT, sigma_tmp, epsilon_tmp, mu_tmp, cpml_params, wavelet_warm, 1, src_ext, ref_ext)
    for _ in _gen:
        break
    
    learning_rate = 1e-2  # 学习率
    alpha_tikhonov_eps = 0.015  # Tikhonov正则化参数
    alpha_tikhonov_sig = 0
    offset = 0

    if rank == 0:
        result_dir = os.path.join(base_output_dir, f'lr{learning_rate:.0e}_alpha{alpha_tikhonov_eps:.3g}_offset{offset}')
        ensure_clean_dir(result_dir)
        path = result_dir if result_dir.endswith(os.sep) else result_dir + os.sep
        print(f"结果保存路径: {path}")
    else:
        result_dir = None
        path = None
    
    comm.Barrier()
    
    path = comm.bcast(path if rank == 0 else None, root=0)
    result_dir = comm.bcast(result_dir if rank == 0 else None, root=0)

    true_model = np.load(os.path.join(data_dir, 'model.npy'))
    input_model = np.load(os.path.join(data_dir, 'input_model.npy'))
    USED_WIDTH = 1000
    true_model = true_model[:, :USED_WIDTH]
    input_model = input_model[:, :USED_WIDTH]
    if rank == 0:
        print(f"裁切后的true_model.shape: {true_model.shape}")
        print(f"裁切后的input_model.shape: {input_model.shape}")

    wavelet = np.load(os.path.join(data_dir, 'wavelet_hp.npy'))

    sigma_true_np = np.ones_like(true_model) * 0

    epsilon_true = torch.tensor(true_model, dtype=torch.float32, device=device)
    sigma_true = torch.tensor(sigma_true_np, dtype=torch.float32, device=device)

    epsilon_true_pad_np = np.pad(true_model, ((0, 0), (PADDING_WIDTH, PADDING_WIDTH)), mode='edge')
    sigma_true_pad_np = np.pad(sigma_true_np, ((0, 0), (PADDING_WIDTH, PADDING_WIDTH)), mode='edge')
    epsilon_init_pad_np = np.pad(input_model, ((0, 0), (PADDING_WIDTH, PADDING_WIDTH)), mode='edge')

    net_input_np = pad_to_multiple(epsilon_init_pad_np, multiple=32)

    net1_input = torch.tensor(net_input_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    #可视化真实模型
    if rank == 0:
        plt.figure(figsize=(6, 4))
        plt.imshow(true_model, aspect='auto', cmap='jet', vmin=MIN_EPS, vmax=MAX_EPS)
        plt.title('True Epsilon Model')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'{path}epsilon_true.png')
        plt.close()
        plt.figure(figsize=(6, 4))
        plt.imshow(sigma_true_np, aspect='auto', cmap='jet')
        plt.title('True Sigma Model')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'{path}sigma_true.png')
        plt.close()

    if rank == 0:
        wavelet_np = wavelet
        plt.figure(figsize=(10, 3))
        plt.plot(wavelet_np, color='orange')
        plt.title('Source Wavelet cut under 400MHz')
        plt.xlabel('Time Step')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{path}wavelet.png')
        plt.close()
    else:
        wavelet_np = None

    wavelet_np = comm.bcast(wavelet_np, root=0)

    net1_input = comm.bcast(net1_input if rank == 0 else None, root=0)

    if rank == 0:
        print("初始化UNet网络...")
        net_1 = UNet(
            in_channels=1,
            out_channels=1,
            target_height=MODEL_DEPTH,
            target_width=true_model.shape[1],
        ).to(device)
        print(f"网络参数数量: {get_parameter_number(net_1)}")
    else:
        net_1 = None

    sigma_pad_np = comm.bcast(sigma_true_pad_np if rank == 0 else None, root=0)
    sigma_pad_tensor = torch.tensor(sigma_pad_np, dtype=torch.float32, device=device)

    if args.mode == 'generate':
        if rank == 0:
            print("生成观测数据...")

        total_shots = true_model.shape[1]
        all_shots = list(range(total_shots))
        shots_per_proc = total_shots // size
        remainder = total_shots % size

        if rank < remainder:
            start_shot = rank * (shots_per_proc + 1)
            end_shot = start_shot + shots_per_proc + 1
        else:
            start_shot = rank * shots_per_proc + remainder
            end_shot = start_shot + shots_per_proc

        local_shots = all_shots[start_shot:end_shot]

        local_d_obs_list = []
        for shot_idx in local_shots:
            epsilon_window_np, _ = extract_window(epsilon_true_pad_np, shot_idx)
            sigma_window_np, _ = extract_window(sigma_true_pad_np, shot_idx)
            source_local = [(0, WINDOW_WIDTH // 2)]
            receiver_local = source_local.copy()
            d_obs_shot = forward_model(
                epsilon_window_np,
                sigma_window_np,
                source_local,
                receiver_local,
                DT,
                DX,
                DZ,
                NPML,
                FREQ,
                STEPS,
                save_wavefield=False,
                wavelet=wavelet_np,
                force_serial=True,
            )
            local_d_obs_list.append(d_obs_shot[0])

        if local_d_obs_list:
            local_d_obs_array = np.stack(local_d_obs_list, axis=0)
        else:
            local_d_obs_array = np.zeros((0, STEPS), dtype=np.float32)

        d_obs_pieces = comm.gather(local_d_obs_array, root=0)
        if rank == 0:
            if d_obs_pieces:
                d_obs_np = np.concatenate(d_obs_pieces, axis=0)
            else:
                d_obs_np = np.zeros((total_shots, STEPS), dtype=np.float32)
            np.save(os.path.join(data_dir, 'd_obs.npy'), d_obs_np)
            print(f"观测数据已保存: {os.path.join(data_dir, 'd_obs.npy')}")
            plt.figure(figsize=(12, 4))
            plt.imshow(d_obs_np.T, aspect='auto', cmap='seismic', vmin=-1, vmax=1)
            plt.title('Observed Data (d_obs)')
            plt.xlabel('Shot')
            plt.ylabel('Time')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"{path}d_obs.png")
            plt.close()
        comm.Barrier()
        return
    else:
        if rank == 0:
            d_obs_np = np.load(os.path.join(data_dir, 'd_obs.npy'))
            d_obs_np = d_obs_np[:true_model.shape[1], :]
        else:
            d_obs_np = None
        d_obs_np = comm.bcast(d_obs_np, root=0)
        if rank == 0:
            plt.figure(figsize=(12, 4))
            plt.imshow(d_obs_np.T, aspect='auto', cmap='seismic', vmin=-1, vmax=1)
            plt.title('Observed Data (d_obs, loaded)')
            plt.xlabel('Shot')
            plt.ylabel('Time')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"{path}d_obs.png")
            plt.close()
        d_obs = torch.tensor(d_obs_np, dtype=torch.float32, device=device)

    if rank == 0:
        optimizer = torch.optim.Adam(net_1.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    else:
        optimizer = None
        scheduler = None

    loss_fn = torch.nn.MSELoss()
    num_epochs = 2001  # 调试阶段仅运行1个epoch
    loss_history = []
    eps_loss_history = []

   ########################################################################################### 训练开始 ########################################################################################### 
    if rank == 0:
        print("开始训练...")
    total_shots = true_model.shape[1]
    for epoch in range(num_epochs):
        if rank == 0:
            if net_1 is not None:
                net_1.eval()
                with torch.no_grad():
                    _, _, net_output_eval = net_1(net1_input)
                    pred_eval_eps = torch.sigmoid(net_output_eval) * (MAX_EPS - MIN_EPS) + MIN_EPS
                    if fix_layer > 0:
                        epsilon_big_eval = pred_eval_eps.clone()
                        epsilon_big_eval[:fix_layer, :] = torch.tensor(
                            true_model[:fix_layer, :], dtype=torch.float32, device=device
                        )
                    else:
                        epsilon_big_eval = pred_eval_eps
                    if epoch % 100 == 0:
                        print('generate unet_output successfully.....')
                        plt.figure(figsize=(12, 4))
                        plt.imshow(net1_input.squeeze().detach().cpu().numpy(), aspect='auto', cmap='jet', vmin=MIN_EPS, vmax=MAX_EPS)
                        plt.title(f'UNet Input (epoch {epoch})')
                        plt.colorbar()
                        plt.tight_layout()
                        plt.savefig(f"{path}epoch_{epoch}_unet_input.png")
                        plt.close()
                        vals = net_output_eval.detach().cpu().numpy().ravel()
                        plt.figure(figsize=(8, 4))
                        plt.hist(vals, bins=100, color='steelblue', alpha=0.9)
                        plt.title(f'UNet Output Pre-Sigmoid Histogram (epoch {epoch})')
                        plt.xlabel('Value')
                        plt.ylabel('Count')
                        plt.tight_layout()
                        plt.savefig(f"{path}epoch_{epoch}_unet_output_pre_sigmoid_hist.png")
                        plt.close()
            else:
                epsilon_big_eval = torch.tensor(true_model, dtype=torch.float32, device=device)
            epsilon_big_pad_eval = F.pad(
                epsilon_big_eval,
                (PADDING_WIDTH, PADDING_WIDTH),
                mode='replicate',
            )
            epsilon_big_pad_np = (
                epsilon_big_pad_eval.squeeze().detach().cpu().numpy()
            )
        else:
            epsilon_big_pad_np = None

        epsilon_big_pad_np = comm.bcast(epsilon_big_pad_np, root=0)
        epsilon_big_pad_tensor = torch.tensor(epsilon_big_pad_np, dtype=torch.float32, device=device)

        if rank ==0:
            print('allocating shots to each process......')
        selected_shots = select_uniform_shots(epoch, total_shots, SHOT_STRIDE)
        n_selected = len(selected_shots)
        shots_per_proc_epoch = n_selected // size
        remainder_epoch = n_selected % size

        if rank < remainder_epoch:
            local_start = rank * (shots_per_proc_epoch + 1)
            local_end = local_start + shots_per_proc_epoch + 1
        else:
            local_start = rank * shots_per_proc_epoch + remainder_epoch
            local_end = local_start + shots_per_proc_epoch

        local_selected_shots = selected_shots[local_start:local_end]

        accumulated_grad_pad = torch.zeros_like(epsilon_big_pad_tensor, dtype=torch.float32, device=device)
        epoch_loss = 0.0
        shots_processed = 0
        local_debug_records = []

        for shot_idx in local_selected_shots:
            if rank == 0:
                print(f'processing shot {shot_idx}, extracting calculation area......')
            epsilon_window, offset_x = extract_window(epsilon_big_pad_tensor, shot_idx)
            sigma_window, _ = extract_window(sigma_pad_tensor, shot_idx)
            
            epsilon_window = epsilon_window.clone().detach().requires_grad_(True)
            sigma_window = sigma_window.clone().detach()

            source_local = [(0, WINDOW_WIDTH // 2)]
            receiver_local = source_local.copy()
            d_obs_shot = d_obs[shot_idx:shot_idx+1, :]

            # print(f"epsilon_window shape: {epsilon_window.shape}")
            # print(f"sigma_window shape: {sigma_window.shape}")
            #生成模拟正演数据
            if rank == 0:
                print(f'processing shot {shot_idx},generating synthetic data')
            d_syn_shot = ForwardModelFunction.apply(
                epsilon_window,
                sigma_window,
                source_local,
                receiver_local,
                DT,
                DX,
                DZ,
                NPML,
                FREQ,
                STEPS,
                wavelet_np,
                False,
            )
            if rank == 0:
                print(f'processing shot {shot_idx},successfully generated synthetic data ')
            d_syn_shot = d_syn_shot.float()
            d_syn_np = d_syn_shot.detach().cpu().numpy().squeeze()
            d_obs_np = d_obs_shot.detach().cpu().numpy().squeeze()
            residual_np = (d_syn_np - d_obs_np).astype(np.float32)
            data_loss = loss_fn(d_syn_shot, d_obs_shot)
            #reg_loss = tikhonov_reg(epsilon_window, sigma_window, alpha_tikhonov_eps, alpha_tikhonov_sig, DX, DZ)
            total_loss = data_loss #+ reg_loss
            #梯度反传？
            if rank == 0:
                print(f'processing shot {shot_idx},computing gradient ')
            total_loss.backward()

            if epsilon_window.grad is not None:
                window_grad = epsilon_window.grad.detach()
                start_col = offset_x
                end_col = start_col + WINDOW_WIDTH
                accumulated_grad_pad[:, start_col:end_col] += window_grad
                local_debug_records.append(
                    {
                        "shot_idx": shot_idx,
                        "d_syn": d_syn_np,
                        "d_obs": d_obs_np,
                        "residual": residual_np,
                        "grad_window": window_grad.cpu().numpy(),
                        "offset_x": offset_x,
                    }
                )
            epoch_loss += total_loss.item()
            shots_processed += 1

        if MPI_AVAILABLE and size > 1:
            grad_tensor_cpu = accumulated_grad_pad.detach().cpu()
            grad_np = np.ascontiguousarray(grad_tensor_cpu.numpy())
            grad_global = np.empty_like(grad_np)
            comm.Allreduce(grad_np, grad_global, op=MPI.SUM)
            accumulated_grad_pad = torch.tensor(grad_global, dtype=torch.float32, device=device)

            epoch_loss = comm.allreduce(epoch_loss, op=MPI.SUM)
            shots_processed = comm.allreduce(shots_processed, op=MPI.SUM)

        grad_eps_no_pad = accumulated_grad_pad[:, PADDING_WIDTH:-PADDING_WIDTH]
        grad_eps_no_pad = grad_hook(grad_eps_no_pad)
        grad_eps_no_pad = grad_eps_no_pad.contiguous()

        debug_records_all = comm.gather(local_debug_records, root=0)
        if rank == 0:
            flat_records = [rec for proc_records in debug_records_all for rec in proc_records]

            if flat_records:
                shot_order = {shot: idx for idx, shot in enumerate(selected_shots)}
                syn_map = np.zeros((len(selected_shots), STEPS), dtype=np.float32)
                obs_map = np.zeros_like(syn_map)
                residual_map = np.zeros_like(syn_map)

                for rec in flat_records:
                    shot_idx = rec["shot_idx"]
                    if shot_idx in shot_order:
                        pos = shot_order[shot_idx]
                        syn_map[pos] = rec["d_syn"]
                        obs_map[pos] = rec["d_obs"]
                        residual_map[pos] = rec["residual"]

                #np.save(f"{path}epoch_{epoch:04d}_d_syn.npy", syn_map)
                #np.save(f"{path}epoch_{epoch:04d}_d_obs.npy", obs_map)
                #np.save(f"{path}epoch_{epoch:04d}_residual.npy", residual_map)

                plt.figure(figsize=(10, 4))
                plt.imshow(syn_map.T, aspect='auto', cmap='seismic', vmin=-1, vmax=1)
                plt.title(f'Synthetic B-scan (epoch {epoch})')
                plt.xlabel('Selected Shot Index')
                plt.ylabel('Time')
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(f"{path}epoch_{epoch:04d}_d_syn.png")
                plt.close()

                plt.figure(figsize=(10, 4))
                plt.imshow(obs_map.T, aspect='auto', cmap='seismic', vmin=-1, vmax=1)
                plt.title(f'Observed B-scan (epoch {epoch})')
                plt.xlabel('Selected Shot Index')
                plt.ylabel('Time')
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(f"{path}epoch_{epoch:04d}_d_obs.png")
                plt.close()

                plt.figure(figsize=(10, 4))
                plt.imshow(residual_map.T, aspect='auto', cmap='seismic')
                plt.title(f'Residual B-scan (epoch {epoch})')
                plt.xlabel('Selected Shot Index')
                plt.ylabel('Time')
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(f"{path}epoch_{epoch:04d}_residual.png")
                plt.close()





        if rank == 0 and net_1 is not None:
            net_1.train()
            optimizer.zero_grad()
            _, _, net_output_train = net_1(net1_input)
            pred_train_eps = torch.sigmoid(net_output_train + offset) * (MAX_EPS - MIN_EPS) + MIN_EPS
            if fix_layer > 0:
                fixed_eps_top = torch.tensor(
                    true_model[:fix_layer, :], dtype=torch.float32, device=device
                )
                epsilon_big_train = pred_train_eps.clone()
                epsilon_big_train[:fix_layer, :] = fixed_eps_top
                grad_mask = grad_eps_no_pad.clone()
                grad_mask[:fix_layer, :] = 0.0
            else:
                epsilon_big_train = pred_train_eps
                grad_mask = grad_eps_no_pad
            epsilon_big_train.backward(gradient=grad_mask)
            optimizer.step()
            net_1.eval()
            with torch.no_grad():
                _, _, curr_output = net_1(net1_input)
                pred_curr_eps = torch.sigmoid(curr_output + offset) * (MAX_EPS - MIN_EPS) + MIN_EPS
                if fix_layer > 0:
                    fixed_eps_top = torch.tensor(
                        true_model[:fix_layer, :], dtype=torch.float32, device=device
                    )
                    curr_epsilon = pred_curr_eps.clone()
                    curr_epsilon[:fix_layer, :] = fixed_eps_top
                else:
                    curr_epsilon = pred_curr_eps
            np.save(f"{path}epoch_{epoch:04d}_epsilon.npy", curr_epsilon.detach().cpu().numpy())
            plt.figure(figsize=(12, 4))
            plt.imshow(curr_epsilon.detach().cpu().numpy(), aspect='auto', cmap='jet', vmin=MIN_EPS, vmax=MAX_EPS)
            plt.title(f'Epsilon Model (epoch {epoch})')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"{path}epoch_{epoch:04d}_epsilon.png")
            plt.close()

            loss = epoch_loss
            loss_history.append(loss)
            if scheduler is not None:
                scheduler.step()

            eps_loss = loss_fn(epsilon_big_eval, epsilon_true).item()
            eps_loss_history.append(eps_loss)

            if epoch % 1 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}, Epsilon MSE = {eps_loss:.6f}, Shots = {shots_processed}")
                grad_plot = grad_eps_no_pad.detach().cpu().numpy().copy()
                grad_plot[:fix_layer, :] = 0.0
                plt.figure(figsize=(12, 4))
                plt.imshow(grad_plot, aspect='auto', cmap='seismic', vmin=-1, vmax=1)
                plt.title(f'Gradient (epoch {epoch})')
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(f"{path}epoch_{epoch}_gradient.png")
                plt.close()



            if epoch % 100 == 0 and epoch > 0:
                plot_loss_curves(loss_history, eps_loss_history, path, epoch)

        comm.Barrier()

    if rank == 0 and net_1 is not None:
        print("绘制最终loss曲线...")
        plot_loss_curves(loss_history, eps_loss_history, path, num_epochs - 1)
        net_1.eval()
        with torch.no_grad():
            _, _, final_output = net_1(net1_input)
            pred_final_eps = torch.sigmoid(final_output + offset) * (MAX_EPS - MIN_EPS) + MIN_EPS
            if fix_layer > 0:
                fixed_eps_top = torch.tensor(
                    true_model[:fix_layer, :], dtype=torch.float32, device=device
                )
                final_epsilon = pred_final_eps.clone()
                final_epsilon[:fix_layer, :] = fixed_eps_top
            else:
                final_epsilon = pred_final_eps
        np.save(f"{path}final_epsilon_model.npy", final_epsilon.detach().cpu().numpy())
        plt.figure(figsize=(12, 4))
        plt.imshow(final_epsilon.detach().cpu().numpy(), aspect='auto', cmap='jet', vmin=MIN_EPS, vmax=MAX_EPS)
        plt.title("Final Epsilon Model")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{path}final_epsilon_model.png")
        plt.close()
        print("训练完成！最终结果已保存。")

if __name__ == "__main__":
    main()