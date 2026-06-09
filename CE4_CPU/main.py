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
import shutil
import sys
import random
import time
from scipy.ndimage import gaussian_filter
from utils.forward import forward_model, load_filtered_wavelet
from utils.gradient import compute_gradient, compute_tikhonov_gradient, compute_laplacian_2d
from utils.unet import UNet

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

# ========== 全局变量：子波缓存 ==========
# 用于ForwardModelFunction（torch Function无法直接传递额外参数）
_global_wavelet = None

# ========== 辅助函数：窗口提取 ==========
def extract_window(model_padded, source_x_global, window_depth=410, window_width=800):
    """
    从padded模型中提取窗口，source位于窗口顶层中间
    
    参数:
        model_padded: (depth, width_padded) 已padding的模型，width_padded = 5000 + 2*410 = 5820
        source_x_global: source在原始(未padding)模型中的x坐标 (0~4999)
        window_depth: 窗口深度，默认410
        window_width: 窗口宽度，默认800
    
    返回:
        window: (window_depth, window_width) 提取的窗口
        offset_x: 窗口在padded模型中的起始x坐标
    """
    # 计算source在padded模型中的位置（需要加上左侧padding）
    source_x_padded = source_x_global + 410
    
    # 计算窗口在padded模型中的位置，使source位于窗口顶层中间
    window_center_x = source_x_padded
    offset_x = window_center_x - window_width // 2
    
    # 确保窗口不超出边界
    offset_x = max(0, min(offset_x, model_padded.shape[1] - window_width))
    
    # 提取窗口
    window = model_padded[:window_depth, offset_x:offset_x+window_width]
    
    return window, offset_x

def map_gradient_to_full(grad_window, offset_x, full_shape, window_depth=410, window_width=800):
    """
    将窗口的梯度映射回完整模型
    
    参数:
        grad_window: (window_depth, window_width) 窗口的梯度
        offset_x: 窗口在padded模型中的起始x坐标
        full_shape: 完整padded模型的形状 (depth, width_padded)
        
    返回:
        grad_full: (depth, width_padded) 映射后的梯度，只有窗口位置有值
    """
    grad_full = np.zeros(full_shape, dtype=np.float32)
    grad_full[:window_depth, offset_x:offset_x+window_width] = grad_window
    return grad_full

def sync_padding_with_edge(model, padding_width=410):
    """
    将padding区域与边缘同步
    
    参数:
        model: (depth, width_padded) 包含padding的模型，width_padded = 5000 + 2*padding_width
        padding_width: 左右padding的宽度
        
    返回:
        model_synced: 同步后的模型
    """
    model_synced = model.copy()
    # 左侧padding与左边缘同步
    model_synced[:, :padding_width] = model[:, padding_width:padding_width+1]
    # 右侧padding与右边缘同步
    model_synced[:, -padding_width:] = model[:, -padding_width-1:-padding_width]
    return model_synced

# ========== 自动微分算子定义 ==========
class ForwardModelFunction(Function):
    """
    自定义自动微分函数，包装FDTD正演模拟（支持窗口化）
    """
    @staticmethod
    def forward(ctx, epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps):
        """
        前向传播：执行FDTD正演模拟
        注意：这里的epsilon和sigma已经是提取好的窗口
        """
        # 将torch张量转换为numpy数组用于FDTD计算
        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()
        
        # 执行正演模拟
        # ForwardModelFunction无法直接传wavelet参数，但forward_model会使用缓存加载
        data, wave_field = forward_model(epsilon_np, sigma_np, source_list, receiver_list, 
                           dt, dx, dz, npml, freq, steps, save_wavefield=True, wavelet=None)
        
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

        # 转换回torch张量
        grad_eps_tensor = torch.from_numpy(grad_eps).to(epsilon.device)

        # 在MPI环境下对梯度做Allreduce求和，汇总所有进程的贡献
        try:
            if MPI_AVAILABLE and 'comm' in globals() and comm.Get_size() > 1:
                # eps梯度
                grad_eps_np_local = grad_eps_tensor.detach().cpu().numpy()
                grad_eps_np_global = np.empty_like(grad_eps_np_local)
                comm.Allreduce(grad_eps_np_local, grad_eps_np_global, op=MPI.SUM)
                grad_eps_np_global = grad_eps_np_global / float(comm.Get_size())
                grad_eps_tensor = torch.from_numpy(grad_eps_np_global).to(epsilon.device)

        except Exception:
            # 若无MPI或出错，则退化为本地梯度
            pass

        return grad_eps_tensor, None, None, None, None, None, None, None, None, None

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
    
    # 浅层掩码：前20行梯度设为0
    grad[:20, :] = grad[:20, :] * 0
    
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

def eps_to_sig(epsilon_r, freq=4e8):
    """将介电常数转换为电导率"""
    ep0 = 8.841941282883074e-12
    epsilon=epsilon_r * ep0
    index = 0.440 * np.log(epsilon_r) / np.log(1.93) - 2.943
    sig = 2*np.pi*freq*epsilon*10**(index)
    return sig

def select_minibatch_shots(n_total_shots=5000, n_selected=100):
    """
    选择mini-batch的炮点
    
    算法：
    1. 随机选择 z ∈ [0, 49]
    2. 选择所有 i 满足 i ≡ z (mod 50) 的炮点索引
    
    参数:
        n_total_shots: 总炮点数，默认5000
        n_selected: 选择的炮点数，默认100
    
    返回:
        selected_indices: 选中的炮点索引列表
    """
    z = random.randint(0, 49)
    selected_indices = [i for i in range(n_total_shots) if i % 50 == z]
    return selected_indices

# ========== 主程序 ==========
def main():
    """
    主函数支持两种运行模式：
    - 模式1（generate_mode=True）：仅生成合成观测数据并保存
    - 模式2（generate_mode=False）：读取已有观测数据，进行全波形反演
    """
    # ========== 运行模式设置 ==========
    import argparse
    parser = argparse.ArgumentParser(description='全波形反演程序')
    parser.add_argument('--mode', type=str, default='inversion', 
                       choices=['generate', 'inversion'],
                       help='运行模式：generate=仅生成观测数据, inversion=执行反演')
    parser.add_argument('--d_obs_file', type=str, default='data/d_obs_syn.npy',
                       help='观测数据文件路径（inversion模式使用）')
    args = parser.parse_args()
    
    generate_mode = (args.mode == 'generate')
    d_obs_file = args.d_obs_file
    
    if rank == 0:
        if generate_mode:
            print("="*70)
            print("模式1：仅生成合成观测数据")
            print("="*70)
        else:
            print("="*70)
            print("模式2：执行全波形反演")
            print(f"读取观测数据: {d_obs_file}")
            print("="*70)
    
    # 设置随机种子
    same_seeds(2025)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # 创建结果文件夹（仅在rank 0执行）
    if rank == 0:
        # 优先在脚本所在目录下创建，避免当前工作目录只读导致失败
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_output_dir = os.path.join(script_dir, 'result_gpr_mpi')
        try:
            ensure_clean_dir(base_output_dir)
        except PermissionError:
            # 回退到用户home目录
            home_dir = os.path.expanduser('~')
            base_output_dir = os.path.join(home_dir, 'result_gpr_mpi')
            ensure_clean_dir(base_output_dir)
        print(f"结果将保存到: {base_output_dir}")
    else:
        base_output_dir = None

    # 广播基础输出目录给所有进程
    base_output_dir = comm.bcast(base_output_dir if rank == 0 else None, root=0)
    
    comm.Barrier()  # 等待所有进程
    
    # 设置设备 - 将MPI进程分配到不同的GPU上
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        target_gpu = 2  # 手动指定使用GPU 2
        
        if rank == 0:
            # 验证GPU 2是否存在
            if target_gpu >= n_gpus:
                print(f"警告：GPU {target_gpu}不存在，共有{n_gpus}个GPU，改用GPU 0")
                target_gpu = 0
            
            # 打印GPU信息
            total = torch.cuda.get_device_properties(target_gpu).total_memory
            print(f"使用GPU {target_gpu}，总内存: {total/1024**3:.2f}GB")
            print("提示：请使用nvidia-smi确认GPU使用情况")
        
        # 只有rank 0使用GPU，其他进程使用CPU以避免内存冲突
        if rank == 0:
            device = torch.device(f'cuda:{target_gpu}')
            torch.cuda.set_device(target_gpu)
            print(f"Rank 0将使用GPU {target_gpu}")
        else:
            # 其他进程使用CPU
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        if rank == 0:
            print("CUDA不可用，使用CPU")
    
    if rank == 0:
        print(f"使用设备: {device}")
        print(f"MPI进程数: {size}")
        print(f"注意：只有rank 0使用GPU，其他进程使用CPU以节省GPU内存")
    
    # ========== 训练设置 ==========
    learning_rate = 1e-3  # 学习率
    alpha_tikhonov_eps = 0.015  # Tikhonov正则化参数
    offset = 0
    freq = 4e8  # 频率
    
    # 窗口参数
    window_depth = 410
    window_width = 800
    padding_width = 410  # 左右padding宽度
    
    # 构造结果文件夹名
    if rank == 0:
        result_dir = os.path.join(base_output_dir, f'lr{learning_rate:.0e}_alpha{alpha_tikhonov_eps:.3g}_offset{offset}')
        ensure_clean_dir(result_dir)
        path = result_dir if result_dir.endswith(os.sep) else result_dir + os.sep
        print(f"结果保存路径: {path}")
    
    comm.Barrier()
    
    # ========== 参数设置 ==========
    # 读取数据文件
    big_model = np.load('data/model.npy')  # (410, 5000)
    big_model_pad = np.pad(big_model, ((0, 0), (padding_width, padding_width)), mode='edge')  # (410, 5820)
    if big_model_pad.shape != (410, 5820):
        print(f"big_model_pad.shape: {big_model_pad.shape}")
        assert big_model_pad.shape == (410, 5820)

    d_obs = np.load('data/filter_data.npy')  # (5000, 2048)
    assert d_obs.shape == (5000, 2048)
    d_obs_cut = d_obs[:, :1250]  # 截取到1250步

    input_model = np.load('data/input_model.npy')  # (410, 5000)
    assert input_model.shape == (410, 5000)

    initial_model = np.load('data/model.npy')  # (410, 5000)
    initial_model_pad = np.pad(initial_model, ((0, 0), (padding_width, padding_width)), mode='edge')  # (410, 5820)
    assert initial_model.shape == (410, 5000)

    # 真实模型（用于生成观测数据）
    epsilon_true_pad = torch.tensor(initial_model_pad, dtype=torch.float32, device=device)
    sigma_true_pad = torch.tensor(eps_to_sig(initial_model_pad, freq), dtype=torch.float32, device=device)

    # 网格参数
    dx, dz = 0.06, 0.06
    dt = 1.4e-10
    npml = 10
    steps = 3500

    # 加载滤波后的源子波（只加载一次）
    if rank == 0:
        print("加载滤波后的源子波...")
    global _global_wavelet
    _global_wavelet = load_filtered_wavelet(expected_steps=steps, verbose=(rank==0))
    if rank == 0:
        print(f"子波形状: {_global_wavelet.shape}, 数据范围: [{_global_wavelet.min():.6e}, {_global_wavelet.max():.6e}]")
    wavelet = _global_wavelet  # 本地引用
    
    # 确保所有张量都是float32类型
    torch.set_default_dtype(torch.float32)
    
    # 参数范围
    min_eps, max_eps = 1.0, 9.0
    
    # 所有炮点（原始模型坐标系，0~4999）
    source_list_full = [(0, i) for i in range(0, 5000, 1)]
    
    if rank == 0:
        print(f"总炮点数量: {len(source_list_full)}")
        # 保存真实模型图像（未padding版本）
        plt.figure(figsize=(20, 4))
        plt.imshow(epsilon_true_pad[:,padding_width:-padding_width].cpu().numpy(), aspect='auto', cmap='jet', vmin=min_eps, vmax=max_eps)
        plt.title('True Epsilon Model (410x5000)')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'{path}epsilon_true.png')
        plt.close()
    
        plt.figure(figsize=(20, 4))
        plt.imshow(sigma_true_pad[:,padding_width:-padding_width].cpu().numpy(), aspect='auto', cmap='jet', vmin=min_eps, vmax=max_eps)
        plt.title('True Sigma Model (410x5000)')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'{path}sigma_true.png')
        plt.close()
    

    #==================== 模式1：生成观测数据 / 模式2：读取观测数据 =================================
    if generate_mode:
        # ========== 模式1：生成观测数据（采用移动窗口+MPI并行化） ==========
        if rank == 0:
            print("="*70)
            print("使用移动窗口策略生成合成观测数据...")
            print("="*70)
        
        # MPI进程分配：每个进程负责部分炮点
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
        
        print(f"Rank {rank}: 负责炮点 {start_shot} 到 {end_shot-1} (共 {len(local_shots)} 个)")
        
        # 初始化本地观测数据存储
        local_d_obs_list = []
        
        # 遍历本进程负责的炮点，使用移动窗口进行正演
        for shot_idx in local_shots:
            source_x = shot_idx  # 炮点在原始模型中的x坐标 (0~4999)
            
            # 提取以当前炮点为中心的窗口
            epsilon_window_np, offset_x = extract_window(
                epsilon_true_pad.cpu().numpy(), source_x, window_depth, window_width
            )
            sigma_window_np, _ = extract_window(
                sigma_true_pad.cpu().numpy(), source_x, window_depth, window_width
            )
            
            # 窗口内的source和receiver位置（相对于窗口坐标系）
            # source位于窗口顶层中间
            window_source = [(0, window_width // 2)]
            window_receiver = window_source.copy()  # 自激自收
            
            # 在窗口上进行正演模拟
            d_obs_shot = forward_model(
                epsilon_window_np,
                sigma_window_np,
                window_source,
                window_receiver,
                dt, dx, dz, npml, freq, steps,
                save_wavefield=False,
                wavelet=wavelet  # 传入已加载的子波
            )
            
            # 保存该炮的数据
            local_d_obs_list.append(d_obs_shot[0])  # d_obs_shot是(1, steps)，取第一行
            
            # 每处理100个炮点打印一次进度
            if (shot_idx - start_shot) % 100 == 0:
                print(f"Rank {rank}: 已处理 {shot_idx - start_shot + 1}/{len(local_shots)} 个炮点")
        
        # 将本进程的所有炮点数据堆叠为二维数组
        local_d_obs_array = np.stack(local_d_obs_list, axis=0)  # (n_local_shots, steps)
        
        print(f"Rank {rank}: 完成正演，本地数据形状: {local_d_obs_array.shape}")
        
        # 同步所有进程
        comm.Barrier()
        
        # 在rank 0上收集所有进程的观测数据
        all_d_obs_list = comm.gather(local_d_obs_array, root=0)
        
        if rank == 0:
            # 将各进程的观测数据按炮点顺序拼接
            d_obs_np = np.concatenate(all_d_obs_list, axis=0)  # (5000, steps)
            print(f"\n合成观测数据生成完成！")
            print(f"观测数据形状: {d_obs_np.shape}")
            print(f"数据类型: {d_obs_np.dtype}")
            print(f"数据范围: [{d_obs_np.min():.6e}, {d_obs_np.max():.6e}]")
        else:
            d_obs_np = None
        
        # 广播完整的观测数据到所有进程
        d_obs_np = comm.bcast(d_obs_np, root=0)
    
        # 转换为torch张量（模式1暂时转换，用于保存图像）
        d_obs_tensor_temp = torch.tensor(d_obs_np, dtype=torch.float32, device=device)  # (5000, steps)
        
        if rank == 0:
            # 保存观测数据图像
            plt.figure(figsize=(20, 8))
            plt.imshow(d_obs_tensor_temp.cpu().numpy().T, aspect='auto', cmap='seismic', 
                       vmin=-np.percentile(np.abs(d_obs_np), 98), 
                       vmax=np.percentile(np.abs(d_obs_np), 98))
            plt.title(f'Synthetic Observed Data (5000 shots x {steps} steps)', fontsize=14, fontweight='bold')
            plt.xlabel('Shot Number', fontsize=12)
            plt.ylabel('Time Step', fontsize=12)
            plt.colorbar(label='Amplitude')
            plt.tight_layout()
            plt.savefig(f"{path}d_obs_synthetic.png", dpi=300)
            plt.close()
            print(f"✓ 观测数据图像已保存: {path}d_obs_synthetic.png")
            
            print("="*70)
            
            # 保存观测数据为numpy文件（模式1的特殊输出）
            d_obs_save_path = 'data/d_obs_syn.npy'
            np.save(d_obs_save_path, d_obs_np)
            print(f"✓ 合成观测数据已保存: {d_obs_save_path}")
            print(f"数据形状: {d_obs_np.shape}, 数据类型: {d_obs_np.dtype}")
            print("="*70)
        
        # 模式1完成，退出程序
        if rank == 0:
            print("\n模式1完成！观测数据已生成并保存。")
            print("如需执行反演，请使用: python main.py --mode inversion")
            print("="*70)
        comm.Barrier()
        return  # 模式1结束，退出main函数
    
    else:
        # ========== 模式2：读取已有观测数据 ==========
        if rank == 0:
            print("="*70)
            print(f"读取观测数据: {d_obs_file}")
            print("="*70)
        
        if not os.path.exists(d_obs_file):
            if rank == 0:
                print(f"错误：找不到观测数据文件 {d_obs_file}")
                print("请先运行模式1生成观测数据：python main.py --mode generate")
            comm.Barrier()
            return
        
        # 读取观测数据
        d_obs_np = np.load(d_obs_file)
        
        if rank == 0:
            print(f"观测数据形状: {d_obs_np.shape}")
            print(f"数据类型: {d_obs_np.dtype}")
            print(f"数据范围: [{d_obs_np.min():.6e}, {d_obs_np.max():.6e}]")
            
            # 检查数据形状是否匹配
            if d_obs_np.shape != (5000, steps):
                print(f"警告：观测数据形状 {d_obs_np.shape} 与预期 (5000, {steps}) 不匹配")
                if d_obs_np.shape[1] != steps:
                    print(f"将截取到 {steps} 步")
                    d_obs_np = d_obs_np[:, :steps]
            
            # 保存观测数据图像（如果还没有）
            obs_img_path = f'{path}d_obs_synthetic.png'
            if not os.path.exists(obs_img_path):
                plt.figure(figsize=(20, 8))
                plt.imshow(d_obs_np.T, aspect='auto', cmap='seismic', 
                           vmin=-np.percentile(np.abs(d_obs_np), 98), 
                           vmax=np.percentile(np.abs(d_obs_np), 98))
                plt.title(f'Observed Data (5000 shots x {steps} steps)', fontsize=14, fontweight='bold')
                plt.xlabel('Shot Number', fontsize=12)
                plt.ylabel('Time Step', fontsize=12)
                plt.colorbar(label='Amplitude')
                plt.tight_layout()
                plt.savefig(obs_img_path, dpi=300)
                plt.close()
                print(f"✓ 观测数据图像已保存: {obs_img_path}")
            
            print("="*70)
        
        # 广播观测数据到所有进程
        if MPI_AVAILABLE and size > 1:
            if rank != 0:
                d_obs_np = None
            d_obs_np = comm.bcast(d_obs_np, root=0)
    
    # ========== 将观测数据转换为torch张量（两种模式共用） ==========
    d_obs_tensor = torch.tensor(d_obs_np, dtype=torch.float32, device=device)  # (5000, steps)
    
    if rank == 0:
        print(f"观测数据张量形状: {d_obs_tensor.shape}")
        print(f"观测数据张量设备: {d_obs_tensor.device}")
    
    # ========== UNet网络设置 ==========
    net_1 = None
    if rank == 0:
        print("初始化UNet网络...")
        # UNet输入是input_model (410, 5000)，输出也是(410, 5000)
        # 注意：UNet会自动裁剪输出到原始尺寸（410, 5000）
        # 减少通道数以降低内存消耗（针对5000宽度的模型）
        # 进一步减少通道数以避免OOM
        reduced_encoder_channels = [8, 16, 32, 64, 128]
        reduced_decoder_channels = [128, 64, 32, 16, 8]
        net_1 = UNet(in_channels=1, out_channels=1,
                     encoder_channels=reduced_encoder_channels,
                     decoder_channels=reduced_decoder_channels).to(device)
        print(f"网络参数数量: {get_parameter_number(net_1)}")
        print(f"使用减少的通道数配置以降低内存消耗")
    
    # ========== 网络输入处理 ==========
    # 将input_model作为网络输入（在所有epoch中保持不变）
    # 需要对input_model进行padding以满足UNet的下采样要求
    target_height, target_width = 410, 5000
    
    # 计算需要的padding（确保能被2的n次方整除）
    # 对于4层下采样，需要能被16整除
    required_height = ((target_height + 15) // 16) * 16
    required_width = ((target_width + 15) // 16) * 16
    
    pad_height = required_height - target_height
    pad_width = required_width - target_width
    
    # 确保padding是偶数
    if pad_height % 2 == 1:
        pad_height += 1
        required_height += 1
    if pad_width % 2 == 1:
        pad_width += 1
        required_width += 1
    
    if rank == 0:
        print(f"UNet输入padding: height={pad_height}, width={pad_width}")
        print(f"UNet输入尺寸: ({required_height}, {required_width})")
    
    # 对input_model进行padding并转换为torch张量
    input_model_padded = np.pad(input_model, 
                                 ((pad_height//2, pad_height//2), (pad_width//2, pad_width//2)), 
                                 mode='edge')
    
    # 扩展维度 [1, 1, H, W]
    net1_input = torch.tensor(input_model_padded, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    if rank == 0:
        print(f"网络输入形状: {net1_input.shape}")
    
    # ========== 训练设置 ==========
    optimizer = None
    if rank == 0:
        optimizer = torch.optim.Adam([
            {'params': net_1.parameters(), 'lr': learning_rate},
        ])
    
    loss_fn = torch.nn.MSELoss()
    num_epochs = 2001
    
    # 损失记录
    loss_history = []
    eps_loss_history = []
    
    # 训练循环
    if rank == 0:
        print("开始训练...")
    
    for epoch in range(num_epochs):
        # 生成当前epoch的大模型（仅用于可视化和窗口提取，不用于梯度计算）
        if rank == 0 and net_1 is not None:
            net_1.eval()  # 先设置为评估模式生成可视化用的模型
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.no_grad():
                _, _, net1_output_vis = net_1(net1_input)  # UNet返回 (encoder_out, decoder_out, final_output)
                epsilon_big = torch.sigmoid(net1_output_vis + offset) * (max_eps - min_eps) + min_eps  # (410, 5000)
            
            # Padding到(410, 5800)
            epsilon_big_pad = F.pad(epsilon_big, (padding_width, padding_width), mode='replicate')  # (410, 5800)
            
            # 计算对应的sigma
            sigma_big_pad = torch.tensor(eps_to_sig(epsilon_big_pad.cpu().numpy(), freq), 
                                        dtype=torch.float32, device=device)
        else:
            epsilon_big_pad = None
            sigma_big_pad = None
        
        # 广播大模型到所有进程
        if MPI_AVAILABLE and size > 1:
            epsilon_big_pad_np = epsilon_big_pad.cpu().numpy() if rank == 0 else None
            sigma_big_pad_np = sigma_big_pad.cpu().numpy() if rank == 0 else None
            
            epsilon_big_pad_np = comm.bcast(epsilon_big_pad_np, root=0)
            sigma_big_pad_np = comm.bcast(sigma_big_pad_np, root=0)
            
            if rank != 0:
                epsilon_big_pad = torch.tensor(epsilon_big_pad_np, dtype=torch.float32, device=device)
                sigma_big_pad = torch.tensor(sigma_big_pad_np, dtype=torch.float32, device=device)
        
        # 选择mini-batch炮点
        if rank == 0:
            selected_shot_indices = select_minibatch_shots(n_total_shots=5000, n_selected=100)
            print(f"Epoch {epoch}: 选择了 {len(selected_shot_indices)} 个炮点")
        else:
            selected_shot_indices = None
        
        # 广播选择的炮点索引
        selected_shot_indices = comm.bcast(selected_shot_indices, root=0)
        
        # MPI并行：每个进程处理mini-batch的不同子集
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
        print(f"Rank {rank}: 处理炮点 {mini_start} 到 {mini_end-1} (共 {len(local_selected_shots)} 个)")
        
        # 初始化梯度累积器（在padded模型空间中）
        accumulated_grad_eps = np.zeros((410, 5820), dtype=np.float32)
        
        # 初始化epoch损失
        epoch_loss = 0.0
        n_shots_processed = 0
        
        # 清零优化器梯度
        if rank == 0 and optimizer is not None:
            optimizer.zero_grad()
        
        # 遍历本进程负责的mini-batch炮点
        for shot_idx in local_selected_shots:
            source_x = shot_idx  # 炮点在原始模型中的x坐标
            
            # 提取当前炮点对应的窗口
            epsilon_window_np, offset_x = extract_window(
                epsilon_big_pad.cpu().numpy(), source_x, window_depth, window_width
            )
            sigma_window_np, _ = extract_window(
                sigma_big_pad.cpu().numpy(), source_x, window_depth, window_width
            )
            
            # 转换为torch张量（需要梯度）
            epsilon_window = torch.tensor(epsilon_window_np, dtype=torch.float32, device=device, requires_grad=True)
            sigma_window = torch.tensor(sigma_window_np, dtype=torch.float32, device=device)
            
            # 窗口内的source和receiver位置（相对于窗口，source在顶层中间）
            window_source = [(0, window_width // 2)]
            window_receiver = window_source.copy()
            
            # 获取对应的观测数据
            d_obs_shot = d_obs_tensor[shot_idx:shot_idx+1, :]  
            
            # 正演模拟（使用窗口模型）
            d_syn_shot = ForwardModelFunction.apply(
                epsilon_window, sigma_window, window_source, window_receiver,
                dt, dx, dz, npml, freq, steps
            )
            
            # 计算数据损失
            data_loss = loss_fn(d_syn_shot, d_obs_shot)
            
            # 反向传播以计算窗口梯度
            data_loss.backward()
            
            # 提取窗口的梯度
            if epsilon_window.grad is not None:
                grad_window_np = epsilon_window.grad.detach().cpu().numpy()
                
                # 将窗口梯度映射回完整模型
                grad_full = map_gradient_to_full(grad_window_np, offset_x, (410, 5820), window_depth, window_width)
                
                # 累积梯度
                accumulated_grad_eps += grad_full
            
            # 累积损失
            epoch_loss += data_loss.item()
            n_shots_processed += 1
            
            if rank == 0 and shot_idx % 20 == 0:
                print(f"Epoch {epoch}: 处理炮点 {shot_idx}/{len(selected_shot_indices)}, loss={data_loss.item():.6f}")
        
        # MPI进程间同步梯度（所有进程贡献，直接求和不除以size）
        if MPI_AVAILABLE and size > 1:
            accumulated_grad_global = np.empty_like(accumulated_grad_eps)
            comm.Allreduce(accumulated_grad_eps, accumulated_grad_global, op=MPI.SUM)
            accumulated_grad_eps = accumulated_grad_global  # 不除以size，因为每个进程处理不同的炮点
        
        # 同步损失（求和所有进程的损失）
        if MPI_AVAILABLE and size > 1:
            epoch_loss_global = comm.allreduce(epoch_loss, op=MPI.SUM)
            n_shots_processed_global = comm.allreduce(n_shots_processed, op=MPI.SUM)
            epoch_loss = epoch_loss_global
            n_shots_processed = n_shots_processed_global
        
        # 将累积的梯度转换为torch张量
        accumulated_grad_eps_tensor = torch.tensor(accumulated_grad_eps, dtype=torch.float32, device=device)
        
        # 确保padding区域的梯度为零（重要：padding区域不参与更新）
        accumulated_grad_eps_tensor[:, :padding_width] = 0.0
        accumulated_grad_eps_tensor[:, -padding_width:] = 0.0
        
        # 提取非padding区域的梯度 (410, 5000)
        grad_eps_no_pad = accumulated_grad_eps_tensor[:, padding_width:-padding_width]
        
        # 应用梯度钩子（浅层掩码等）
        grad_eps_no_pad = grad_hook(grad_eps_no_pad)
        
        # 在rank 0上，通过UNet反向传播更新网络参数
        if rank == 0 and net_1 is not None:
            # 设置为训练模式
            net_1.train()
            
            # 清理CUDA缓存和梯度缓存
            optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 重新进行UNet前向传播（需要梯度）
            _, _, net1_output = net_1(net1_input)  # UNet返回 (encoder_out, decoder_out, final_output)
            
            # 应用sigmoid和缩放
            epsilon_big_temp = torch.sigmoid(net1_output + offset) * (max_eps - min_eps) + min_eps
            
            # 使用累积的梯度进行反向传播
            # 这里使用手动反向传播，将累积的梯度传递给UNet输出
            epsilon_big_temp.backward(gradient=grad_eps_no_pad)
            
            # 更新网络参数
            optimizer.step()
            
            # 清理梯度缓存
            optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 平均损失
            avg_loss = epoch_loss / n_shots_processed if n_shots_processed > 0 else 0.0
            loss_history.append(avg_loss)
            
            print(f"Epoch {epoch}: 平均损失 = {avg_loss:.6f}")
            
            # 注意：padding区域的同步由模型生成时的padding操作自动处理
            # F.pad(..., mode='replicate') 会自动将边缘值复制到padding区域
            # 因此不需要额外的同步步骤
            
            # 可视化（每隔一定epoch）
            if epoch % 50 == 0 or epoch in [1, 2, 3, 5, 10]:
                # 保存当前模型
                plt.figure(figsize=(20, 4))
                plt.imshow(epsilon_big.detach().cpu().numpy(), aspect='auto', cmap='jet', vmin=min_eps, vmax=max_eps)
                plt.title(f'Epsilon Model (epoch {epoch})')
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(f'{path}epoch_{epoch}_epsilon.png')
                plt.close()
                
                # 保存梯度图
                plt.figure(figsize=(20, 4))
                plt.imshow(grad_eps_no_pad.detach().cpu().numpy(), aspect='auto', cmap='RdBu_r')
                plt.title(f'Accumulated Gradient (epoch {epoch})')
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(f'{path}epoch_{epoch}_gradient.png')
                plt.close()
                
                print(f"Epoch {epoch}: 已保存可视化结果")
        
        # 同步所有进程
        comm.Barrier()
        
        # 每隔一定epoch保存loss曲线
        if rank == 0 and epoch % 100 == 0 and epoch > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(loss_history, 'b-', linewidth=2)
            plt.title('Training Loss History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{path}loss_history_epoch_{epoch}.png')
            plt.close()

    # 训练结束后绘制最终的loss曲线（仅root）
    if rank == 0:
        print("训练完成！绘制最终loss曲线...")
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, 'b-', linewidth=2)
        plt.title('Final Training Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{path}loss_history_final.png')
        plt.close()
        
        # 保存最终模型
        with torch.no_grad():
            _, _, final_output = net_1(net1_input)  # UNet返回 (encoder_out, decoder_out, final_output)
            final_epsilon = torch.sigmoid(final_output + offset) * (max_eps - min_eps) + min_eps
        
        np.save(f'{path}final_epsilon_model.npy', final_epsilon.cpu().numpy())
        
        plt.figure(figsize=(20, 4))
        plt.imshow(final_epsilon.cpu().numpy(), aspect='auto', cmap='jet', vmin=min_eps, vmax=max_eps)
        plt.title('Final Epsilon Model')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'{path}final_epsilon_model.png')
        plt.close()
        
        print(f"训练完成！结果已保存到: {path}")

if __name__ == "__main__":
    main()