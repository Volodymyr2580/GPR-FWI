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
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from forward import forward_model
from gradient import compute_gradient, compute_tikhonov_gradient, compute_laplacian_2d
from unet import UNet

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
    def forward(ctx, epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps):
        """
        前向传播：执行FDTD正演模拟
        """
        # 将torch张量转换为numpy数组用于FDTD计算
        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()
        
        # 执行正演模拟
        data, wave_field = forward_model(epsilon_np, sigma_np, source_list, receiver_list, 
                           dt, dx, dz, npml, freq, steps, save_wavefield=True)
        
        # 计算照明项 - 每个进程计算本地照明
        xl, zl = epsilon_np.shape
        local_illumination = np.zeros((xl, zl))
        
        if wave_field is not None:
            for shot_idx in range(len(wave_field)):
                forward_wavefield = np.array(wave_field[shot_idx])
                # 对每个坐标点，计算所有时间步的波场值的平方和
                forward_wavefield_valid = forward_wavefield[:, npml:npml+xl, npml:npml+zl]
                illumination_contribution = np.sum(forward_wavefield_valid**2, axis=0)
                local_illumination += illumination_contribution
        
        # 在MPI环境下合并所有进程的照明
        illumination = local_illumination  # 默认使用本地照明
        
        # 安全地获取MPI通信器
        if MPI_AVAILABLE:
            try:
                # 直接使用导入的MPI模块
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                if comm.Get_size() > 1:
                    global_illumination = np.empty_like(local_illumination)
                    comm.Allreduce(local_illumination, global_illumination, op=MPI.SUM)
                    illumination = global_illumination
            except Exception as e:
                # 记录错误但不中断程序
                print(f"Warning: MPI照明项合并失败: {e}")
                # 继续使用本地照明
        
        # 计算H（展平的照明向量）
        H = illumination.flatten()
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
        ctx.H = H

        return data_tensor, H.reshape(epsilon_np.shape)
    
    @staticmethod
    def backward(ctx, grad_output,grad_H):
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
        H = ctx.H
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
        )
        # 使用保存的照明项进行梯度归一化
        # 取倒数，避免除零错误
        H_safe = H + 1e-10
        H_inv = 1.0 / H_safe
        
        # 将梯度展平并应用照明归一化
        grad_eps_flat = grad_eps.flatten()
        grad_eps_flat = H_inv * grad_eps_flat  # 元素级别乘法（等价于对角矩阵乘法）
        grad_eps = grad_eps_flat.reshape(grad_eps.shape)
        
        if grad_sig is not None:
            grad_sig_flat = grad_sig.flatten()
            grad_sig_flat = H_inv * grad_sig_flat
            grad_sig = grad_sig_flat.reshape(grad_sig.shape)
        # 转换回torch张量
        grad_eps_tensor = torch.from_numpy(grad_eps).to(epsilon.device)
        grad_sig_tensor = torch.from_numpy(grad_sig).to(sigma.device) if grad_sig is not None else None

        # 在MPI环境下对梯度做Allreduce求和，汇总所有进程的贡献
        if MPI_AVAILABLE:
            try:
                # 直接使用导入的MPI模块
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                if comm.Get_size() > 1:
                    # eps梯度
                    grad_eps_np_local = grad_eps_tensor.detach().cpu().numpy()
                    grad_eps_np_global = np.empty_like(grad_eps_np_local)
                    comm.Allreduce(grad_eps_np_local, grad_eps_np_global, op=MPI.SUM)
                    grad_eps_np_global = grad_eps_np_global / float(comm.Get_size())
                    # 归一化处理：将全局梯度归一化到[-1, 1]区间，防止数值过大或过小
                    max_abs = np.max(np.abs(grad_eps_np_global))
                    grad_eps_np_global = grad_eps_np_global / max_abs
                    grad_eps_tensor = torch.from_numpy(grad_eps_np_global).to(epsilon.device)

                    # sig梯度（若存在）
                    if grad_sig_tensor is not None:
                        grad_sig_np_local = grad_sig_tensor.detach().cpu().numpy()
                        grad_sig_np_global = np.empty_like(grad_sig_np_local)
                        comm.Allreduce(grad_sig_np_local, grad_sig_np_global, op=MPI.SUM)
                        grad_sig_np_global = grad_sig_np_global / float(comm.Get_size())
                        max_abs = np.max(np.abs(grad_sig_np_global))
                        grad_sig_np_global = grad_sig_np_global / max_abs
                        grad_sig_tensor = torch.from_numpy(grad_sig_np_global).to(sigma.device)
            except Exception as e:
                # 记录错误但不中断程序
                print(f"Warning: MPI梯度合并失败: {e}")


        return grad_eps_tensor, grad_sig_tensor, None, None, None, None, None, None, None, None

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
    #   grad[:20, :] = grad[:20, :] * 0
    
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


# ========== 主程序 ==========
def main():
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
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if rank == 0:
        print(f"使用设备: {device}")
        print(f"MPI进程数: {size}")
    
    # ========== 训练设置 ==========
    learning_rate = 1e-3  # 学习率
    alpha_tikhonov_eps = 0.015  # Tikhonov正则化参数
    alpha_tikhonov_sig = 0
    offset = 0
    
    # 构造结果文件夹名
    if rank == 0:
        result_dir = os.path.join(base_output_dir, f'lr{learning_rate:.0e}_alpha{alpha_tikhonov_eps:.3g}_offset{offset}')
        ensure_clean_dir(result_dir)
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
    
    # 真实模型参数
    epsilon_true = np.ones((xl, zl)) * 3
    epsilon_true[40:60, 90:110] = 1.5
    epsilon_true[85:, :] = 6
    sigma_true = np.ones((xl, zl)) * 1e-4
    
    # 转换为torch张量
    epsilon_true = torch.tensor(epsilon_true, dtype=torch.float32, device=device)
    sigma_true = torch.tensor(sigma_true, dtype=torch.float32, device=device)
    
    # 确保所有张量都是float32类型
    torch.set_default_dtype(torch.float32)
    
    # 参数范围
    min_eps, max_eps = 1.0, 6.0
    min_sig, max_sig = 1e-6, 1e-3
    
    # 移除未使用的浅层固定参数
    l_size = 0  # 浅层固定深度
    
    # 源点和接收点设置
    source_list = [(0, i) for i in range(10, zl-10, 2)]  # 28个源点
    receiver_list = source_list.copy()
    
    if rank == 0:
        print(f"源点数量: {len(source_list)}")
        print(f"接收点数量: {len(receiver_list)}")
        print(f"源点位置: {source_list}")
        print(f"接收点位置: {receiver_list}")

        # 额外保存：真实模型 epsilon_true 图像
        plt.figure(figsize=(6, 4))
        plt.imshow(epsilon_true.detach().cpu().numpy(), aspect='auto', cmap='jet', vmin=min_eps, vmax=max_eps)
        plt.title('True Epsilon Model')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'{path}epsilon_true.png')
        plt.close()
    
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
        # 将各进程的观测数据在shot维拼接为二维数组 [n_shots, n_steps]
        d_obs_np = np.concatenate(d_obs_list, axis=0) if len(d_obs_list) > 0 else np.zeros((0, steps), dtype=np.float32)
    else:
        d_obs_np = None
    # 广播拼接好的二维观测数据到所有进程
    d_obs_np = comm.bcast(d_obs_np, root=0)
    
    d_obs = torch.tensor(d_obs_np, dtype=torch.float32, device=device)
    if rank == 0:
        print(f"观测数据形状: {d_obs.shape}")
        print(f"观测数据类型: {d_obs.dtype}")

        # 额外保存：观测数据 d_obs 图像
        plt.figure(figsize=(12, 4))
        plt.imshow(d_obs.detach().cpu().numpy().T, aspect='auto', cmap='seismic')
        plt.title('Observed Data (d_obs)')
        plt.colorbar(); plt.tight_layout()
        plt.savefig(f"{path}d_obs.png"); plt.close()

    
    
    # ========== 数据加载器设置 ==========
    # 创建数据集
    if d_obs.dim() == 4:
        d_obs = d_obs.squeeze(0)  # 移除可能的batch维度
    
    dataset = MyDataset([d_obs], [source_list], [receiver_list])
    train_loader = data_utils.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # ========== UNet网络设置 ==========
    net_1 = None
    if rank == 0:
        print("初始化UNet网络...")
        net_1 = UNet(in_channels=1).to(device)  # 介电常数网络
        print(f"网络参数数量: {get_parameter_number(net_1)}")
    
    # ========== 网络输入处理 ==========
    # 对真实模型进行高斯滤波处理
    eps_input = torch.tensor(1/gaussian_filter(1/epsilon_true.cpu().numpy(), 5), dtype=torch.float32)
    
    # 计算padding
    target_height, target_width = 100, 200
    required_height = ((target_height + 31) // 32) * 32
    required_width = ((target_width + 31) // 32) * 32
    
    pad_height = required_height - target_height
    pad_width = required_width - target_width
    
    if pad_height % 2 == 1:
        pad_height += 1
    if pad_width % 2 == 1:
        pad_width += 1
    
    if rank == 0:
        print(f"padding: height={pad_height}, width={pad_width}")
    
    # 对称padding
    eps_input_padded = F.pad(eps_input, (pad_width//2, pad_width//2, pad_height//2, pad_height//2))
    
    # 扩展维度
    net1_input = eps_input_padded.unsqueeze(0).unsqueeze(0).to(device)
    
    # 广播网络输入到所有进程
    if MPI_AVAILABLE and size > 1:
        net1_input = comm.bcast(net1_input, root=0)
    
    # ========== 迁移项设置 ==========
    epsilon_mig = torch.zeros_like(epsilon_true, requires_grad=True)
    sigma_mig = torch.zeros_like(sigma_true, requires_grad=True)
    
    # 移除未使用的浅层固定模型变量
    #       epsilon_layer = epsilon_true[:l_size, :].clone()
    # sigma_layer = sigma_true[:l_size, :].clone()
    
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
    
    random_activate = False
    first_iter_saved = False
    for epoch in range(num_epochs):
        if rank == 0 and net_1 is not None:
            net_1.train()
        epoch_loss = 0
        total_samples = 0
        selected_indices = range(len(source_list))
        selected_source_list = source_list
        
        selected_receiver_list = selected_source_list.copy()
        selected_d_obs = d_obs[selected_indices,:]

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
        selected_d_obs_local = selected_d_obs[start_idx:end_idx, :]
        
        for i, (s_d_i, x_s_i, x_r_i) in enumerate(train_loader):
            batch_size_now = s_d_i.size(0)
            if rank == 0 and optimizer is not None:
                optimizer.zero_grad()
            
            # 准备各进程使用的模型参数outputs_eps
            if rank == 0 and net_1 is not None:
                # 网络前向传播仅在rank 0上进行
                batch_net1_input = net1_input[:, i:i+1, :, :].to(device)
                _, _, net1_output_model = net_1(batch_net1_input)

                # Sigmoid归一化
                max_value = net1_output_model.max().item()
                min_value = net1_output_model.min().item()
                mean_value = net1_output_model.mean().item()
                print(f"net1_output_model 统计量: 最大值={max_value}, 最小值={min_value}, 平均值={mean_value}")

                net1_ = torch.sigmoid(net1_output_model + offset)
                # 缩放到物理范围
                net1_o = net1_ * (max_eps - min_eps) + min_eps

                # 连接浅层固定模型和网络输出
                epsilon = net1_o
                # epsilon = torch.cat((epsilon_layer, net1_o), dim=0)

                # 添加迁移项并约束范围
                outputs_eps_root = epsilon + epsilon_mig
                outputs_eps_root[:l_size, :] = epsilon_true[:l_size, :]
                outputs_eps_root = torch.clamp(outputs_eps_root, min=min_eps, max=max_eps)
                # root上需要保留梯度，用于回传到UNet
                outputs_eps_root.retain_grad()

                # 广播数值到其他进程
                outputs_eps_np = outputs_eps_root.detach().cpu().numpy()
            else:
                outputs_eps_np = None

            # 广播outputs_eps到所有进程
            outputs_eps_np = comm.bcast(outputs_eps_np, root=0)

            # 各进程构建本地可求导副本；root使用原始变量以保持计算图
            if rank == 0 and net_1 is not None:
                outputs_eps_use = outputs_eps_root
            else:
                outputs_eps_use = torch.tensor(outputs_eps_np, dtype=torch.float32, device=device, requires_grad=True)
                outputs_eps_use.retain_grad()

            # 使用MPI并行正演，保存波场
            if rank == 0:
                print(f"Epoch {epoch}, Batch {i}: 开始正演模拟...")

            d_syn, H = ForwardModelFunction.apply(
                outputs_eps_use, sigma_true, selected_source_list, selected_receiver_list, 
                dt, dx, dz, npml, freq, steps
            )
            if rank == 0:
                if epoch % 50 == 0 or epoch in [1, 2, 3, 5, 10, 50, 100]:
                    h_save_dir = os.path.join(path, "illumination_H")
                    os.makedirs(h_save_dir, exist_ok=True)
                    plt.figure(figsize=(6, 4))
                    plt.imshow(H, cmap='jet', aspect='auto')
                    plt.colorbar(label='Illumination (H)')
                    plt.title(f"Illumination H (epoch={epoch}, batch={i})")
                    plt.xlabel('Z')
                    plt.ylabel('X')
                    img_save_path = os.path.join(h_save_dir, f"H_epoch{epoch}.png")
                    plt.savefig(img_save_path, bbox_inches='tight')
                    plt.close()
                    print(f"已保存H的图片到: {img_save_path}")
            # 确保数据类型一致 - 都转换为float32
            d_syn = d_syn.float()
            selected_d_obs_local = selected_d_obs_local.float()

            if rank == 0:
                print(f"合成数据形状(本进程): {d_syn.shape}")
                print(f"合成数据类型: {d_syn.dtype}")
                print(f"观测数据形状(本进程): {selected_d_obs_local.shape}")
                print(f"观测数据类型: {selected_d_obs_local.dtype}")

            # 额外保存：第一次迭代的 d_syn 与 residual 图像（需要全局拼接）
            if not first_iter_saved and epoch == 0 and i == 0:
                d_syn_local_np = d_syn.detach().cpu().numpy()
                d_syn_list = comm.gather(d_syn_local_np, root=0)
                if rank == 0:
                    d_syn_global = np.concatenate(d_syn_list, axis=0)
                    plt.figure(figsize=(12, 4))
                    plt.imshow(d_syn_global.T, aspect='auto', cmap='seismic')
                    plt.title('First Iteration Synthetic Data (d_syn)')
                    plt.colorbar(); plt.tight_layout()
                    plt.savefig(f"{path}d_syn_first.png"); plt.close()

                    residual = (d_syn_global - d_obs.detach().cpu().numpy())
                    plt.figure(figsize=(12, 4))
                    plt.imshow(residual.T, aspect='auto', cmap='seismic')
                    plt.title('First Iteration Residual (d_syn - d_obs)')
                    plt.colorbar(); plt.tight_layout()
                    plt.savefig(f"{path}residual_first.png"); plt.close()
                first_iter_saved = True
            
            # 计算数据损失
            data_loss = loss_fn(d_syn, selected_d_obs_local)
            
            # 计算正则化损失（对当前使用的模型参数添加正则）
            reg_loss = tikhonov_reg(outputs_eps_use, sigma_true, alpha_tikhonov_eps, alpha_tikhonov_sig, dx, dz)
            
            # 总损失
            total_loss = data_loss + reg_loss
            
            if np.isnan(float(total_loss.item())):
                raise ValueError('loss is nan while training')
            
            # 反向传播
            total_loss.backward(retain_graph=True)

            # 参数更新
            if rank == 0 and optimizer is not None:
                optimizer.step()

            # 统计与可视化（仅root）
            if rank == 0:
                epoch_loss += total_loss.item() * batch_size_now
                total_samples += batch_size_now

                if epoch % 50 == 0 or epoch in [1, 2, 3, 5, 10, 50, 100]:
                    plt.figure(figsize=(12, 4))
                    plt.imshow(d_syn.detach().cpu().numpy().T, aspect='auto', cmap='seismic')
                    plt.title(f'Synthetic Data (epoch {epoch})')
                    plt.colorbar(); plt.tight_layout()
                    plt.savefig(f"{path}epoch_{epoch}_d_syn.png"); plt.close()

                    # 可视化结果
                    plt.figure(figsize=(15, 5))
                    plt.imshow(outputs_eps_use.detach().cpu().numpy(), aspect='auto', 
                              cmap='jet', vmin=0, vmax=10)
                    plt.title(f'Epsilon (epoch {epoch})')
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(f'{path}epoch_{epoch}_epsilon.png')
                    plt.close()
                    print(f"Epoch {epoch}: 已保存epsilon图")
                    
                    # 检查梯度状态
                    print(f"Epoch {epoch}: outputs_eps.grad is None: {outputs_eps_use.grad is None}")
                    print(f"Epoch {epoch}: outputs_eps.requires_grad: {outputs_eps_use.requires_grad}")
                    
                    # 可视化梯度
                    if outputs_eps_use.grad is not None:
                        print(f"Epoch {epoch}: 开始绘制梯度...")
                        grad_np = outputs_eps_use.grad.detach().cpu().numpy()
                        print(f"Epoch {epoch}: 梯度形状: {grad_np.shape}")
                        print(f"Epoch {epoch}: 梯度范围: [{np.min(grad_np):.6f}, {np.max(grad_np):.6f}]")
                        
                        # 归一化梯度
                        max_abs_grad = np.max(np.abs(grad_np))
                        if max_abs_grad != 0:
                            grad_np_norm = grad_np / max_abs_grad
                        else:
                            grad_np_norm = grad_np.copy()
                        print(f"Epoch {epoch}: 归一化后梯度范围: [{np.min(grad_np_norm):.6f}, {np.max(grad_np_norm):.6f}]")
                        
                        # 梯度图（归一化后）
                        plt.figure(figsize=(15, 5))
                        plt.imshow(grad_np_norm, aspect='auto', cmap='RdBu_r', 
                                  vmin=-1, vmax=1)
                        plt.title(f'Epsilon Gradient (Normed, epoch {epoch})')
                        plt.colorbar()
                        plt.tight_layout()
                        plt.savefig(f'{path}epoch_{epoch}_epsilon_grad.png')
                        plt.close()
                        print(f"Epoch {epoch}: 已保存归一化梯度图")
                        
                        # 保存归一化梯度数据
                        #np.save(f'{path}epoch_{epoch}_epsilon_grad.npy', grad_np_norm)
                        print(f"Epoch {epoch}: 已保存归一化梯度数据")
                    else:
                        print(f"Epoch {epoch}: 梯度为None，无法绘制")
                    
                    # 保存数据
                    #np.save(f'{path}epoch_{epoch}_epsilon.npy', outputs_eps_use.detach().cpu().numpy())
                    print(f"Epoch {epoch}: 已保存epsilon数据")

        # 记录损失（仅root）
        if rank == 0 and total_samples > 0:
            epoch_loss = epoch_loss / total_samples
            loss_model_eps = loss_fn(outputs_eps_use, epsilon_true) / loss_fn(epsilon_true, torch.zeros_like(epsilon_true))

            loss_history.append(epoch_loss)
            eps_loss_history.append(loss_model_eps.item())

            print(f'Epoch {epoch}: Total Loss = {epoch_loss:.6f}, Eps Loss = {loss_model_eps:.6f}')

            if epoch % 100 == 0 and epoch > 0:
                plot_loss_curves(loss_history, eps_loss_history, path, epoch)
        
        # 同步所有进程
        comm.Barrier()
        
        # 保持仅在rank 0更新网络参数，其他进程不持有网络
        # 无需在此同步网络权重

    # 训练结束后绘制最终的loss曲线（仅root）
    if rank == 0:
        print("绘制最终loss曲线...")
        plot_loss_curves(loss_history, eps_loss_history, path, num_epochs-1)
        print("训练完成！最终loss曲线已保存。")

if __name__ == "__main__":
    main()