#!/usr/bin/env python3
"""
MPI并行计算主程序
用于运行GPR电磁波双参数FWI的并行计算
"""

import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
import matplotlib
from mpi4py import MPI

# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加当前目录到Python路径
sys.path.append('.')

from forward import forward_model
from gradient import compute_gradient, compute_tikhonov_gradient
from unet import UNet

def plot_and_save_data(data, title, filename, save_dir=None):
    """
    绘制并保存数据图像
    """
    if save_dir is None:
        save_dir = os.path.expanduser("~/fwi_results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(data.T, aspect='auto', cmap='seismic', origin='lower')
    plt.colorbar(label='Amplitude')
    plt.title(title)
    plt.xlabel('Shot Index')
    plt.ylabel('Time Step')
    plt.tight_layout()
    
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图像已保存到: {filepath}")

def plot_and_save_model(model, title, filename, save_dir=None):
    """
    绘制并保存模型图像
    """
    if save_dir is None:
        save_dir = os.path.expanduser("~/fwi_results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(model.T, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.tight_layout()
    
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"模型图像已保存到: {filepath}")

def plot_and_save_gradient(gradient, title, filename, save_dir=None):
    """
    绘制并保存梯度图像
    """
    if save_dir is None:
        save_dir = os.path.expanduser("~/fwi_results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(gradient.T, aspect='auto', cmap='RdBu_r', origin='lower')
    plt.colorbar(label='Gradient Value')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.tight_layout()
    
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"梯度图像已保存到: {filepath}")

def main():
    """
    MPI主程序
    """
    # 初始化MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # 设置错误处理
    try:
        # 设置numpy错误处理 - 忽略underflow警告，只关注严重错误
        np.seterr(under='ignore', divide='warn', invalid='warn', over='warn')
        
        # 检查内存使用
        if rank == 0:
            import psutil
            memory_info = psutil.virtual_memory()
            print(f"系统总内存: {memory_info.total / (1024**3):.1f} GB")
            print(f"可用内存: {memory_info.available / (1024**3):.1f} GB")
            print(f"内存使用率: {memory_info.percent:.1f}%")
            print("=" * 50)
    except ImportError:
        if rank == 0:
            print("psutil未安装，无法显示内存信息")
            print("=" * 50)
    
    if rank == 0:
        print(f"Starting MPI parallel computation with {size} processes")
        print("=" * 50)
    
    # 设置模型参数
    xl, zl = 100, 200  # 模型尺寸
    dt = 4e-11     # 时间步长
    dx = dz = 0.02      # 空间步长
    npml = 10          # PML层厚度
    freq = 4e8       # 震源主频 
    steps = 1000       # 时间步数
    
    # 设置反演参数
    num_epochs = 1  # 当前先执行一次epoch
    learning_rate = 0.1  # 学习率
    
    # 创建真实模型 (epsilon_true) - 固定不变
    epsilon_true = np.ones((xl, zl)) * 3  # 介电常数
    sigma = np.ones((xl, zl)) * 1e-4      # 电导率（固定）
    
    # 添加一些异常体到真实模型
    epsilon_true[40:60, 90:110] = 1.5
    epsilon_true[85:, :] = 6
    
    # 设置炮点和检波点位置（mode1: 炮点和检波点位置相同）
    source_list = [(0, i) for i in range(10, zl-10, 2)]  # 28个源点
    receiver_list = source_list.copy()
    n_shots = len(source_list)
    
    if rank == 0:
        print(f"真实模型尺寸: {xl} x {zl}")
        print(f"源点数量: {n_shots}")
        print(f"MPI进程数: {size}")
        print(f"每个进程处理的shot数: {n_shots // size + (1 if n_shots % size > 0 else 0)}")
        print("=" * 50)
    
    # 同步所有进程
    comm.Barrier()
    
    # 记录开始时间
    start_time = time.time()
    
    # ==================== 第一步：真实模型正演，得到观测数据 ====================
    if rank == 0:
        print("第一步：真实模型正演，生成观测数据...")
    
    # 执行真实模型正演（并行）
    data_obs= forward_model(
        epsilon_true, sigma, source_list, receiver_list, 
        dt, dx, dz, npml, freq, steps, 
        save_wavefield=False
    )
    
    # 收集所有进程的观测数据
    all_data_obs = comm.gather(data_obs, root=0)
    
    if rank == 0:
        # 合并所有进程的观测数据
        # 检查实际数据维度
        actual_shots = all_data_obs[0].shape[0] if len(all_data_obs) > 0 else n_shots
        data_obs_combined = np.zeros((actual_shots, steps))
        for proc_data in all_data_obs:
            data_obs_combined += proc_data
        
        # 更新n_shots为实际值
        n_shots = actual_shots
        
        # 绘制并保存真实观测数据
        plot_and_save_data(data_obs_combined, "真实模型观测数据 (data_obs)", "data_obs.png")
        
        print(f"真实模型正演完成，观测数据形状: {data_obs_combined.shape}")
        print(f"观测数据最大值: {np.max(np.abs(data_obs_combined)):.2e}")
        print(f"观测数据最小值: {np.min(data_obs_combined):.2e}")
        print("=" * 50)
    
    # 广播合并后的观测数据到所有进程
    data_obs_combined = comm.bcast(data_obs_combined if rank == 0 else None, root=0)
    
    # ==================== 第二步：开始反演迭代 ====================
    if rank == 0:
        print("第二步：开始反演迭代...")
    
    # 定义初始模型
    epsilon = np.ones_like(epsilon_true) * 3  # 初始介电常数
    epsilon[85:, :] = 6  # 部分已知信息
    
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 30)
        
        # 2.1 对待反演模型进行正演
        if rank == 0:
            print("2.1 当前模型正演...")
        
        try:
            data_model, wavefield_data = forward_model(
                epsilon, sigma, source_list, receiver_list, 
                dt, dx, dz, npml, freq, steps, 
                save_wavefield=True
            )
        except Exception as e:
            if rank == 0:
                print(f"正演计算失败: {e}")
                print("尝试减少内存使用...")
            
            # 尝试不保存波场数据
            data_model, wavefield_data = forward_model(
                epsilon, sigma, source_list, receiver_list, 
                dt, dx, dz, npml, freq, steps, 
                save_wavefield=False
            )
            if rank == 0:
                print("正演计算成功（无波场数据）")
        
        # 收集所有进程的正演数据
        all_data_model = comm.gather(data_model, root=0)
        all_wavefield_data = comm.gather(wavefield_data, root=0)
        
        if rank == 0:
            # 合并正演数据
            data_model_combined = np.zeros((n_shots, steps))
            for proc_data in all_data_model:
                data_model_combined += proc_data
            
            # 合并波场数据
            wavefield_data_combined = []
            for proc_wavefield in all_wavefield_data:
                wavefield_data_combined.extend(proc_wavefield)
            
            # 绘制并保存当前模型正演数据
            plot_and_save_data(data_model_combined, f"Epoch {epoch+1} 当前模型正演数据", f"epoch_{epoch+1}_data_model.png")
            
            print(f"当前模型正演完成，数据形状: {data_model_combined.shape}")
            print(f"波场数据长度: {len(wavefield_data_combined)}")
        
        # 广播合并后的数据到所有进程
        data_model_combined = comm.bcast(data_model_combined if rank == 0 else None, root=0)
        wavefield_data_combined = comm.bcast(wavefield_data_combined if rank == 0 else None, root=0)
        
        # 2.2 计算残差
        if rank == 0:
            print("2.2 计算残差...")
            residual = data_model_combined - data_obs_combined
            
            # 计算损失函数（MSE）
            loss = np.mean(residual**2)
            print(f"当前损失 (MSE): {loss:.6e}")
            
            # 绘制并保存残差数据
            plot_and_save_data(residual, f"Epoch {epoch+1} 残差数据 (residual)", f"epoch_{epoch+1}_residual.png")
        
        # 广播残差到所有进程
        residual = comm.bcast(residual if rank == 0 else None, root=0)
        
        # 2.3 计算梯度
        if rank == 0:
            print("2.3 计算梯度...")
        
        try:
            grad_eps, _ = compute_gradient(
                epsilon, sigma, residual, source_list, receiver_list,
                dt, dx, dz, npml, freq, steps,
                sigma_required_gradient=False, wavefield_data=wavefield_data_combined
            )
        except Exception as e:
            if rank == 0:
                print(f"梯度计算失败: {e}")
        
        if rank == 0:
            print(f"梯度计算完成，梯度形状: {grad_eps.shape}")
            print(f"梯度最大值: {np.max(np.abs(grad_eps)):.2e}")
            print(f"梯度最小值: {np.min(grad_eps):.2e}")
        
        # 2.4 梯度归一化
        if rank == 0:
            print("2.4 梯度归一化...")
            grad_norm = np.linalg.norm(grad_eps)
            if grad_norm > 0:
                grad_eps_normalized = grad_eps / grad_norm
            else:
                grad_eps_normalized = grad_eps
            
            # 绘制并保存归一化后的梯度
            plot_and_save_gradient(grad_eps_normalized, f"Epoch {epoch+1} 归一化梯度", f"epoch_{epoch+1}_gradient_normalized.png")
        
        # 广播归一化后的梯度到所有进程
        grad_eps_normalized = comm.bcast(grad_eps_normalized if rank == 0 else None, root=0)
        
        # 2.5 更新模型
        if rank == 0:
            print("2.5 更新模型...")
            epsilon_new = epsilon - learning_rate * grad_eps_normalized
            
            # 绘制并保存更新后的模型
            plot_and_save_model(epsilon_new, f"Epoch {epoch+1} 更新后的模型", f"epoch_{epoch+1}_epsilon_updated.png")
            
            # 计算模型变化
            model_change = np.linalg.norm(epsilon_new - epsilon)
            print(f"模型变化量: {model_change:.6e}")
            
            # 更新模型
            epsilon = epsilon_new.copy()
        
        # 广播更新后的模型到所有进程
        epsilon = comm.bcast(epsilon if rank == 0 else None, root=0)
        
        if rank == 0:
            print(f"Epoch {epoch + 1} 完成")
            print("=" * 50)
    
    # ==================== 第三步：保存最终结果 ====================
    if rank == 0:
        total_time = time.time() - start_time
        print(f"反演完成！总耗时: {total_time:.2f} 秒")
        
        # 保存最终结果
        try:
            import os
            home_dir = os.path.expanduser("~")
            result_file = os.path.join(home_dir, 'fwi_final_results.npz')
            np.savez(result_file,
                     epsilon_true=epsilon_true,
                     epsilon_final=epsilon,
                     sigma=sigma,
                     data_obs=data_obs_combined,
                     source_list=source_list,
                     receiver_list=receiver_list,
                     learning_rate=learning_rate,
                     num_epochs=num_epochs)
            print(f"最终结果已保存到: '{result_file}'")
        except Exception as e:
            print(f"警告: 无法保存结果: {e}")
        
        # 绘制最终对比图
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(epsilon_true.T, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(label='介电常数')
        plt.title('真实模型 (epsilon_true)')
        plt.xlabel('X')
        plt.ylabel('Z')
        
        plt.subplot(1, 3, 2)
        plt.imshow(epsilon.T, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(label='介电常数')
        plt.title('反演结果 (epsilon)')
        plt.xlabel('X')
        plt.ylabel('Z')
        
        plt.subplot(1, 3, 3)
        plt.imshow((epsilon - epsilon_true).T, aspect='auto', cmap='RdBu_r', origin='lower')
        plt.colorbar(label='误差')
        plt.title('反演误差 (epsilon - epsilon_true)')
        plt.xlabel('X')
        plt.ylabel('Z')
        
        plt.tight_layout()
        plt.savefig(os.path.expanduser('~/fwi_results/final_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("最终对比图已保存到: ~/fwi_results/final_comparison.png")
        print("=" * 50)
        print("MPI并行FWI反演计算成功完成！")
    
    # 同步所有进程
    comm.Barrier()

if __name__ == "__main__":
    main()
