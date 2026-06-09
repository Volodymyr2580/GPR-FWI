#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行梯度计算模块 - mode1
使用Pool进程池来并行处理每个炮点的梯度计算
"""

import numpy as np
import sys
import os
from multiprocessing import Pool
import time
from Time_loop import reverse_time_loop
from Add_CPML import Add_CPML

def single_shot_gradient(args):
    """
    单个炮点的梯度计算函数（用于并行处理）
    args: (shot_idx, source_pos, epsilon, sigma, residual, receiver_list, dt, dx, dz, npml, freq, steps, sigma_required_gradient, wavefield_data)
    """
    shot_idx, source_pos, epsilon, sigma, residual, receiver_list, dt, dx, dz, npml, freq, steps, sigma_required_gradient, wavefield_data = args
    
    xl, zl = epsilon.shape
    
    # 扩展模型到包含PML边界
    epsilon_extended = np.ones((xl + 2*npml, zl + 2*npml))
    sigma_extended = np.ones((xl + 2*npml, zl + 2*npml)) * 1e-4
    mu_extended = np.ones((xl + 2*npml, zl + 2*npml))
    
    # 复制内部区域
    epsilon_extended[npml:npml+xl, npml:npml+zl] = epsilon
    sigma_extended[npml:npml+xl, npml:npml+zl] = sigma
    
    # 设置PML边界 - 使用内部区域的值填充边界
    epsilon_extended[:npml, :] = epsilon_extended[npml, :]
    epsilon_extended[-npml:, :] = epsilon_extended[-npml-1, :]
    epsilon_extended[:, :npml] = epsilon_extended[:, npml].reshape(-1, 1)
    epsilon_extended[:, -npml:] = epsilon_extended[:, -npml-1].reshape(-1, 1)
    
    sigma_extended[:npml, :] = sigma_extended[npml, :]
    sigma_extended[-npml:, :] = sigma_extended[-npml-1, :]
    sigma_extended[:, :npml] = sigma_extended[:, npml].reshape(-1, 1)
    sigma_extended[:, -npml:] = sigma_extended[:, -npml-1].reshape(-1, 1)
    
    # 初始化CPML参数
    cpml_params = Add_CPML(xl, zl, sigma_extended.copy(), epsilon_extended.copy(), mu_extended.copy(), dx, dz, dt)
    
    # 初始化梯度
    grad_eps = np.zeros_like(epsilon)
    grad_sig = np.zeros_like(sigma) if sigma_required_gradient else None
    
    ep0 = 8.841941282883074e-12
    
    # 获取当前炮的残差
    shot_residual = residual[shot_idx, :]
    
    # 获取正演波场数据
    if wavefield_data is not None:
        forward_wavefield = np.array(wavefield_data[shot_idx])
        # 计算正传波场的中心差分
        forward_diff = (forward_wavefield[2:] - forward_wavefield[:-2]) / (2 * dt)
    else:
        print(f"Warning: Shot {shot_idx} - wavefield_data not provided")
        return shot_idx, grad_eps, grad_sig
    
    # 运行伴随波场时间循环
    receiver_pos = source_pos  # 对于B-scan，接收点就是源点
    receiver_pos_extended = (receiver_pos[0] + npml, receiver_pos[1] + npml)
    
    reverse_loop_gen = reverse_time_loop(xl, zl, dx, dz, dt, sigma_extended.copy(), 
                                         epsilon_extended.copy(), mu_extended.copy(), cpml_params, 
                                         steps, receiver_pos_extended, shot_residual)
    
    # 收集伴随波场
    adjoint_list = []
    for adj in reverse_loop_gen:
        adjoint_list.append(np.array(adj))
    adjoint_list = adjoint_list[::-1]  # 反转时间
    
    # 计算梯度贡献 - 只用有效的时间步
    for k in range(1, steps-1):
        adjoint_field = adjoint_list[k]
        grad_eps += ep0 * adjoint_field[npml:npml+xl, npml:npml+zl] * forward_diff[k-1][npml:npml+xl, npml:npml+zl]
        if sigma_required_gradient:
            grad_sig += adjoint_field[npml:npml+xl, npml:npml+zl] * forward_wavefield[k][npml:npml+xl, npml:npml+zl]
    
    # 添加浅层掩码
    mask = np.ones_like(grad_eps)
    mask[:20,:] = 0
    grad_eps *= mask
    if sigma_required_gradient and grad_sig is not None:
        grad_sig *= mask
    
    return shot_idx, grad_eps, grad_sig

def compute_gradient_parallel(epsilon, sigma, residual, source_list, receiver_list, dt, dx, dz, npml, freq, steps, sigma_required_gradient=True, wavefield_data=None, n_processes=None):
    """
    并行计算FWI梯度
    epsilon: 当前介电常数模型
    sigma: 当前电导率模型
    residual: 残差
    source_list, receiver_list: 炮点、检波点坐标
    dt, dx, dz, npml, freq: 参数
    sigma_required_gradient: 是否需要对sigma求梯度
    wavefield_data: 正演波场数据（用于梯度计算）
    n_processes: 进程数，默认为CPU核心数
    return: grad_eps, grad_sig (若不需要sigma梯度，则grad_sig为None)
    """
    import multiprocessing
    
    if n_processes is None:
        n_processes = multiprocessing.cpu_count()
    
    print(f"使用 {n_processes} 个进程进行并行梯度计算...")
    start_time = time.time()
    
    xl, zl = epsilon.shape
    n_shots = len(source_list)
    assert residual.shape[0] == n_shots, "残差数量与炮点数量不一致"
    assert residual.shape[-1] == steps, "残差序列与时间步数不一致"
    
    # 准备并行计算的参数
    args_list = []
    for shot_idx, source_pos in enumerate(source_list):
        args = (shot_idx, source_pos, epsilon, sigma, residual, receiver_list, dt, dx, dz, npml, freq, steps, sigma_required_gradient, wavefield_data)
        args_list.append(args)
    
    # 使用进程池进行并行计算
    with Pool(processes=n_processes) as pool:
        results = pool.map(single_shot_gradient, args_list)
    
    # 合并所有炮点的梯度
    grad_eps = np.zeros_like(epsilon)
    grad_sig = np.zeros_like(sigma) if sigma_required_gradient else None
    
    for shot_idx, shot_grad_eps, shot_grad_sig in results:
        grad_eps += shot_grad_eps
        if sigma_required_gradient and shot_grad_sig is not None:
            grad_sig += shot_grad_sig
    
    elapsed_time = time.time() - start_time
    print(f"并行梯度计算完成，耗时: {elapsed_time:.2f} 秒")
    
    if sigma_required_gradient:
        return grad_eps, grad_sig 
    else:
        return grad_eps, None 