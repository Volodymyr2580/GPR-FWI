#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行正演计算模块 - mode1
使用Pool进程池来并行处理每个炮点的正演计算
"""

import numpy as np
import sys
import os
from multiprocessing import Pool
import time
sys.path.append('../RMSprop')
from Time_loop import time_loop
from Add_CPML import Add_CPML
from Wavelet import ricker

def single_shot_forward(args):
    """
    单个炮点的正演计算函数（用于并行处理）
    args: (shot_idx, source_pos, epsilon, sigma, receiver_list, dt, dx, dz, npml, freq, steps, save_wavefield)
    """
    shot_idx, source_pos, epsilon, sigma, receiver_list, dt, dx, dz, npml, freq, steps, save_wavefield = args
    
    xl, zl = epsilon.shape
    
    # 扩展模型到包含PML边界
    epsilon_extended = np.ones((xl + 2*npml, zl + 2*npml))
    sigma_extended = np.ones((xl + 2*npml, zl + 2*npml)) * 1e-3
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
    
    # 生成Ricker子波
    t = np.arange(0, steps * dt, dt)
    wavelet = ricker(t, freq)
    
    # 调整源点位置到扩展网格（包含PML边界）
    source_pos_extended = (source_pos[0] + npml, source_pos[1] + npml)
    
    # 运行时间循环
    time_loop_gen = time_loop(xl, zl, dx, dz, dt, sigma_extended.copy(), epsilon_extended.copy(), 
                             mu_extended.copy(), cpml_params, wavelet, steps, source_pos_extended, 
                             source_pos_extended)  # 对于B-scan，接收点就是源点
    
    shot_data = np.zeros(steps)
    shot_wavefield = []
    
    for step_idx, (ey_field, receiver_data) in enumerate(time_loop_gen):
        shot_data[step_idx] = receiver_data
        
        if save_wavefield:
            shot_wavefield.append(np.array(ey_field).copy())
    
    if save_wavefield:
        return shot_idx, shot_data, np.array(shot_wavefield)
    else:
        return shot_idx, shot_data, None

def forward_model_parallel(epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps=1000, save_wavefield=False, n_processes=None):
    """
    2D FDTD并行正演模拟，返回合成观测数据
    epsilon: 介电常数模型 (2D array)
    sigma: 电导率模型 (2D array)
    source_list: 炮点坐标列表 [(x1,z1), (x2,z2), ...]
    receiver_list: 检波点坐标列表 [(x1,z1), (x2,z2), ...]
    dt, dx, dz: 时间/空间步长
    npml: PML层厚度
    freq: 震源主频
    steps: 时间步数
    save_wavefield: 是否保存波场数据用于FWI梯度计算
    n_processes: 进程数，默认为CPU核心数
    return: data (n_shots, n_steps) 或 (n_shots, n_receivers, n_steps), wavefield_data (如果save_wavefield=True)
    """
    import multiprocessing
    
    if n_processes is None:
        n_processes = multiprocessing.cpu_count()
    
    print(f"使用 {n_processes} 个进程进行并行正演计算...")
    start_time = time.time()
    
    # 准备并行计算的参数
    args_list = []
    for shot_idx, source_pos in enumerate(source_list):
        args = (shot_idx, source_pos, epsilon, sigma, receiver_list, dt, dx, dz, npml, freq, steps, save_wavefield)
        args_list.append(args)
    
    # 使用进程池进行并行计算
    with Pool(processes=n_processes) as pool:
        results = pool.map(single_shot_forward, args_list)
    
    # 整理结果
    n_shots = len(source_list)
    data = np.zeros((n_shots, steps))
    wavefield_data = [] if save_wavefield else None
    
    for shot_idx, shot_data, shot_wavefield in results:
        data[shot_idx] = shot_data
        if save_wavefield and shot_wavefield is not None:
            wavefield_data.append(shot_wavefield)
    
    if save_wavefield:
        wavefield_data = np.array(wavefield_data)
    
    elapsed_time = time.time() - start_time
    print(f"并行正演计算完成，耗时: {elapsed_time:.2f} 秒")
    
    if save_wavefield:
        return data, wavefield_data
    else:
        return data 