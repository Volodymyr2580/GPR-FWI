import numpy as np
import sys
import os
sys.path.append('../RMSprop')
from Time_loop import time_loop
from Add_CPML import Add_CPML
from Wavelet import ricker

def forward_model(epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps=1000, save_wavefield=False):
    """
    2D FDTD正演模拟，返回合成观测数据
    epsilon: 介电常数模型 (2D array)
    sigma: 电导率模型 (2D array)
    source_list: 炮点坐标列表 [(x1,z1), (x2,z2), ...]
    receiver_list: 检波点坐标列表 [(x1,z1), (x2,z2), ...]
    dt, dx, dz: 时间/空间步长
    npml: PML层厚度
    freq: 震源主频
    steps: 时间步数
    save_wavefield: 是否保存波场数据用于FWI梯度计算
    return: data (n_shots, n_steps) 或 (n_shots, n_receivers, n_steps), wavefield_data (如果save_wavefield=True)
    """
    xl, zl = epsilon.shape
    n_shots = len(source_list)
    n_receivers = len(receiver_list)
    
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
    
    # 初始化数据存储
    data = np.zeros((n_shots, steps))
    if save_wavefield:
        wavefield_data = []
    
    # 对每个炮点进行正演
    for shot_idx, source_pos in enumerate(source_list):
        # 调整源点位置到扩展网格（包含PML边界）
        source_pos_extended = (source_pos[0] + npml, source_pos[1] + npml)
        
        # 运行时间循环
        time_loop_gen = time_loop(xl, zl, dx, dz, dt, sigma_extended.copy(), epsilon_extended.copy(), 
                                 mu_extended.copy(), cpml_params, wavelet, steps, source_pos_extended, 
                                 source_pos_extended)  # 对于B-scan，接收点就是源点
        
        shot_wavefield = []
        for step_idx, (ey_field, receiver_data) in enumerate(time_loop_gen):
            data[shot_idx, step_idx] = receiver_data
            
            if save_wavefield:
                shot_wavefield.append(np.array(ey_field).copy())
        
        if save_wavefield:
            wavefield_data.append(shot_wavefield)
    
    if save_wavefield:
        return data, np.array(wavefield_data)
    else:
        return data 