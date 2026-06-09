#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU加速版本的forward模型
支持多GPU并行计算不同的炮点
"""
import numpy as np
import sys
import os

# 尝试导入CuPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    print("Warning: CuPy not available, GPU acceleration disabled")

from .Time_loop_gpu import time_loop_gpu, GPU_AVAILABLE as TIMELOOP_GPU_AVAILABLE
from .Time_loop import time_loop  # CPU版本作为fallback
from .Add_CPML import Add_CPML

# 添加MPI支持
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

# 全局变量：子波降采样率
WAVEFIELD_SUBSAMPLE = 10

# 全局变量：子波缓存
_global_wavelet_cache = None

def load_filtered_wavelet(expected_steps=3500, verbose=False, use_cache=True):
    """加载滤波后的源子波"""
    global _global_wavelet_cache
    
    if use_cache and _global_wavelet_cache is not None:
        cached_wavelet, cached_steps = _global_wavelet_cache
        if len(cached_wavelet) == expected_steps:
            if verbose:
                print(f"Using cached wavelet, shape: {cached_wavelet.shape}")
            return cached_wavelet
    
    wavelet_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'filtered_source_wavelet.npy')
    if not os.path.exists(wavelet_path):
        wavelet_path = 'data/filtered_source_wavelet.npy'
    
    if os.path.exists(wavelet_path):
        wavelet = np.load(wavelet_path)
        if verbose:
            print(f"Loaded filtered source wavelet from {wavelet_path}, shape: {wavelet.shape}")
        
        if len(wavelet) != expected_steps:
            if verbose:
                print(f"Warning: wavelet length {len(wavelet)} != expected {expected_steps}, adjusting")
            if len(wavelet) > expected_steps:
                wavelet = wavelet[:expected_steps]
            else:
                wavelet = np.pad(wavelet, (0, expected_steps - len(wavelet)), mode='constant')
        
        if use_cache:
            _global_wavelet_cache = (wavelet.copy(), expected_steps)
        
        return wavelet
    else:
        raise FileNotFoundError(f"Filtered source wavelet not found at {wavelet_path}")


def forward_model_multigpu(epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, 
                           freq, steps=1000, save_wavefield=False, wavelet=None, 
                           gpu_ids=[0, 1, 2, 3], verbose=False):
    """
    多GPU并行正演模拟
    
    参数:
        gpu_ids: 可用的GPU设备ID列表，例如 [0, 1, 2, 3] 表示使用4张GPU
    """
    if not GPU_AVAILABLE:
        if verbose:
            print("GPU不可用，回退到CPU版本")
        from .forward import forward_model
        return forward_model(epsilon, sigma, source_list, receiver_list, dt, dx, dz, 
                           npml, freq, steps, save_wavefield, wavelet)
    
    xl, zl = epsilon.shape
    n_shots = len(source_list)
    assert n_shots == len(receiver_list), "source_list和receiver_list长度必须一致"
    
    if verbose:
        print(f"使用多GPU并行模式，GPU设备: {gpu_ids}")
        print(f"总炮点数: {n_shots}, GPU数量: {len(gpu_ids)}")
    
    # 扩展模型到包含PML边界
    epsilon_extended = np.ones((xl + 2*npml, zl + 2*npml), dtype=np.float32)
    sigma_extended = np.ones((xl + 2*npml, zl + 2*npml), dtype=np.float32) * 1e-3
    mu_extended = np.ones((xl + 2*npml, zl + 2*npml), dtype=np.float32)
    
    epsilon_extended[npml:npml+xl, npml:npml+zl] = epsilon
    sigma_extended[npml:npml+xl, npml:npml+zl] = sigma
    
    # 边界扩展
    epsilon_extended[:npml, :] = epsilon_extended[npml, :]
    epsilon_extended[-npml:, :] = epsilon_extended[-npml-1, :]
    epsilon_extended[:, :npml] = epsilon_extended[:, npml].reshape(-1, 1)
    epsilon_extended[:, -npml:] = epsilon_extended[:, -npml-1].reshape(-1, 1)
    
    sigma_extended[:npml, :] = sigma_extended[npml, :]
    sigma_extended[-npml:, :] = sigma_extended[-npml-1, :]
    sigma_extended[:, :npml] = sigma_extended[:, npml].reshape(-1, 1)
    sigma_extended[:, -npml:] = sigma_extended[:, -npml-1].reshape(-1, 1)
    
    # 计算CPML参数
    cpml_params = Add_CPML(xl, zl, sigma_extended.copy(), epsilon_extended.copy(), 
                          mu_extended.copy(), dx, dz, dt)
    
    # 加载子波
    if wavelet is None:
        wavelet = load_filtered_wavelet(expected_steps=steps, use_cache=True, verbose=verbose)
    else:
        wavelet = np.asarray(wavelet, dtype=np.float32).reshape(-1)
        if wavelet.size != steps:
            if verbose:
                print(f"Warning: provided wavelet length {wavelet.size} != expected {steps}, auto-adjusting")
            if wavelet.size > steps:
                wavelet = wavelet[:steps]
            else:
                wavelet = np.pad(wavelet, (0, steps - wavelet.size), mode='constant')

    # 分配炮点到不同的GPU
    n_gpus = len(gpu_ids)
    shots_per_gpu = n_shots // n_gpus
    remainder = n_shots % n_gpus
    
    data = np.zeros((n_shots, steps), dtype=np.float32)
    wavefield_data = [] if save_wavefield else None
    
    # 使用多进程或多线程并行处理不同GPU上的炮点
    # 由于GIL限制，这里使用顺序处理但切换GPU设备
    # 真正的并行需要使用multiprocessing，但会增加复杂度
    
    shot_idx = 0
    for gpu_idx, device_id in enumerate(gpu_ids):
        # 计算当前GPU处理的炮点范围
        if gpu_idx < remainder:
            n_local_shots = shots_per_gpu + 1
        else:
            n_local_shots = shots_per_gpu
        
        if n_local_shots == 0:
            continue
        
        end_shot_idx = shot_idx + n_local_shots
        
        if verbose:
            print(f"GPU {device_id}: 处理炮点 {shot_idx} 到 {end_shot_idx-1}")
        
        # 处理当前GPU负责的所有炮点
        for local_shot in range(shot_idx, end_shot_idx):
            source_pos = source_list[local_shot]
            receiver_pos = receiver_list[local_shot]
            
            source_pos_extended = (source_pos[0] + npml, source_pos[1] + npml)
            receiver_pos_extended = (receiver_pos[0] + npml, receiver_pos[1] + npml)
            
            # 使用GPU加速的time_loop
            time_loop_gen = time_loop_gpu(
                xl, zl, dx, dz, dt, 
                sigma_extended.copy(), 
                epsilon_extended.copy(),
                mu_extended.copy(), 
                cpml_params, 
                wavelet, 
                steps, 
                source_pos_extended, 
                receiver_pos_extended,
                device_id=device_id  # 指定GPU设备
            )
            
            shot_wavefield = []
            for step_idx, (ey_field, receiver_data) in enumerate(time_loop_gen):
                data[local_shot, step_idx] = receiver_data
                if save_wavefield and (step_idx % WAVEFIELD_SUBSAMPLE == 0):
                    shot_wavefield.append(np.array(ey_field).copy())
            
            if save_wavefield:
                wavefield_data.append(shot_wavefield)
            
            if verbose and local_shot % 10 == 0:
                print(f"  GPU {device_id}: 完成炮点 {local_shot}/{end_shot_idx-1}")
        
        shot_idx = end_shot_idx
    
    if save_wavefield:
        return data, wavefield_data
    else:
        return data


def forward_model_gpu(epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, 
                     freq, steps=1000, save_wavefield=False, wavelet=None, 
                     device_id=0, verbose=False):
    """
    单GPU正演模拟（与原forward_model接口兼容）
    
    参数:
        device_id: GPU设备ID，默认为0
        
    优化：支持torch CUDA张量零拷贝转换
    """
    if not GPU_AVAILABLE:
        if verbose:
            print("GPU不可用，回退到CPU版本")
        from .forward import forward_model
        return forward_model(epsilon, sigma, source_list, receiver_list, dt, dx, dz, 
                           npml, freq, steps, save_wavefield, wavelet)
    
    # 检测输入类型并智能转换（避免GPU→CPU→GPU）
    import torch
    
    use_gpu_direct = False
    if isinstance(epsilon, torch.Tensor) and epsilon.is_cuda:
        # ✓ 零拷贝：torch CUDA → cupy
        with cp.cuda.Device(device_id):
            epsilon_gpu = cp.from_dlpack(epsilon.detach().contiguous())
            sigma_gpu = cp.from_dlpack(sigma.detach().contiguous()) if isinstance(sigma, torch.Tensor) else cp.asarray(sigma)
            xl, zl = epsilon_gpu.shape
            use_gpu_direct = True
    elif isinstance(epsilon, torch.Tensor):
        # CPU tensor
        epsilon = epsilon.detach().cpu().numpy()
        sigma = sigma.detach().cpu().numpy() if isinstance(sigma, torch.Tensor) else sigma
        xl, zl = epsilon.shape
    else:
        # numpy array
        xl, zl = epsilon.shape
    
    n_shots = len(source_list)
    assert n_shots == len(receiver_list), "source_list和receiver_list长度必须一致"
    
    # 扩展模型（根据数据位置选择操作）
    if use_gpu_direct:
        # 在GPU上直接扩展
        with cp.cuda.Device(device_id):
            epsilon_extended = cp.ones((xl + 2*npml, zl + 2*npml), dtype=cp.float32)
            sigma_extended = cp.ones((xl + 2*npml, zl + 2*npml), dtype=cp.float32) * 1e-3
            mu_extended = cp.ones((xl + 2*npml, zl + 2*npml), dtype=cp.float32)
            
            epsilon_extended[npml:npml+xl, npml:npml+zl] = epsilon_gpu
            sigma_extended[npml:npml+xl, npml:npml+zl] = sigma_gpu
    else:
        # 在CPU上扩展
        epsilon_extended = np.ones((xl + 2*npml, zl + 2*npml), dtype=np.float32)
        sigma_extended = np.ones((xl + 2*npml, zl + 2*npml), dtype=np.float32) * 1e-3
        mu_extended = np.ones((xl + 2*npml, zl + 2*npml), dtype=np.float32)
        
        epsilon_extended[npml:npml+xl, npml:npml+zl] = epsilon
        sigma_extended[npml:npml+xl, npml:npml+zl] = sigma
    
    # 边界扩展（GPU或CPU）
    if use_gpu_direct:
        with cp.cuda.Device(device_id):
            epsilon_extended[:npml, :] = epsilon_extended[npml, :]
            epsilon_extended[-npml:, :] = epsilon_extended[-npml-1, :]
            epsilon_extended[:, :npml] = epsilon_extended[:, npml].reshape(-1, 1)
            epsilon_extended[:, -npml:] = epsilon_extended[:, -npml-1].reshape(-1, 1)
            
            sigma_extended[:npml, :] = sigma_extended[npml, :]
            sigma_extended[-npml:, :] = sigma_extended[-npml-1, :]
            sigma_extended[:, :npml] = sigma_extended[:, npml].reshape(-1, 1)
            sigma_extended[:, -npml:] = sigma_extended[:, -npml-1].reshape(-1, 1)
            
            # CPML需要numpy，暂时转一次（CPML计算很快）
            cpml_params = Add_CPML(xl, zl, cp.asnumpy(sigma_extended.copy()), 
                                  cp.asnumpy(epsilon_extended.copy()), 
                                  cp.asnumpy(mu_extended.copy()), dx, dz, dt)
    else:
        epsilon_extended[:npml, :] = epsilon_extended[npml, :]
        epsilon_extended[-npml:, :] = epsilon_extended[-npml-1, :]
        epsilon_extended[:, :npml] = epsilon_extended[:, npml].reshape(-1, 1)
        epsilon_extended[:, -npml:] = epsilon_extended[:, -npml-1].reshape(-1, 1)
        
        sigma_extended[:npml, :] = sigma_extended[npml, :]
        sigma_extended[-npml:, :] = sigma_extended[-npml-1, :]
        sigma_extended[:, :npml] = sigma_extended[:, npml].reshape(-1, 1)
        sigma_extended[:, -npml:] = sigma_extended[:, -npml-1].reshape(-1, 1)
        
        cpml_params = Add_CPML(xl, zl, sigma_extended.copy(), epsilon_extended.copy(), 
                              mu_extended.copy(), dx, dz, dt)
    
    if wavelet is None:
        wavelet = load_filtered_wavelet(expected_steps=steps, use_cache=True, verbose=verbose)
    else:
        wavelet = np.asarray(wavelet, dtype=np.float32).reshape(-1)
        if wavelet.size != steps:
            if verbose:
                print(f"Warning: provided wavelet length {wavelet.size} != expected {steps}, auto-adjusting")
            if wavelet.size > steps:
                wavelet = wavelet[:steps]
            else:
                wavelet = np.pad(wavelet, (0, steps - wavelet.size), mode='constant')

    data = np.zeros((n_shots, steps), dtype=np.float32)
    if save_wavefield:
        wavefield_data = []
    
    # 统一处理：time_loop_gpu现在支持cupy输入
    for shot_idx, (source_pos, receiver_pos) in enumerate(zip(source_list, receiver_list)):
        source_pos_extended = (source_pos[0] + npml, source_pos[1] + npml)
        receiver_pos_extended = (receiver_pos[0] + npml, receiver_pos[1] + npml)
        
        # time_loop_gpu会自动识别输入类型（numpy或cupy）
        time_loop_gen = time_loop_gpu(
            xl, zl, dx, dz, dt,
            sigma_extended,  # 可能是numpy或cupy
            epsilon_extended,
            mu_extended,
            cpml_params,
            wavelet,
            steps,
            source_pos_extended,
            receiver_pos_extended,
            device_id=device_id
        )
        
        shot_wavefield = []
        for step_idx, (ey_field, receiver_data) in enumerate(time_loop_gen):
            data[shot_idx, step_idx] = receiver_data
            if save_wavefield and (step_idx % WAVEFIELD_SUBSAMPLE == 0):
                shot_wavefield.append(np.array(ey_field).copy())
        
        if save_wavefield:
            wavefield_data.append(shot_wavefield)
        
        if verbose and shot_idx % 10 == 0:
            print(f"完成炮点 {shot_idx}/{n_shots}")
    
    if save_wavefield:
        return data, wavefield_data
    else:
        return data

