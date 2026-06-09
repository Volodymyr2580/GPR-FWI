import numpy as np
import sys
import os
import psutil
import gc
sys.path.append('../RMSprop')
from Time_loop import time_loop
from Add_CPML import Add_CPML
wavelet = np.load('wavelet_resampled.npy')
# 添加MPI支持
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("Warning: mpi4py not available, running in serial mode")

def get_memory_usage_forward():
    """获取当前进程的内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024  # 转换为MB
    memory_gb = memory_mb / 1024  # 转换为GB
    return memory_mb, memory_gb

def print_memory_status_forward(location="", rank=0):
    """打印内存使用状态"""
    if rank == 0:
        memory_mb, memory_gb = get_memory_usage_forward()
        print(f"[Forward内存监控 {location}] 进程内存使用: {memory_mb:.1f} MB ({memory_gb:.2f} GB)")
        gc.collect()

def forward_model_mpi(epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps=1000, save_wavefield=False):
    """
    MPI并行版本的2D FDTD正演模拟
    使用MPI并行处理多个shot_idx
    """
    if not MPI_AVAILABLE:
        return forward_model(epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps, save_wavefield)
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print_memory_status_forward("MPI正演开始", rank)
    
    xl, zl = epsilon.shape
    n_shots = len(source_list)
    assert n_shots == len(receiver_list), "mode1: source_list和receiver_list长度必须一致"
    
    # 计算每个进程处理的shot数量
    shots_per_proc = n_shots // size
    remainder = n_shots % size
    
    # 分配shot索引给每个进程
    if rank < remainder:
        start_idx = rank * (shots_per_proc + 1)
        end_idx = start_idx + shots_per_proc + 1
    else:
        start_idx = rank * shots_per_proc + remainder
        end_idx = start_idx + shots_per_proc
    
    local_source_list = source_list
    local_receiver_list = receiver_list
    local_n_shots = len(local_source_list)
    
    # 扩展模型到包含PML边界
    epsilon_extended = np.ones((xl + 2*npml, zl + 2*npml))
    sigma_extended = np.ones((xl + 2*npml, zl + 2*npml)) * 1e-4
    mu_extended = np.ones((xl + 2*npml, zl + 2*npml))
    epsilon_extended[npml:npml+xl, npml:npml+zl] = epsilon
    epsilon_extended[:npml, :] = epsilon_extended[npml, :]
    epsilon_extended[-npml:, :] = epsilon_extended[-npml-1, :]
    epsilon_extended[:, :npml] = epsilon_extended[:, npml].reshape(-1, 1)
    epsilon_extended[:, -npml:] = epsilon_extended[:, -npml-1].reshape(-1, 1)
    

    cpml_params = Add_CPML(xl, zl, sigma_extended.copy(), epsilon_extended.copy(), mu_extended.copy(), dx, dz, dt)
    t = np.arange(0, steps * dt, dt)

    # 本地数据存储
    if local_n_shots > 0:
        local_data = np.zeros((local_n_shots, steps))
        local_wavefield_data = []  # 始终初始化，即使save_wavefield=False
    else:
        # 如果进程没有分配到shots，创建空数组
        local_data = np.zeros((0, steps))
        local_wavefield_data = []

    # 处理本地分配的shots
    if local_n_shots > 0:
        if rank == 0:
            print(f"Process {rank}: Processing {len(local_source_list)} local shots")
            print(f"Process {rank}: Local source positions: {local_source_list}")
            print_memory_status_forward("开始处理shots", rank)
        
        for local_shot_idx, (source_pos, receiver_pos) in enumerate(zip(local_source_list, local_receiver_list)):
            source_pos_extended = (source_pos[0] + npml, source_pos[1] + npml)
            receiver_pos_extended = (receiver_pos[0] + npml, receiver_pos[1] + npml)
            
            if rank == 0 and local_shot_idx == 0:
                print(f"Process {rank}: Processing shot {local_shot_idx}, source_ext: {source_pos_extended}, receiver_ext: {receiver_pos_extended}")
            
            time_loop_gen = time_loop(xl, zl, dx, dz, dt, sigma_extended.copy(), epsilon_extended.copy(),
                                      mu_extended.copy(), cpml_params, wavelet, steps, source_pos_extended, receiver_pos_extended)
            
            if save_wavefield:
                # 只在需要保存波场时创建列表
                shot_wavefield = []
                for step_idx, (ey_field, receiver_data) in enumerate(time_loop_gen):
                    local_data[local_shot_idx, step_idx] = receiver_data
                    shot_wavefield.append(np.array(ey_field).copy())
                local_wavefield_data.append(shot_wavefield)
            else:
                # 不保存波场时，只提取接收点数据
                for step_idx, (ey_field, receiver_data) in enumerate(time_loop_gen):
                    local_data[local_shot_idx, step_idx] = receiver_data
                # ey_field会在循环结束后自动释放
            
            if rank == 0 and local_shot_idx == 0:
                print(f"Process {rank}: Shot {local_shot_idx} completed, data range: [{np.min(local_data[local_shot_idx])}, {np.max(local_data[local_shot_idx])}]")
                print(f"Process {rank}: Shot {local_shot_idx} contains nan: {np.any(np.isnan(local_data[local_shot_idx]))}")
                print_memory_status_forward(f"Shot {local_shot_idx} 完成", rank)
    else:
        if rank == 0:
            print(f"Process {rank}: No shots allocated to this process")
    
    
    # 合并所有进程的数据
    if rank == 0:
        # 主进程收集所有数据
        all_data = [local_data]  # 主进程的数据
        for i in range(1, size):
            data_from_proc = comm.recv(source=i, tag=0)
            all_data.append(data_from_proc)
        
        # 合并所有数据
        combined_data = np.concatenate(all_data, axis=0)
        
        if save_wavefield:
            # 对于波场数据，只返回主进程的数据（避免内存问题）
            return combined_data, local_wavefield_data
        else:
            return combined_data
    else:
        # 从进程发送数据到主进程
        comm.send(local_data, dest=0, tag=0)
        
        if save_wavefield:
            return None, local_wavefield_data
        else:
            return None

def forward_model(epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps=1000, save_wavefield=False):
    """
    mode1: 2D FDTD正演模拟，返回合成观测数据 [n_shots, n_time]
    如果MPI可用，自动使用并行版本
    epsilon: 介电常数模型 (2D array)
    sigma: 电导率模型 (2D array)
    source_list: 炮点坐标列表 [(x1,z1), (x2,z2), ...]
    receiver_list: 检波点坐标列表（与source_list相同）
    dt, dx, dz: 时间/空间步长
    npml: PML层厚度
    freq: 震源主频
    steps: 时间步数
    save_wavefield: 是否保存波场数据用于FWI梯度计算
    return: data (n_shots, n_time), wavefield_data (如果save_wavefield=True)
    """
    # 如果MPI可用且进程数大于1，使用并行版本
    if MPI_AVAILABLE and MPI.COMM_WORLD.Get_size() > 1:
        return forward_model_mpi(epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps, save_wavefield)
    
    # 串行版本（原有代码）
    xl, zl = epsilon.shape
    n_shots = len(source_list)
    assert n_shots == len(receiver_list), "mode1: source_list和receiver_list长度必须一致"

    # 扩展模型到包含PML边界
    epsilon_extended = np.ones((xl + 2*npml, zl + 2*npml))
    sigma_extended = np.ones((xl + 2*npml, zl + 2*npml)) * 1e-3
    mu_extended = np.ones((xl + 2*npml, zl + 2*npml))
    epsilon_extended[npml:npml+xl, npml:npml+zl] = epsilon
    sigma_extended[npml:npml+xl, npml:npml+zl] = sigma
    epsilon_extended[:npml, :] = epsilon_extended[npml, :]
    epsilon_extended[-npml:, :] = epsilon_extended[-npml-1, :]
    epsilon_extended[:, :npml] = epsilon_extended[:, npml].reshape(-1, 1)
    epsilon_extended[:, -npml:] = epsilon_extended[:, -npml-1].reshape(-1, 1)
    sigma_extended[:npml, :] = sigma_extended[npml, :]
    sigma_extended[-npml:, :] = sigma_extended[-npml-1, :]
    sigma_extended[:, :npml] = sigma_extended[:, npml].reshape(-1, 1)
    sigma_extended[:, -npml:] = sigma_extended[:, -npml-1].reshape(-1, 1)

    cpml_params = Add_CPML(xl, zl, sigma_extended.copy(), epsilon_extended.copy(), mu_extended.copy(), dx, dz, dt)
    t = np.arange(0, steps * dt, dt)

    data = np.zeros((n_shots, steps))
    if save_wavefield:
        wavefield_data = []

    for shot_idx, (source_pos, receiver_pos) in enumerate(zip(source_list, receiver_list)):
        source_pos_extended = (source_pos[0] + npml, source_pos[1] + npml)
        receiver_pos_extended = (receiver_pos[0] + npml, receiver_pos[1] + npml)
        time_loop_gen = time_loop(xl, zl, dx, dz, dt, sigma_extended.copy(), epsilon_extended.copy(),
                                  mu_extended.copy(), cpml_params, wavelet, steps, source_pos_extended, receiver_pos_extended)
        
        if save_wavefield:
            # 只在需要保存波场时创建列表
            shot_wavefield = []
            for step_idx, (ey_field, receiver_data) in enumerate(time_loop_gen):
                data[shot_idx, step_idx] = receiver_data
                shot_wavefield.append(np.array(ey_field).copy())
            wavefield_data.append(shot_wavefield)
        else:
            # 不保存波场时，只提取接收点数据
            for step_idx, (ey_field, receiver_data) in enumerate(time_loop_gen):
                data[shot_idx, step_idx] = receiver_data
            # ey_field会在循环结束后自动释放
            
    if save_wavefield:
        return data, wavefield_data  # 保持列表格式，不转换为numpy数组
    else:
        return data