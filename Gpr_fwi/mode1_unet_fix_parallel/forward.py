import numpy as np
import sys
import os
sys.path.append('../RMSprop')
from Time_loop import time_loop
from Add_CPML import Add_CPML
from Wavelet import ricker

# 添加MPI支持
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("Warning: mpi4py not available, running in serial mode")

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
    
    local_source_list = source_list[start_idx:end_idx]
    local_receiver_list = receiver_list[start_idx:end_idx]
    local_n_shots = len(local_source_list)
    
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
    wavelet = ricker(t, freq)

    # 本地数据存储
    if local_n_shots > 0:
        local_data = np.zeros((local_n_shots, steps))
        local_wavefield_data = []  # 始终初始化，即使save_wavefield=False
    else:
        # 如果进程没有分配到shots，创建空数组
        local_data = np.zeros((0, steps))
        local_wavefield_data = []
    
    if save_wavefield:
        # 如果需要保存波场数据，保持列表结构
        pass
    else:
        # 如果不需要保存波场数据，设置为空列表
        local_wavefield_data = []

    # 处理本地分配的shots
    if local_n_shots > 0:
        if rank == 0:
            print(f"Process {rank}: Processing {len(local_source_list)} local shots")
            print(f"Process {rank}: Local source positions: {local_source_list}")
        
        for local_shot_idx, (source_pos, receiver_pos) in enumerate(zip(local_source_list, local_receiver_list)):
            source_pos_extended = (source_pos[0] + npml, source_pos[1] + npml)
            receiver_pos_extended = (receiver_pos[0] + npml, receiver_pos[1] + npml)
            
            if rank == 0 and local_shot_idx == 0:
                print(f"Process {rank}: Processing shot {local_shot_idx}, source_ext: {source_pos_extended}, receiver_ext: {receiver_pos_extended}")
            
            time_loop_gen = time_loop(xl, zl, dx, dz, dt, sigma_extended.copy(), epsilon_extended.copy(),
                                      mu_extended.copy(), cpml_params, wavelet, steps, source_pos_extended, receiver_pos_extended)
            shot_wavefield = []
            
            for step_idx, (ey_field, receiver_data) in enumerate(time_loop_gen):
                local_data[local_shot_idx, step_idx] = receiver_data
                if save_wavefield:
                    shot_wavefield.append(np.array(ey_field).copy())
            
            if save_wavefield:
                local_wavefield_data.append(shot_wavefield)
            
            if rank == 0 and local_shot_idx == 0:
                print(f"Process {rank}: Shot {local_shot_idx} completed, data range: [{np.min(local_data[local_shot_idx])}, {np.max(local_data[local_shot_idx])}]")
                print(f"Process {rank}: Shot {local_shot_idx} contains nan: {np.any(np.isnan(local_data[local_shot_idx]))}")
    else:
        if rank == 0:
            print(f"Process {rank}: No shots allocated to this process")
    
    # 收集所有进程的数据
    if rank == 0:
        print(f"Process {rank}: Gathering data from {size} processes")
        print(f"Process {rank}: Local data shape: {local_data.shape}")
        if local_n_shots > 0:
            print(f"Process {rank}: Local data range: [{np.min(local_data)}, {np.max(local_data)}]")
            print(f"Process {rank}: Local data contains nan: {np.any(np.isnan(local_data))}")
        print(f"Process {rank}: Local wavefield data length: {len(local_wavefield_data)}")
    
    all_data = comm.gather(local_data, root=0)
    
    if save_wavefield:
        # 确保所有进程都有相同的数据结构
        # 检查数据类型和结构
        if rank == 0:
            print(f"Process {rank}: Local wavefield data type: {type(local_wavefield_data)}")
            print(f"Process {rank}: Local wavefield data length: {len(local_wavefield_data)}")
            if len(local_wavefield_data) > 0:
                print(f"Process {rank}: First shot wavefield length: {len(local_wavefield_data[0])}")
        
        # 将复杂的嵌套列表转换为更简单的格式
        # 使用pickle序列化来避免MPI gather的结构问题
        import pickle
        serialized_wavefield = pickle.dumps(local_wavefield_data)
        
        # 收集序列化后的波场数据
        all_serialized_wavefield = comm.gather(serialized_wavefield, root=0)
        
        # 主进程反序列化数据
        if rank == 0:
            all_wavefield_data = []
            for serialized_data in all_serialized_wavefield:
                if len(serialized_data) > 0:
                    all_wavefield_data.append(pickle.loads(serialized_data))
                else:
                    all_wavefield_data.append([])
        else:
            all_wavefield_data = None
    
    # 主进程重新组织数据
    if rank == 0:
        print(f"Process {rank}: Received {len(all_data)} data arrays")
        
        # 重新组织观测数据
        data = np.zeros((n_shots, steps))
        data_idx = 0
        for i, proc_data in enumerate(all_data):
            n_local = proc_data.shape[0]
            print(f"Process {rank}: Process {i} data shape: {proc_data.shape}, range: [{np.min(proc_data)}, {np.max(proc_data)}], contains nan: {np.any(np.isnan(proc_data))}")
            data[data_idx:data_idx+n_local] = proc_data
            data_idx += n_local
        
        print(f"Process {rank}: Combined data shape: {data.shape}")
        print(f"Process {rank}: Combined data range: [{np.min(data)}, {np.max(data)}]")
        print(f"Process {rank}: Combined data contains nan: {np.any(np.isnan(data))}")
        
        # 重新组织波场数据
        if save_wavefield:
            wavefield_data = []
            for proc_wavefield in all_wavefield_data:
                wavefield_data.extend(proc_wavefield)
            # 不要尝试转换为numpy数组，保持列表格式
            # wavefield_data = np.array(wavefield_data)  # 这行会导致问题
            print(f"Process {rank}: Combined wavefield data length: {len(wavefield_data)}")
            if len(wavefield_data) > 0:
                print(f"Process {rank}: First shot wavefield length: {len(wavefield_data[0])}")
        else:
            wavefield_data = None
    else:
        # 非主进程创建空数组
        data = np.zeros((n_shots, steps))
        wavefield_data = None if not save_wavefield else []
    
    # 广播结果到所有进程
    data = comm.bcast(data, root=0)
    if save_wavefield:
        # 使用pickle序列化来广播波场数据，避免复杂数据结构的问题
        if rank == 0:
            serialized_wavefield = pickle.dumps(wavefield_data)
        else:
            serialized_wavefield = None
        
        # 广播序列化后的数据
        serialized_wavefield = comm.bcast(serialized_wavefield, root=0)
        
        # 非主进程反序列化数据
        if rank != 0:
            wavefield_data = pickle.loads(serialized_wavefield)
    
    if rank == 0:
        print(f"Process {rank}: Final data shape: {data.shape}")
        print(f"Process {rank}: Final data range: [{np.min(data)}, {np.max(data)}]")
        print(f"Process {rank}: Final data contains nan: {np.any(np.isnan(data))}")
    
    if save_wavefield:
        return data, wavefield_data
    else:
        return data

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
    wavelet = ricker(t, freq)

    data = np.zeros((n_shots, steps))
    if save_wavefield:
        wavefield_data = []

    for shot_idx, (source_pos, receiver_pos) in enumerate(zip(source_list, receiver_list)):
        source_pos_extended = (source_pos[0] + npml, source_pos[1] + npml)
        receiver_pos_extended = (receiver_pos[0] + npml, receiver_pos[1] + npml)
        time_loop_gen = time_loop(xl, zl, dx, dz, dt, sigma_extended.copy(), epsilon_extended.copy(),
                                  mu_extended.copy(), cpml_params, wavelet, steps, source_pos_extended, receiver_pos_extended)
        shot_wavefield = []
        for step_idx, (ey_field, receiver_data) in enumerate(time_loop_gen):
            data[shot_idx, step_idx] = receiver_data
            if save_wavefield:
                shot_wavefield.append(np.array(ey_field).copy())
        if save_wavefield:
            wavefield_data.append(shot_wavefield)
    if save_wavefield:
        return data, wavefield_data  # 保持列表格式，不转换为numpy数组
    else:
        return data