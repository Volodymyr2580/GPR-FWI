import numpy as np
import sys
import os
from .Time_loop import time_loop
from .Add_CPML import Add_CPML

# 添加MPI支持
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("Warning: mpi4py not available, running in serial mode")

# 全局变量：子波降采样率（每多少步保存一次波场）
WAVEFIELD_SUBSAMPLE = 10

# 全局变量：子波缓存（用于避免重复加载）
_global_wavelet_cache = None

def load_filtered_wavelet(expected_steps=3500, verbose=False, use_cache=True):
    """
    加载滤波后的源子波
    
    参数:
        expected_steps: 期望的子波长度
        verbose: 是否打印加载信息
        use_cache: 是否使用全局缓存（避免重复加载）
    
    返回：wavelet数组
    """
    global _global_wavelet_cache
    
    # 检查缓存
    if use_cache and _global_wavelet_cache is not None:
        cached_wavelet, cached_steps = _global_wavelet_cache
        if len(cached_wavelet) == expected_steps:
            if verbose:
                print(f"Using cached wavelet, shape: {cached_wavelet.shape}")
            return cached_wavelet
    
    # 加载文件
    wavelet_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'filtered_source_wavelet.npy')
    if not os.path.exists(wavelet_path):
        # 如果文件不存在，尝试相对于当前工作目录查找
        wavelet_path = 'data/filtered_source_wavelet.npy'
    
    if os.path.exists(wavelet_path):
        wavelet = np.load(wavelet_path)
        if verbose:
            print(f"Loaded filtered source wavelet from {wavelet_path}, shape: {wavelet.shape}")
        # 确保长度匹配
        if len(wavelet) != expected_steps:
            if verbose:
                print(f"Warning: wavelet length {len(wavelet)} != expected {expected_steps}, truncating or padding")
            if len(wavelet) > expected_steps:
                wavelet = wavelet[:expected_steps]
            else:
                wavelet = np.pad(wavelet, (0, expected_steps - len(wavelet)), mode='constant')
        
        # 更新缓存
        if use_cache:
            _global_wavelet_cache = (wavelet.copy(), expected_steps)
        
        return wavelet
    else:
        raise FileNotFoundError(f"Filtered source wavelet not found at {wavelet_path}")

def forward_model_mpi(epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps=1000, save_wavefield=False, wavelet=None):
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
    
    # 每个进程仅处理自己负责的shots切片
    local_source_list = source_list[start_idx:end_idx]
    local_receiver_list = receiver_list[start_idx:end_idx]
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
    
    # 如果没有提供子波，则加载（使用缓存）
    if wavelet is None:
        wavelet = load_filtered_wavelet(expected_steps=steps, use_cache=True)

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
            print(f"Process {rank}: Processing {len(local_source_list)} local shots (indices {start_idx}:{end_idx})")
            print(f"Process {rank}: Local source positions: {local_source_list}")
            if save_wavefield:
                print(f"Process {rank}: Wavefield will be subsampled every {WAVEFIELD_SUBSAMPLE} steps")
        
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
                # 波场降采样：每WAVEFIELD_SUBSAMPLE步保存一次
                if save_wavefield and (step_idx % WAVEFIELD_SUBSAMPLE == 0):
                    shot_wavefield.append(np.array(ey_field).copy())
            
            if save_wavefield:
                local_wavefield_data.append(shot_wavefield)
            
            if rank == 0 and local_shot_idx == 0:
                print(f"Process {rank}: Shot {local_shot_idx} completed, data range: [{np.min(local_data[local_shot_idx])}, {np.max(local_data[local_shot_idx])}]")
                print(f"Process {rank}: Shot {local_shot_idx} contains nan: {np.any(np.isnan(local_data[local_shot_idx]))}")
                if save_wavefield:
                    print(f"Process {rank}: Wavefield saved with {len(shot_wavefield)} time steps (subsampled from {steps})")
    else:
        if rank == 0:
            print(f"Process {rank}: No shots allocated to this process")
    
    
    if save_wavefield:
        # 每个进程返回自己的本地波场数据，避免复杂的MPI传输
        return local_data, local_wavefield_data
    else:
        return local_data

def forward_model(epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps=1000, save_wavefield=False, wavelet=None):
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
    # 如果MPI可用且进程数大于1，并且炮点数足够多，才使用并行版本
    # 当炮点数较少时（如单炮），使用串行版本避免负载不均
    n_shots = len(source_list)
    if MPI_AVAILABLE and MPI.COMM_WORLD.Get_size() > 1 and n_shots >= MPI.COMM_WORLD.Get_size():
        return forward_model_mpi(epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps, save_wavefield, wavelet)
    
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
    
    # 如果没有提供子波，则加载（使用缓存）
    if wavelet is None:
        wavelet = load_filtered_wavelet(expected_steps=steps, use_cache=True)

    data = np.zeros((n_shots, steps))
    if save_wavefield:
        wavefield_data = []
        print(f"Wavefield will be subsampled every {WAVEFIELD_SUBSAMPLE} steps")

    for shot_idx, (source_pos, receiver_pos) in enumerate(zip(source_list, receiver_list)):
        source_pos_extended = (source_pos[0] + npml, source_pos[1] + npml)
        receiver_pos_extended = (receiver_pos[0] + npml, receiver_pos[1] + npml)
        time_loop_gen = time_loop(xl, zl, dx, dz, dt, sigma_extended.copy(), epsilon_extended.copy(),
                                  mu_extended.copy(), cpml_params, wavelet, steps, source_pos_extended, receiver_pos_extended)
        shot_wavefield = []
        for step_idx, (ey_field, receiver_data) in enumerate(time_loop_gen):
            data[shot_idx, step_idx] = receiver_data
            # 波场降采样：每WAVEFIELD_SUBSAMPLE步保存一次
            if save_wavefield and (step_idx % WAVEFIELD_SUBSAMPLE == 0):
                shot_wavefield.append(np.array(ey_field).copy())
        if save_wavefield:
            wavefield_data.append(shot_wavefield)
    if save_wavefield:
        return data, wavefield_data  # 保持列表格式，不转换为numpy数组
    else:
        return data