import numpy as np
import cupy as cp
import sys
import os

sys.path.append("../RMSprop")
from .Time_loop_cupy import time_loop
from .Add_CPML import Add_CPML

# 添加MPI支持
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("Warning: mpi4py not available, running in serial mode")

# 波场降采样间隔
WAVEFIELD_SAMPLE_INTERVAL = 10

# # 预滤波震源路径
# SOURCE_WAVELET_PATH = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "..", "data", "wavelet_hp.npy")
# )
# _SOURCE_WAVELET_CACHE = None


# def load_source_wavelet(steps):
#     """
#     加载预滤波震源并截断到给定时间步长
#     """
#     global _SOURCE_WAVELET_CACHE
#     if _SOURCE_WAVELET_CACHE is None:
#         if not os.path.exists(SOURCE_WAVELET_PATH):
#             raise FileNotFoundError(f"未找到震源文件: {SOURCE_WAVELET_PATH}")
#         wavelet = np.load(SOURCE_WAVELET_PATH)
#         if wavelet.ndim != 1:
#             wavelet = np.asarray(wavelet).reshape(-1)
#         _SOURCE_WAVELET_CACHE = wavelet.astype(np.float64, copy=False)
#     if _SOURCE_WAVELET_CACHE.shape[0] < steps:
#         raise ValueError(
#             f"震源长度({_SOURCE_WAVELET_CACHE.shape[0]})小于所需时间步数({steps})，请检查 steps 设置"
#         )
#     return _SOURCE_WAVELET_CACHE[:steps].copy()

def load_source_wavelet(steps):
    """
    默认的震源加载函数。
    如果未提供外部震源数据，此函数会抛出异常，提示调用方显式提供。
    """
    raise RuntimeError(
        "No cached source wavelet available. Please provide the `wavelet` argument when calling forward_model."
    )


def _prepare_wavelet(wavelet, steps):
    if wavelet is None:
        return load_source_wavelet(steps)
    w = np.asarray(wavelet, dtype=np.float64).reshape(-1)
    if w.shape[0] < steps:
        raise ValueError(
            f"Provided wavelet length ({w.shape[0]}) is smaller than required steps ({steps})."
        )
    w = w[:steps].copy()
    return cp.asarray(w)


def forward_model_mpi(
    epsilon,
    sigma,
    source_list,
    receiver_list,
    dt,
    dx,
    dz,
    npml,
    freq,
    steps=1000,
    save_wavefield=False,
    wavelet=None,
):
    """
    MPI并行版本的2D FDTD正演模拟
    使用MPI并行处理多个shot_idx
    """
    if not MPI_AVAILABLE:
        return forward_model(
            epsilon,
            sigma,
            source_list,
            receiver_list,
            dt,
            dx,
            dz,
            npml,
            freq,
            steps,
            save_wavefield,
            wavelet=wavelet,
        )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    xl, zl = epsilon.shape
    epsilon = cp.asarray(epsilon)
    sigma = cp.asarray(sigma)
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
    epsilon_extended = cp.ones((xl + 2*npml, zl + 2*npml))
    sigma_extended = cp.ones((xl + 2*npml, zl + 2*npml)) * 0.005
    mu_extended = cp.ones((xl + 2*npml, zl + 2*npml))
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
    

    cpml_params = Add_CPML(xl, zl, cp.asnumpy(sigma_extended.copy()), cp.asnumpy(epsilon_extended.copy()), cp.asnumpy(mu_extended.copy()), dx, dz, dt)
    wavelet_arr = _prepare_wavelet(wavelet, steps)

    # 本地数据存储
    if local_n_shots > 0:
        local_data = cp.zeros((local_n_shots, steps))
        local_wavefield_data = []
    else:
        # 如果进程没有分配到shots，创建空数组
        local_data = cp.zeros((0, steps))
        local_wavefield_data = []

    # 处理本地分配的shots
    if local_n_shots > 0:
        if rank == 0:
            print(f"Process {rank}: Processing {len(local_source_list)} local shots (indices {start_idx}:{end_idx})")
            print(f"Process {rank}: Local source positions: {local_source_list}")
        
        for local_shot_idx, (source_pos, receiver_pos) in enumerate(zip(local_source_list, local_receiver_list)):
            source_pos_extended = (source_pos[0] + npml, source_pos[1] + npml)
            receiver_pos_extended = (receiver_pos[0] + npml, receiver_pos[1] + npml)
            
            if rank == 0 and local_shot_idx == 0:
                print(f"Process {rank}: Processing shot {local_shot_idx}, source_ext: {source_pos_extended}, receiver_ext: {receiver_pos_extended}")
            
            time_loop_gen = time_loop(
                xl,
                zl,
                dx,
                dz,
                dt,
                sigma_extended.copy(),
                epsilon_extended.copy(),
                mu_extended.copy(),
                cpml_params,
                wavelet_arr,
                steps,
                source_pos_extended,
                receiver_pos_extended,
            )
            shot_wavefield = []
            
            for step_idx, (ey_field, receiver_data) in enumerate(time_loop_gen):
                local_data[local_shot_idx, step_idx] = receiver_data
                if save_wavefield:
                    if step_idx % WAVEFIELD_SAMPLE_INTERVAL == 0:
                        shot_wavefield.append(ey_field.copy())
            
            if save_wavefield:
                local_wavefield_data.append(shot_wavefield)
            
            if rank == 0 and local_shot_idx == 0:
                rng = cp.asnumpy(local_data[local_shot_idx])
                print(f"Process {rank}: Shot {local_shot_idx} completed, data range: [{rng.min()}, {rng.max()}]")
                print(f"Process {rank}: Shot {local_shot_idx} contains nan: {np.isnan(rng).any()}")
    else:
        if rank == 0:
            print(f"Process {rank}: No shots allocated to this process")
    
    
    if save_wavefield:
        # 每个进程返回自己的本地波场数据，避免复杂的MPI传输
        return local_data, local_wavefield_data
    else:
        return local_data

def forward_model(
    epsilon,
    sigma,
    source_list,
    receiver_list,
    dt,
    dx,
    dz,
    npml,
    freq,
    steps=1000,
    save_wavefield=False,
    wavelet=None,
    force_serial=False,
):
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
    if (not force_serial) and MPI_AVAILABLE and MPI.COMM_WORLD.Get_size() > 1:
        return forward_model_mpi(
            epsilon,
            sigma,
            source_list,
            receiver_list,
            dt,
            dx,
            dz,
            npml,
            freq,
            steps,
            save_wavefield,
            wavelet=wavelet,
        )

    # 串行版本（原有代码）
    xl, zl = epsilon.shape
    epsilon = cp.asarray(epsilon)
    sigma = cp.asarray(sigma)
    n_shots = len(source_list)
    assert n_shots == len(receiver_list), "mode1: source_list和receiver_list长度必须一致"

    # 扩展模型到包含PML边界
    epsilon_extended = cp.ones((xl + 2*npml, zl + 2*npml))
    sigma_extended = cp.ones((xl + 2*npml, zl + 2*npml)) * 0.005
    mu_extended = cp.ones((xl + 2*npml, zl + 2*npml))
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

    cpml_params = Add_CPML(xl, zl, cp.asnumpy(sigma_extended.copy()), cp.asnumpy(epsilon_extended.copy()), cp.asnumpy(mu_extended.copy()), dx, dz, dt)
    wavelet_arr = _prepare_wavelet(wavelet, steps)

    data = cp.zeros((n_shots, steps))
    if save_wavefield:
        wavefield_data = []

    for shot_idx, (source_pos, receiver_pos) in enumerate(zip(source_list, receiver_list)):
        source_pos_extended = (source_pos[0] + npml, source_pos[1] + npml)
        receiver_pos_extended = (receiver_pos[0] + npml, receiver_pos[1] + npml)
        time_loop_gen = time_loop(
            xl,
            zl,
            dx,
            dz,
            dt,
            sigma_extended.copy(),
            epsilon_extended.copy(),
            mu_extended.copy(),
            cpml_params,
            wavelet_arr,
            steps,
            source_pos_extended,
            receiver_pos_extended,
        )
        shot_wavefield = []
        for step_idx, (ey_field, receiver_data) in enumerate(time_loop_gen):
            data[shot_idx, step_idx] = receiver_data
            if save_wavefield:
                if step_idx % WAVEFIELD_SAMPLE_INTERVAL == 0:
                    shot_wavefield.append(ey_field.copy())
        if save_wavefield:
            wavefield_data.append(shot_wavefield)
    if save_wavefield:
        return data, wavefield_data
    else:
        return data