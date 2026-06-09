import numpy as np
import sys
import os

sys.path.append("../RMSprop")
from .Time_loop import time_loop
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
    wavelet_np = np.asarray(wavelet, dtype=np.float32).reshape(-1)

    if wavelet_np.shape[0] < steps:
        raise ValueError(
            f"Provided wavelet length ({wavelet_np.shape[0]}) is smaller than required steps ({steps})."
        )
    return np.ascontiguousarray(wavelet_np[:steps], dtype=np.float32)


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
    epsilon_extended = np.ones((xl + 2*npml, zl + 2*npml), dtype=np.float32)
    #### 格外注意，后续可能产生问题
    sigma_extended = np.ones((xl + 2*npml, zl + 2*npml), dtype=np.float32) * 0
    mu_extended = np.ones((xl + 2*npml, zl + 2*npml), dtype=np.float32)
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
    

    ep0 = 8.841941282883074e-12
    epsilon_cp = np.ascontiguousarray(epsilon_extended.copy() * ep0, dtype=np.float32)
    sigma_cp = np.ascontiguousarray(sigma_extended.copy(), dtype=np.float32)
    ca = (1 - sigma_cp * dt / 2 / epsilon_cp) / (1 + sigma_cp * dt / 2 / epsilon_cp)
    cb = 1 / epsilon_cp / (1 + sigma_cp * dt / 2 / epsilon_cp)
    ca_r = (epsilon_cp * 2) / (epsilon_cp * 2 + sigma_cp * dt)
    base = _get_cpml_base(xl, zl, dx, dz, dt, npml)
    cpml_params = _CPMLParams(base, ca, cb, ca_r)
    wavelet_arr = _prepare_wavelet(wavelet, steps)

    # 本地数据存储
    if local_n_shots > 0:
        local_data = np.zeros((local_n_shots, steps), dtype=np.float32)
        local_wavefield_data = []  # 始终初始化，即使save_wavefield=False
    else:
        # 如果进程没有分配到shots，创建空数组
        local_data = np.zeros((0, steps), dtype=np.float32)
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
                        shot_wavefield.append(np.array(ey_field).copy())
            
            if save_wavefield:
                local_wavefield_data.append(shot_wavefield)
            
            if rank == 0 and local_shot_idx == 0:
                print(f"Process {rank}: Shot {local_shot_idx} completed, data range: [{np.min(local_data[local_shot_idx])}, {np.max(local_data[local_shot_idx])}]")
                print(f"Process {rank}: Shot {local_shot_idx} contains nan: {np.any(np.isnan(local_data[local_shot_idx]))}")
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
    n_shots = len(source_list)
    assert n_shots == len(receiver_list), "mode1: source_list和receiver_list长度必须一致"

    # 扩展模型到包含PML边界
    epsilon_extended = np.ones((xl + 2*npml, zl + 2*npml), dtype=np.float32)
    sigma_extended = np.ones((xl + 2*npml, zl + 2*npml), dtype=np.float32) * 0
    mu_extended = np.ones((xl + 2*npml, zl + 2*npml), dtype=np.float32)
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

    ep0 = 8.841941282883074e-12
    epsilon_cp = np.ascontiguousarray(epsilon_extended.copy() * ep0, dtype=np.float32)
    sigma_cp = np.ascontiguousarray(sigma_extended.copy(), dtype=np.float32)
    ca = (1 - sigma_cp * dt / 2 / epsilon_cp) / (1 + sigma_cp * dt / 2 / epsilon_cp)
    cb = 1 / epsilon_cp / (1 + sigma_cp * dt / 2 / epsilon_cp)
    ca_r = (epsilon_cp * 2) / (epsilon_cp * 2 + sigma_cp * dt)
    base = _get_cpml_base(xl, zl, dx, dz, dt, npml)
    cpml_params = _CPMLParams(base, ca, cb, ca_r)
    wavelet_arr = _prepare_wavelet(wavelet, steps)

    data = np.zeros((n_shots, steps), dtype=np.float32)
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
                    shot_wavefield.append(np.array(ey_field).copy())
        if save_wavefield:
            wavefield_data.append(shot_wavefield)
    if save_wavefield:
        return data, wavefield_data  # 保持列表格式，不转换为numpy数组
    else:
        return data
_CPML_CACHE = {}

class _CPMLParams:
    def __init__(self, base, ca, cb, ca_r):
        self.npml = base["npml"]
        self.a_x = base["a_x"]
        self.b_x = base["b_x"]
        self.k_x = base["k_x"]
        self.a_z = base["a_z"]
        self.b_z = base["b_z"]
        self.k_z = base["k_z"]
        self.a_x_half = base["a_x_half"]
        self.b_x_half = base["b_x_half"]
        self.k_x_half = base["k_x_half"]
        self.a_z_half = base["a_z_half"]
        self.b_z_half = base["b_z_half"]
        self.k_z_half = base["k_z_half"]
        self.ca = ca
        self.cb = cb
        self.ca_r = ca_r

def _get_cpml_base(xl, zl, dx, dz, dt, npml):
    key = (xl, zl, dx, dz, dt, npml)
    entry = _CPML_CACHE.get(key)
    if entry is None:
        sigma_tmp = np.ones((xl + 2*npml, zl + 2*npml)) * 0
        epsilon_tmp = np.ones((xl + 2*npml, zl + 2*npml))
        mu_tmp = np.ones((xl + 2*npml, zl + 2*npml))
        cpml = Add_CPML(xl, zl, sigma_tmp, epsilon_tmp, mu_tmp, dx, dz, dt, npml)
        entry = {
            "npml": int(cpml.npml),
            "a_x": np.asarray(cpml.a_x, dtype=np.float32),
            "b_x": np.asarray(cpml.b_x, dtype=np.float32),
            "k_x": np.asarray(cpml.k_x, dtype=np.float32),
            "a_z": np.asarray(cpml.a_z, dtype=np.float32),
            "b_z": np.asarray(cpml.b_z, dtype=np.float32),
            "k_z": np.asarray(cpml.k_z, dtype=np.float32),
            "a_x_half": np.asarray(cpml.a_x_half, dtype=np.float32),
            "b_x_half": np.asarray(cpml.b_x_half, dtype=np.float32),
            "k_x_half": np.asarray(cpml.k_x_half, dtype=np.float32),
            "a_z_half": np.asarray(cpml.a_z_half, dtype=np.float32),
            "b_z_half": np.asarray(cpml.b_z_half, dtype=np.float32),
            "k_z_half": np.asarray(cpml.k_z_half, dtype=np.float32),
        }
        _CPML_CACHE[key] = entry
    return entry