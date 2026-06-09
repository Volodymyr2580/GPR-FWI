import numpy as np
import sys
import os
from Time_loop import reverse_time_loop
from Add_CPML import Add_CPML

# 添加MPI支持
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("Warning: mpi4py not available, running in serial mode")

def compute_laplacian_2d(field, dx, dz):
    """
    计算二维拉普拉斯算子 ∇²f = ∂²f/∂x² + ∂²f/∂z²
    使用中心差分格式
    """
    nx, nz = field.shape
    laplacian = np.zeros_like(field)
    
    # 内部点的拉普拉斯算子
    laplacian[1:-1, 1:-1] = (
        (field[2:, 1:-1] - 2*field[1:-1, 1:-1] + field[:-2, 1:-1]) / (dx**2) +
        (field[1:-1, 2:] - 2*field[1:-1, 1:-1] + field[1:-1, :-2]) / (dz**2)
    )
    
    # 边界处理（使用一阶差分）
    # 左边界
    laplacian[0, 1:-1] = (
        (field[1, 1:-1] - field[0, 1:-1]) / (dx**2) +
        (field[0, 2:] - 2*field[0, 1:-1] + field[0, :-2]) / (dz**2)
    )
    # 右边界
    laplacian[-1, 1:-1] = (
        (field[-1, 1:-1] - field[-2, 1:-1]) / (dx**2) +
        (field[-1, 2:] - 2*field[-1, 1:-1] + field[-1, :-2]) / (dz**2)
    )
    # 上边界
    laplacian[1:-1, 0] = (
        (field[2:, 0] - 2*field[1:-1, 0] + field[:-2, 0]) / (dx**2) +
        (field[1:-1, 1] - field[1:-1, 0]) / (dz**2)
    )
    # 下边界
    laplacian[1:-1, -1] = (
        (field[2:, -1] - 2*field[1:-1, -1] + field[:-2, -1]) / (dx**2) +
        (field[1:-1, -1] - field[1:-1, -2]) / (dz**2)
    )
    # 四个角点
    laplacian[0, 0] = (field[1, 0] - field[0, 0]) / (dx**2) + (field[0, 1] - field[0, 0]) / (dz**2)
    laplacian[0, -1] = (field[1, -1] - field[0, -1]) / (dx**2) + (field[0, -1] - field[0, -2]) / (dz**2)
    laplacian[-1, 0] = (field[-1, 0] - field[-2, 0]) / (dx**2) + (field[-1, 1] - field[-1, 0]) / (dz**2)
    laplacian[-1, -1] = (field[-1, -1] - field[-2, -1]) / (dx**2) + (field[-1, -1] - field[-1, -2]) / (dz**2)
    
    return laplacian

def compute_tikhonov_gradient(model, dx, dz, alpha_tikhonov=0.01):
    """
    计算二阶Tikhonov正则项的梯度
    正则项: R(m) = (α/2) * ||∇²m||²
    梯度: ∇R(m) = α * ∇²(∇²m)
    """
    # 计算拉普拉斯算子
    laplacian = compute_laplacian_2d(model, dx, dz)
    # 计算拉普拉斯算子的拉普拉斯算子（四阶导数）
    tikhonov_grad =  compute_laplacian_2d(laplacian, dx, dz)
    # 对tikhonov_grad进行归一化处理
    max_abs = np.max(np.abs(tikhonov_grad))
    if max_abs > 0:
        tikhonov_grad = tikhonov_grad / max_abs
    
    return alpha_tikhonov *tikhonov_grad

def compute_gradient_mpi(epsilon, sigma, residual, source_list, receiver_list, dt, dx, dz, npml, freq, steps, sigma_required_gradient=False, wavefield_data=None, global_start_idx=0):
    """
    MPI并行版本的梯度计算
    使用MPI并行处理多个shot_idx
    """
    if not MPI_AVAILABLE:
        return compute_gradient(epsilon, sigma, residual, source_list, receiver_list, dt, dx, dz, npml, freq, steps, sigma_required_gradient, wavefield_data)
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    xl, zl = epsilon.shape
    n_shots = len(source_list)
    assert n_shots == len(receiver_list)
    
    
    # 本地梯度计算 - 严格按照原有串行版本的计算方法
    local_grad_eps = np.zeros_like(epsilon)
    local_grad_sig = np.zeros_like(sigma) if sigma_required_gradient else None
    ep0 = 8.841941282883074e-12
    # 扩展模型 - 严格按照原有方法
    epsilon_extended = np.ones((xl + 2*npml, zl + 2*npml))
    sigma_extended = np.ones((xl + 2*npml, zl + 2*npml)) * 1e-4
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

    for shot_idx, (source_pos, receiver_pos) in enumerate(zip(source_list, receiver_list)):
        receiver_pos_extended = (receiver_pos[0] + npml, receiver_pos[1] + npml)
        
        # 伴随反传 - 严格按照原有方法
        reverse_loop_gen = reverse_time_loop(xl, zl, dx, dz, dt, sigma_extended.copy(), epsilon_extended.copy(), mu_extended.copy(), cpml_params, steps, receiver_pos_extended, residual[shot_idx, :])
        
        if wavefield_data is not None:
            
            # wavefield_data现在是嵌套列表，需要转换为numpy数组进行计算
            forward_wavefield = np.array(wavefield_data[shot_idx])
                
            # 严格按照原有方法计算梯度
            forward_diff = (forward_wavefield[2:] - forward_wavefield[:-2]) / (2 * dt)
            adjoint_list = [np.array(adj) for adj in reverse_loop_gen][::-1]
                
            for k in range(1, steps-1):
                adjoint_field = adjoint_list[k]
                local_grad_eps += ep0 * adjoint_field[npml:npml+xl, npml:npml+zl] * forward_diff[k-1][npml:npml+xl, npml:npml+zl]
                if sigma_required_gradient:
                    local_grad_sig += adjoint_field[npml:npml+xl, npml:npml+zl] * forward_wavefield[k][npml:npml+xl, npml:npml+zl]
            
    
    # 收集所有进程的梯度
    all_grad_eps = comm.gather(local_grad_eps, root=0)
    if sigma_required_gradient:
        all_grad_sig = comm.gather(local_grad_sig, root=0)
    
    if rank == 0:
        # 合并所有进程的梯度
        grad_eps = np.zeros_like(epsilon)
        for proc_grad in all_grad_eps:
            grad_eps += proc_grad
        
        if sigma_required_gradient:
            grad_sig = np.zeros_like(sigma)
            for proc_grad in all_grad_sig:
                grad_sig += proc_grad
        else:
            grad_sig = None
        
        # 应用mask - 严格按照原有方法
        mask = np.ones_like(grad_eps)
        mask[:20,:] = 0
        grad_eps *= mask
        if sigma_required_gradient:
            grad_sig *= mask
    else:
        # 非主进程创建空梯度
        grad_eps = np.zeros_like(epsilon)
        grad_sig = np.zeros_like(sigma) if sigma_required_gradient else None
    
    # 广播结果到所有进程
    grad_eps = comm.bcast(grad_eps, root=0)
    if sigma_required_gradient:
        grad_sig = comm.bcast(grad_sig, root=0)
        return grad_eps, grad_sig
    else:
        return grad_eps, None

def compute_gradient(epsilon, sigma, residual, source_list, receiver_list, dt, dx, dz, npml, freq, steps, sigma_required_gradient=False, wavefield_data=None, global_start_idx=0):
    """
    mode1: 只处理单一receiver（与source同位置）
    如果MPI可用，自动使用并行版本
    epsilon: 当前介电常数模型
    sigma: 当前电导率模型
    residual: [n_shots, n_time]
    source_list, receiver_list: 炮点、检波点坐标（长度一致，位置相同）
    return: grad_eps, grad_sig (若不需要sigma梯度，则grad_sig为None)
    """
    # 如果MPI可用且进程数大于1，使用并行版本
    if MPI_AVAILABLE and MPI.COMM_WORLD.Get_size() > 1:
        return compute_gradient_mpi(epsilon, sigma, residual, source_list, receiver_list, dt, dx, dz, npml, freq, steps, sigma_required_gradient, wavefield_data, global_start_idx)
    
    # 串行版本（原有代码）
    xl, zl = epsilon.shape
    n_shots = len(source_list)
    assert n_shots == len(receiver_list)
    grad_eps = np.zeros_like(epsilon)
    grad_sig = np.zeros_like(sigma) if sigma_required_gradient else None
    ep0 = 8.841941282883074e-12
    for shot_idx, (source_pos, receiver_pos) in enumerate(zip(source_list, receiver_list)):
        # 扩展模型
        epsilon_extended = np.ones((xl + 2*npml, zl + 2*npml))
        sigma_extended = np.ones((xl + 2*npml, zl + 2*npml)) * 1e-4
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
        receiver_pos_extended = (receiver_pos[0] + npml, receiver_pos[1] + npml)
        # 伴随反传
        reverse_loop_gen = reverse_time_loop(xl, zl, dx, dz, dt, sigma_extended.copy(), epsilon_extended.copy(), mu_extended.copy(), cpml_params, steps, receiver_pos_extended, residual[shot_idx, :])
        if wavefield_data is not None:
            # 安全检查波场数据
            if shot_idx < len(wavefield_data) and len(wavefield_data[shot_idx]) > 0:
                # wavefield_data现在是嵌套列表，需要转换为numpy数组进行计算
                forward_wavefield = np.array(wavefield_data[shot_idx])
                forward_diff = (forward_wavefield[2:] - forward_wavefield[:-2]) / (2 * dt)
                adjoint_list = [np.array(adj) for adj in reverse_loop_gen][::-1]
                for k in range(1, steps-1):
                    adjoint_field = adjoint_list[k]
                    grad_eps += ep0 * adjoint_field[npml:npml+xl, npml:npml+zl] * forward_diff[k-1][npml:npml+xl, npml:npml+zl]
                    if sigma_required_gradient:
                        grad_sig += adjoint_field[npml:npml+xl, npml:npml+zl] * forward_wavefield[k][npml:npml+xl, npml:npml+zl]
            else:
                print(f"Warning: No valid wavefield data for shot {shot_idx}")
                print(f"Warning: wavefield_data length: {len(wavefield_data)}, shot {shot_idx} data length: {len(wavefield_data[shot_idx]) if shot_idx < len(wavefield_data) else 'N/A'}")
    mask = np.ones_like(grad_eps)
    mask[:20,:] = 0
    grad_eps *= mask
    if sigma_required_gradient:
        grad_sig *= mask
        return grad_eps, grad_sig
    else:
        return grad_eps, None