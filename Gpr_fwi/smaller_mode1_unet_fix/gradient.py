import numpy as np
import sys
import os
from Time_loop import reverse_time_loop
from Add_CPML import Add_CPML

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

def compute_gradient(epsilon, sigma, residual, source_list, receiver_list, dt, dx, dz, npml, freq, steps, sigma_required_gradient=False, wavefield_data=None):
    """
    mode1: 只处理单一receiver（与source同位置）
    epsilon: 当前介电常数模型
    sigma: 当前电导率模型
    residual: [n_shots, n_time]
    source_list, receiver_list: 炮点、检波点坐标（长度一致，位置相同）
    return: grad_eps, grad_sig (若不需要sigma梯度，则grad_sig为None)
    """
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
            forward_wavefield = np.array(wavefield_data[shot_idx])
            forward_diff = (forward_wavefield[2:] - forward_wavefield[:-2]) / (2 * dt)
            adjoint_list = [np.array(adj) for adj in reverse_loop_gen][::-1]
            for k in range(1, steps-1):
                adjoint_field = adjoint_list[k]
                grad_eps += ep0 * adjoint_field[npml:npml+xl, npml:npml+zl] * forward_diff[k-1][npml:npml+xl, npml:npml+zl]
                if sigma_required_gradient:
                    grad_sig += adjoint_field[npml:npml+xl, npml:npml+zl] * forward_wavefield[k][npml:npml+xl, npml:npml+zl]
    mask = np.ones_like(grad_eps)
    mask[:5,:] = 0
    grad_eps *= mask
    if sigma_required_gradient:
        grad_sig *= mask
        return grad_eps, grad_sig
    else:
        return grad_eps, None