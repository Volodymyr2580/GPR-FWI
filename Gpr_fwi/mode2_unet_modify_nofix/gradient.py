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
    计算FWI梯度 (mode2: 常规排列) + 二阶Tikhonov正则项
    epsilon: 当前介电常数模型
    sigma: 当前电导率模型
    residual: 残差 (n_shots, n_receivers, n_steps)
    source_list, receiver_list: 炮点、检波点坐标
    dt, dx, dz, npml, freq: 参数
    sigma_required_gradient: 是否需要对sigma求梯度
    wavefield_data: 正演波场数据（用于梯度计算）
    alpha_tikhonov_eps: epsilon的Tikhonov正则化参数
    alpha_tikhonov_sig: sigma的Tikhonov正则化参数
    return: grad_eps, grad_sig (若不需要sigma梯度，则grad_sig为None)
    """
    xl, zl = epsilon.shape
    n_shots = len(source_list)
    n_receivers = len(receiver_list)
    assert residual.shape[0] == n_shots, "残差数量与炮点数量不一致"
    assert residual.shape[1] == n_receivers, "残差接收器数量与接收器数量不一致"
    assert residual.shape[2] == steps, "残差序列与时间步数不一致"
    
    # 扩展模型到包含PML边界
    epsilon_extended = np.ones((xl + 2*npml, zl + 2*npml))
    sigma_extended = np.ones((xl + 2*npml, zl + 2*npml)) * 1e-4
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
    
    # 初始化梯度
    grad_eps = np.zeros_like(epsilon)
    grad_sig = np.zeros_like(sigma) if sigma_required_gradient else None
    
    ep0 = 8.841941282883074e-12
    
    # 对每个炮点计算梯度
    for shot_idx, source_pos in enumerate(source_list):
        # 获取正演波场数据
        if wavefield_data is not None:
            forward_wavefield = np.array(wavefield_data[shot_idx])
            # 计算正传波场的中心差分
            forward_diff = (forward_wavefield[2:] - forward_wavefield[:-2]) / (2 * dt)
        else:
            print("Warning: wavefield_data not provided, gradient calculation may be incomplete")
            continue
        
        # 获取当前炮的所有接收器残差
        shot_residual = residual[shot_idx, :, :]  # (n_receivers, n_steps)
        
        # 运行伴随波场时间循环 - 同时处理所有接收器
        reverse_loop_gen = reverse_time_loop(xl, zl, dx, dz, dt, sigma_extended.copy(), 
                                           epsilon_extended.copy(), mu_extended.copy(), cpml_params, 
                                           steps, receiver_list, shot_residual)
        
        # 收集伴随波场
        adjoint_list = []
        for adj in reverse_loop_gen:
            adjoint_list.append(np.array(adj))
        adjoint_list = adjoint_list[::-1]  # 反转时间
        
        
        # 计算梯度贡献 - 只用有效的时间步
        for k in range(1, steps-1):
            adjoint_field = adjoint_list[k]
            grad_eps += ep0 * adjoint_field[npml:npml+xl, npml:npml+zl] * forward_diff[k-1][npml:npml+xl, npml:npml+zl]
            if sigma_required_gradient:
                grad_sig += adjoint_field[npml:npml+xl, npml:npml+zl] * forward_wavefield[k][npml:npml+xl, npml:npml+zl]
    
    # 添加浅层掩码
    mask = np.ones_like(grad_eps)
    #mask[:20,:] = 0
    grad_eps *= mask
    if sigma_required_gradient:
        grad_sig *= mask
        return grad_eps, grad_sig
    else:
        return grad_eps, None