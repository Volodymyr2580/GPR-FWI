import numpy as np
import sys
import os
from Time_loop import reverse_time_loop
from Add_CPML import Add_CPML

def compute_gradient(epsilon, sigma, residual, source_list, receiver_list, dt, dx, dz, npml, freq, steps, sigma_required_gradient=True, wavefield_data=None):
    """
    计算FWI梯度
    epsilon: 当前介电常数模型
    sigma: 当前电导率模型
    residual: 残差
    source_list, receiver_list: 炮点、检波点坐标
    dt, dx, dz, npml, freq: 参数
    sigma_required_gradient: 是否需要对sigma求梯度
    wavefield_data: 正演波场数据（用于梯度计算）
    return: grad_eps, grad_sig (若不需要sigma梯度，则grad_sig为None)
    """
    xl, zl = epsilon.shape
    n_shots = len(source_list)
    assert residual.shape[0] == n_shots, "残差数量与炮点数量不一致"
    assert residual.shape[-1] == steps, "残差序列与时间步数不一致"
    
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
    for shot_idx, ref_pos in enumerate(receiver_list):
        receiver_pos = ref_pos
        receiver_pos_extended = (receiver_pos[0] + npml, receiver_pos[1] + npml)
        reverse_loop_gen = reverse_time_loop(xl, zl, dx, dz, dt, sigma_extended.copy(), 
                                             epsilon_extended.copy(), mu_extended.copy(), cpml_params, 
                                             steps, receiver_pos_extended, residual[shot_idx, :])
        if wavefield_data is not None:
            forward_wavefield = np.array(wavefield_data[shot_idx])
            # 计算正传波场的中心差分
            forward_diff = (forward_wavefield[2:] - forward_wavefield[:-2]) / (2 * dt)
            # 检查正传波场和中心差分的最大值和最小值并输出
            print("forward_wavefield max:", np.max(forward_wavefield), "min:", np.min(forward_wavefield))
            print("forward_diff max:", np.max(forward_diff), "min:", np.min(forward_diff))
            # 伴随波场反转
            adjoint_list = []
            for adj in reverse_loop_gen:
                adjoint_list.append(np.array(adj))
            adjoint_list = adjoint_list[::-1]  # 反转时间

            print("forward_wavefield shape:", forward_wavefield.shape)
            print("adjoint_list shape:", np.array(adjoint_list).shape)
            print("residual max:", np.max(residual), "min:", np.min(residual))
            print("梯度累加前 grad_eps max:", np.max(grad_eps), "min:", np.min(grad_eps))
            # 只用有效的时间步
            for k in range(1, steps-1):
                adjoint_field = adjoint_list[k]
                grad_eps += ep0 * adjoint_field[npml:npml+xl, npml:npml+zl] * forward_diff[k-1][npml:npml+xl, npml:npml+zl]
                if sigma_required_gradient:
                    grad_sig += adjoint_field[npml:npml+xl, npml:npml+zl] * forward_wavefield[k][npml:npml+xl, npml:npml+zl]
            print("梯度累加后 grad_eps max:", np.max(grad_eps), "min:", np.min(grad_eps))

    #添加浅层掩码
    mask = np.ones_like(grad_eps)
    mask[:20,:]=0
    grad_eps *= mask
    if sigma_required_gradient:
        grad_sig *= mask
        return grad_eps, grad_sig 
    else:
        return grad_eps, None