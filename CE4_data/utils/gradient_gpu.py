#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU版本的梯度计算 - 与forward_gpu.py和Time_loop_gpu.py统一使用cupy

关键统一：
1. 所有数组操作使用cupy（GPU）
2. 调用Time_loop_gpu的reverse_time_loop_gpu
3. 避免GPU→CPU→GPU传输
"""
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

from .Time_loop_gpu import reverse_time_loop_gpu
from .Add_CPML import Add_CPML

WAVEFIELD_SUBSAMPLE = 10  # 增大降采样，节省内存


def compute_laplacian_2d_gpu(field, dx, dz):
    """GPU版本的拉普拉斯算子计算"""
    xp = cp.get_array_module(field)
    nx, nz = field.shape
    laplacian = xp.zeros_like(field)
    
    # 内部点（完全向量化）
    laplacian[1:-1, 1:-1] = (
        (field[2:, 1:-1] - 2*field[1:-1, 1:-1] + field[:-2, 1:-1]) / (dx**2) +
        (field[1:-1, 2:] - 2*field[1:-1, 1:-1] + field[1:-1, :-2]) / (dz**2)
    )
    
    # 边界处理
    laplacian[0, 1:-1] = (
        (field[1, 1:-1] - field[0, 1:-1]) / (dx**2) +
        (field[0, 2:] - 2*field[0, 1:-1] + field[0, :-2]) / (dz**2)
    )
    laplacian[-1, 1:-1] = (
        (field[-1, 1:-1] - field[-2, 1:-1]) / (dx**2) +
        (field[-1, 2:] - 2*field[-1, 1:-1] + field[-1, :-2]) / (dz**2)
    )
    laplacian[1:-1, 0] = (
        (field[2:, 0] - 2*field[1:-1, 0] + field[:-2, 0]) / (dx**2) +
        (field[1:-1, 1] - field[1:-1, 0]) / (dz**2)
    )
    laplacian[1:-1, -1] = (
        (field[2:, -1] - 2*field[1:-1, -1] + field[:-2, -1]) / (dx**2) +
        (field[1:-1, -1] - field[1:-1, -2]) / (dz**2)
    )
    
    return laplacian


def compute_tikhonov_gradient_gpu(field, dx, dz, alpha_tikhonov):
    """GPU版本的Tikhonov梯度计算"""
    xp = cp.get_array_module(field)
    laplacian = compute_laplacian_2d_gpu(field, dx, dz)
    tikhonov_grad = compute_laplacian_2d_gpu(laplacian, dx, dz)
    
    max_abs = xp.max(xp.abs(tikhonov_grad))
    if max_abs > 0:
        tikhonov_grad = tikhonov_grad / max_abs
    
    return alpha_tikhonov * tikhonov_grad


def compute_gradient_gpu(epsilon, sigma, residual, source_list, receiver_list, 
                         dt, dx, dz, npml, freq, steps, 
                         sigma_required_gradient=False, wavefield_data=None, 
                         device_id=0):
    """
    GPU版本的梯度计算 - 完全在GPU上运行
    
    统一：
    - 输入可以是numpy或cupy数组
    - 内部计算全部使用cupy（GPU）
    - reverse_time_loop_gpu在GPU上运行
    - 返回numpy数组（保持接口兼容）
    """
    if not GPU_AVAILABLE:
        from .gradient import compute_gradient
        return compute_gradient(epsilon, sigma, residual, source_list, receiver_list, 
                               dt, dx, dz, npml, freq, steps, sigma_required_gradient, wavefield_data)
    
    with cp.cuda.Device(device_id):
        # 转换输入到GPU（如果还没在GPU上）
        xp = cp
        if isinstance(epsilon, np.ndarray):
            epsilon_gpu = cp.asarray(epsilon, dtype=cp.float32)
            sigma_gpu = cp.asarray(sigma, dtype=cp.float32)
            residual_gpu = cp.asarray(residual, dtype=cp.float32)
        else:
            epsilon_gpu = epsilon
            sigma_gpu = sigma
            residual_gpu = residual
        
        xl, zl = epsilon_gpu.shape
        n_shots = len(source_list)
        
        # 检查wavefield_data
        if wavefield_data is None or len(wavefield_data) == 0:
            print(f"[警告gradient_gpu] wavefield_data为None或空，返回零梯度")
            grad_eps = cp.zeros_like(epsilon_gpu)
            grad_sig = cp.zeros_like(sigma_gpu) if sigma_required_gradient else None
            return cp.asnumpy(grad_eps), cp.asnumpy(grad_sig) if grad_sig is not None else None
        
        # 初始化梯度（GPU上）
        grad_eps = cp.zeros_like(epsilon_gpu)
        grad_sig = cp.zeros_like(sigma_gpu) if sigma_required_gradient else None
        ep0 = 8.841941282883074e-12
        
        for shot_idx, (source_pos, receiver_pos) in enumerate(zip(source_list, receiver_list)):
            # 扩展模型（GPU上）
            epsilon_extended = cp.ones((xl + 2*npml, zl + 2*npml), dtype=cp.float32)
            sigma_extended = cp.ones((xl + 2*npml, zl + 2*npml), dtype=cp.float32) * 1e-4
            mu_extended = cp.ones((xl + 2*npml, zl + 2*npml), dtype=cp.float32)
            
            epsilon_extended[npml:npml+xl, npml:npml+zl] = epsilon_gpu
            sigma_extended[npml:npml+xl, npml:npml+zl] = sigma_gpu
            
            # 边界扩展
            epsilon_extended[:npml, :] = epsilon_extended[npml, :]
            epsilon_extended[-npml:, :] = epsilon_extended[-npml-1, :]
            epsilon_extended[:, :npml] = epsilon_extended[:, npml].reshape(-1, 1)
            epsilon_extended[:, -npml:] = epsilon_extended[:, -npml-1].reshape(-1, 1)
            sigma_extended[:npml, :] = sigma_extended[npml, :]
            sigma_extended[-npml:, :] = sigma_extended[-npml-1, :]
            sigma_extended[:, :npml] = sigma_extended[:, npml].reshape(-1, 1)
            sigma_extended[:, -npml:] = sigma_extended[:, -npml-1].reshape(-1, 1)
            
            # CPML参数（需要numpy）
            cpml_params = Add_CPML(xl, zl, cp.asnumpy(sigma_extended.copy()), 
                                  cp.asnumpy(epsilon_extended.copy()), 
                                  cp.asnumpy(mu_extended.copy()), dx, dz, dt)
            
            receiver_pos_extended = (receiver_pos[0] + npml, receiver_pos[1] + npml)
            
            # 调试：残差范围（该shot）
            try:
                res_min = float(cp.min(residual_gpu[shot_idx]).get())
                res_max = float(cp.max(residual_gpu[shot_idx]).get())
                print(f"[调试 gradient_gpu] Shot {shot_idx}: residual范围=[{res_min:.3e}, {res_max:.3e}]")
            except Exception:
                pass

            # ✓ 使用GPU版本的reverse_time_loop
            reverse_loop_gen = reverse_time_loop_gpu(
                xl, zl, dx, dz, dt,
                sigma_extended,  # 直接传cupy数组
                epsilon_extended,
                mu_extended,
                cpml_params,
                steps,
                receiver_pos_extended,
                residual_gpu[shot_idx, :],  # cupy数组
                device_id=device_id
            )
            
            if wavefield_data is not None and shot_idx < len(wavefield_data):
                if len(wavefield_data[shot_idx]) > 0:
                    # forward_wavefield 已经是降采样的列表（长度约 steps/WAVEFIELD_SUBSAMPLE）
                    if isinstance(wavefield_data[shot_idx], list):
                        fwd_list = [cp.asarray(wf) for wf in wavefield_data[shot_idx]]
                    else:
                        # 万一传入的是ndarray，转为list以便与adjoint对齐
                        fw = cp.asarray(wavefield_data[shot_idx])
                        fwd_list = [fw[i] for i in range(fw.shape[0])]

                    # 伴随波场：流式降采样收集，避免存满3500步
                    adj_sub_list = []
                    adj_min = None
                    adj_max = None
                    for tt, adj in enumerate(reverse_loop_gen):
                        if tt % WAVEFIELD_SUBSAMPLE == 0:
                            adj_gpu = cp.asarray(adj)
                            adj_sub_list.append(adj_gpu)
                            # 动态更新伴随波场范围
                            cur_min = float(cp.min(adj_gpu).get())
                            cur_max = float(cp.max(adj_gpu).get())
                            adj_min = cur_min if adj_min is None else min(adj_min, cur_min)
                            adj_max = cur_max if adj_max is None else max(adj_max, cur_max)
                    # 时间反向对齐（与CPU实现一致）
                    adj_sub_list = adj_sub_list[::-1]

                    n_time_subsampled = len(fwd_list)
                    if len(adj_sub_list) != n_time_subsampled:
                        # 对齐长度（取最短）
                        n_time_subsampled = min(n_time_subsampled, len(adj_sub_list))
                        fwd_list = fwd_list[:n_time_subsampled]
                        adj_sub_list = adj_sub_list[:n_time_subsampled]

                    # 前向波场范围（降采样列表）
                    # 避免一次性stack占用显存，逐步统计范围
                    fwd_min, fwd_max = None, None
                    for wf in fwd_list:
                        vmin = float(cp.min(wf).get())
                        vmax = float(cp.max(wf).get())
                        fwd_min = vmin if fwd_min is None else min(fwd_min, vmin)
                        fwd_max = vmax if fwd_max is None else max(fwd_max, vmax)

                    # 调试：打印该shot的波场范围（仅一次）
                    print(f"[调试 gradient_gpu] Shot {shot_idx}: forward范围=[{fwd_min:.3e}, {fwd_max:.3e}], adjoint范围=[{(adj_min if adj_min is not None else 0):.3e}, {(adj_max if adj_max is not None else 0):.3e}]")

                    dt_subsampled = dt * WAVEFIELD_SUBSAMPLE
                    
                    # 内存优化：使用增量计算，避免创建完整的时间序列数组
                    if len(fwd_list) >= 3:  # 确保有足够的数据点进行中心差分
                        
                        # 调试：检查前向波场范围（第一个和最后一个时间步）
                        if shot_idx == 0:
                            try:
                                fwd_min = float(cp.min(fwd_list[0]).get()); fwd_max = float(cp.max(fwd_list[0]).get())
                                print(f"[调试 gradient_gpu] Shot {shot_idx}: 前向波场范围=[{fwd_min:.3e}, {fwd_max:.3e}]")
                            except Exception:
                                pass
                        
                        # 内存优化：逐个时间步计算，避免cp.stack占用大量显存
                        for k in range(1, len(fwd_list)-1):
                            # 使用中心差分公式：∂E/∂t ≈ (E_{k+1} - E_{k-1}) / (2*dt_sub)
                            fwd_prev = fwd_list[k-1]
                            fwd_next = fwd_list[k+1]
                            
                            # 计算当前时间步的时间导数（只在有效区域）
                            fwd_diff = (fwd_next - fwd_prev) / (2 * dt_subsampled)
                            adjoint_field = adj_sub_list[k]
                            
                            # 裁剪到有效区域（避免PML边界）
                            crop_adj = adjoint_field[npml:npml+xl, npml:npml+zl]
                            crop_dif = fwd_diff[npml:npml+xl, npml:npml+zl]
                            
                            # 调试：首个时间步打印关键范围
                            if k == 1 and shot_idx == 0:
                                try:
                                    ca_min = float(cp.min(crop_adj).get()); ca_max = float(cp.max(crop_adj).get())
                                    cd_min = float(cp.min(crop_dif).get()); cd_max = float(cp.max(crop_dif).get())
                                    print(f"[调试 gradient_gpu] Shot {shot_idx}: crop adj范围=[{ca_min:.3e}, {ca_max:.3e}], crop diff范围=[{cd_min:.3e}, {cd_max:.3e}], dt_sub={dt_subsampled:.3e}")
                                except Exception:
                                    pass
                            
                            # 若中央差分为全零，回退为一阶差分
                            if cp.max(cp.abs(crop_dif)) == 0:
                                fwd_diff_fallback = (fwd_list[k] - fwd_list[k-1]) / dt_subsampled
                                crop_dif = fwd_diff_fallback[npml:npml+xl, npml:npml+zl]
                                if k == 1 and shot_idx == 0:
                                    try:
                                        cd_min2 = float(cp.min(crop_dif).get()); cd_max2 = float(cp.max(crop_dif).get())
                                        print(f"[调试 gradient_gpu] Shot {shot_idx}: 回退一阶差分 crop diff范围=[{cd_min2:.3e}, {cd_max2:.3e}]")
                                    except Exception:
                                        pass
                            
                            # 累积梯度贡献
                            grad_eps += (ep0 * crop_adj * crop_dif)
                            
                            if sigma_required_gradient:
                                # 对于电导率梯度，使用当前时间步的波场
                                grad_sig += (adjoint_field[npml:npml+xl, npml:npml+zl] *
                                             fwd_list[k][npml:npml+xl, npml:npml+zl])
                        
                        # 显存清理：及时释放不再使用的波场数据
                        if k % 10 == 0:  # 每10个时间步清理一次
                            cp.get_default_memory_pool().free_all_blocks()
                    else:
                        print(f"[警告 gradient_gpu] Shot {shot_idx}: 波场数据不足，无法计算梯度 (需要至少3个时间点)")
        
        # 应用mask（GPU上）
        mask = cp.ones_like(grad_eps)
        mask[:5, :] = 0
        grad_eps *= mask
        if sigma_required_gradient:
            grad_sig *= mask
        
        # 调试：梯度范围（mask前/后）
        try:
            gmin = float(cp.min(grad_eps).get()); gmax = float(cp.max(grad_eps).get())
            print(f"[调试 gradient_gpu] grad_eps范围(应用mask后)=[{gmin:.3e}, {gmax:.3e}]")
        except Exception:
            pass

        # 转回numpy（返回接口兼容）
        grad_eps_np = cp.asnumpy(grad_eps)
        grad_sig_np = cp.asnumpy(grad_sig) if sigma_required_gradient else None
        
        return grad_eps_np, grad_sig_np


def compute_gradient(epsilon, sigma, residual, source_list, receiver_list, 
                    dt, dx, dz, npml, freq, steps, 
                    sigma_required_gradient=False, wavefield_data=None, device_id=0):
    """
    统一入口：自动选择GPU或CPU版本
    """
    if GPU_AVAILABLE:
        return compute_gradient_gpu(epsilon, sigma, residual, source_list, receiver_list,
                                   dt, dx, dz, npml, freq, steps, sigma_required_gradient, 
                                   wavefield_data, device_id)
    else:
        from .gradient import compute_gradient as compute_gradient_cpu
        return compute_gradient_cpu(epsilon, sigma, residual, source_list, receiver_list,
                                   dt, dx, dz, npml, freq, steps, sigma_required_gradient, wavefield_data)


def compute_laplacian_2d(field, dx, dz):
    """兼容接口：自动检测numpy或cupy"""
    if GPU_AVAILABLE and isinstance(field, cp.ndarray):
        return compute_laplacian_2d_gpu(field, dx, dz)
    else:
        # CPU版本
        nx, nz = field.shape
        laplacian = np.zeros_like(field)
        laplacian[1:-1, 1:-1] = (
            (field[2:, 1:-1] - 2*field[1:-1, 1:-1] + field[:-2, 1:-1]) / (dx**2) +
            (field[1:-1, 2:] - 2*field[1:-1, 1:-1] + field[1:-1, :-2]) / (dz**2)
        )
        return laplacian


def compute_tikhonov_gradient(field, dx, dz, alpha_tikhonov):
    """兼容接口：自动检测numpy或cupy"""
    if GPU_AVAILABLE and isinstance(field, cp.ndarray):
        return compute_tikhonov_gradient_gpu(field, dx, dz, alpha_tikhonov)
    else:
        # CPU版本
        from .gradient import compute_tikhonov_gradient as compute_tikhonov_gradient_cpu
        return compute_tikhonov_gradient_cpu(field, dx, dz, alpha_tikhonov)

