#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU优化版本的Time_loop - 充分发挥GPU并行计算优势

主要优化:
1. 完全向量化，消除所有Python循环
2. 减少CPU-GPU数据传输（只在最后一次传输）
3. 批量处理，充分利用GPU并行性

性能对比：
- 旧版本：每个时间步都做CPU-GPU传输 + Python循环处理PML → 极慢
- 新版本：完全向量化 + 仅最后传输 → 预期10-30倍加速
"""
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("Warning: CuPy not available, falling back to CPU version")


def update_H_vectorized(xl, zl, dx, dz, dt, sigma, epsilon, mu, npml,
                        a_x, a_z, b_x, b_z, k_x, k_z,
                        Hz, Hx, Ey, memory_dEy_dx, memory_dEy_dz,
                        dEy_dx, dEy_dz):
    """
    完全向量化的磁场更新 - 无Python循环
    """
    x_len = xl + 2*npml
    z_len = zl + 2*npml
    
    # === Hz更新 (dEy/dx) ===
    # 计算导数（向量化，复用缓冲区）
    dEy_dx[1:x_len-1, 1:z_len-1] = Ey[2:x_len, 1:z_len-1]
    dEy_dx[1:x_len-1, 1:z_len-1] -= Ey[1:x_len-1, 1:z_len-1]
    dEy_dx[1:x_len-1, 1:z_len-1] /= dx
    
    # 内部区域（非PML）- 向量化
    Hz[npml:x_len-npml, 1:z_len-1] += (dEy_dx[npml:x_len-npml, 1:z_len-1] * dt / 
                                       mu[npml:x_len-npml, 1:z_len-1])
    
    # 左侧PML（向量化，使用广播）
    if npml > 0:
        i_left = slice(1, npml)
        j_range = slice(1, z_len-1)
        
        # 扩展a_x, b_x, k_x以匹配2D数组（广播）
        a_x_2d = a_x[1:npml, cp.newaxis]
        b_x_2d = b_x[1:npml, cp.newaxis]
        k_x_2d = k_x[1:npml, cp.newaxis]
        
        memory_dEy_dx[i_left, j_range] = (b_x_2d * memory_dEy_dx[i_left, j_range] + 
                                          a_x_2d * dEy_dx[i_left, j_range])
        value_dEy_dx = dEy_dx[i_left, j_range] / k_x_2d + memory_dEy_dx[i_left, j_range]
        Hz[i_left, j_range] += value_dEy_dx * dt / mu[i_left, j_range]
    
    # 右侧PML（向量化）
    if npml > 0:
        i_right = slice(x_len-npml, x_len-1)
        i_mem = slice(0, npml-1)
        j_range = slice(1, z_len-1)
        
        a_x_2d = a_x[x_len-npml:x_len-1, cp.newaxis]
        b_x_2d = b_x[x_len-npml:x_len-1, cp.newaxis]
        k_x_2d = k_x[x_len-npml:x_len-1, cp.newaxis]
        
        memory_dEy_dx[i_mem, j_range] = (b_x_2d * memory_dEy_dx[i_mem, j_range] + 
                                         a_x_2d * dEy_dx[i_right, j_range])
        value_dEy_dx = dEy_dx[i_right, j_range] / k_x_2d + memory_dEy_dx[i_mem, j_range]
        Hz[i_right, j_range] += value_dEy_dx * dt / mu[i_right, j_range]
    
    # === Hx更新 (dEy/dz) ===
    dEy_dz[1:x_len-1, 1:z_len-1] = Ey[1:x_len-1, 2:z_len]
    dEy_dz[1:x_len-1, 1:z_len-1] -= Ey[1:x_len-1, 1:z_len-1]
    dEy_dz[1:x_len-1, 1:z_len-1] /= dz
    
    # 内部区域（非PML）
    Hx[1:x_len-1, npml:z_len-npml] -= (dEy_dz[1:x_len-1, npml:z_len-npml] * dt / 
                                       mu[1:x_len-1, npml:z_len-npml])
    
    # 顶部PML（向量化）
    if npml > 0:
        i_range = slice(1, x_len-1)
        j_top = slice(1, npml)
        
        a_z_2d = a_z[cp.newaxis, 1:npml]
        b_z_2d = b_z[cp.newaxis, 1:npml]
        k_z_2d = k_z[cp.newaxis, 1:npml]
        
        memory_dEy_dz[i_range, j_top] = (b_z_2d * memory_dEy_dz[i_range, j_top] + 
                                         a_z_2d * dEy_dz[i_range, j_top])
        value_dEy_dz = dEy_dz[i_range, j_top] / k_z_2d + memory_dEy_dz[i_range, j_top]
        Hx[i_range, j_top] -= value_dEy_dz * dt / mu[i_range, j_top]
    
    # 底部PML（向量化）
    if npml > 0:
        i_range = slice(1, x_len-1)
        j_bottom = slice(z_len-npml, z_len-1)
        j_mem = slice(0, npml-1)
        
        a_z_2d = a_z[cp.newaxis, z_len-npml:z_len-1]
        b_z_2d = b_z[cp.newaxis, z_len-npml:z_len-1]
        k_z_2d = k_z[cp.newaxis, z_len-npml:z_len-1]
        
        memory_dEy_dz[i_range, j_mem] = (b_z_2d * memory_dEy_dz[i_range, j_mem] + 
                                         a_z_2d * dEy_dz[i_range, j_bottom])
        value_dEy_dz = dEy_dz[i_range, j_bottom] / k_z_2d + memory_dEy_dz[i_range, j_mem]
        Hx[i_range, j_bottom] -= value_dEy_dz * dt / mu[i_range, j_bottom]
    
    return Hz, Hx


def update_E_vectorized(xl, zl, dx, dz, dt, ca, cb, npml,
                        a_x, a_z, b_x, b_z, k_x, k_z,
                        Hz, Hx, Ey, memory_dHz_dx, memory_dHx_dz,
                        dHz_dx, dHx_dz):
    """
    完全向量化的电场更新 - 无Python循环
    """
    x_len = xl + 2*npml
    z_len = zl + 2*npml
    
    # 计算导数（向量化，复用缓冲区）
    dHz_dx[1:x_len-1, 1:z_len-1] = Hz[1:x_len-1, 1:z_len-1]
    dHz_dx[1:x_len-1, 1:z_len-1] -= Hz[0:x_len-2, 1:z_len-1]
    dHz_dx[1:x_len-1, 1:z_len-1] /= dx
    dHx_dz[1:x_len-1, 1:z_len-1] = Hx[1:x_len-1, 1:z_len-1]
    dHx_dz[1:x_len-1, 1:z_len-1] -= Hx[1:x_len-1, 0:z_len-2]
    dHx_dz[1:x_len-1, 1:z_len-1] /= dz
    
    # 内部区域（非PML）- 完全向量化
    i_inner = slice(npml, x_len-npml)
    j_inner = slice(npml, z_len-npml)
    Ey[i_inner, j_inner] = (ca[i_inner, j_inner] * Ey[i_inner, j_inner] + 
                            cb[i_inner, j_inner] * (dHz_dx[i_inner, j_inner] - 
                                                    dHx_dz[i_inner, j_inner]) * dt)
    
    if npml == 0:
        return Ey
    
    # PML区域 - 向量化处理
    # 左侧PML
    i_left = slice(1, npml)
    j_middle = slice(npml, z_len-npml)
    
    a_x_2d = a_x[1:npml, cp.newaxis]
    b_x_2d = b_x[1:npml, cp.newaxis]
    k_x_2d = k_x[1:npml, cp.newaxis]
    
    memory_dHz_dx[i_left, j_middle] = (b_x_2d * memory_dHz_dx[i_left, j_middle] + 
                                       a_x_2d * dHz_dx[i_left, j_middle])
    dHz_dx_pml = dHz_dx[i_left, j_middle] / k_x_2d + memory_dHz_dx[i_left, j_middle]
    Ey[i_left, j_middle] = (ca[i_left, j_middle] * Ey[i_left, j_middle] + 
                            cb[i_left, j_middle] * (dHz_dx_pml - dHx_dz[i_left, j_middle]) * dt)
    
    # 右侧PML
    i_right = slice(x_len-npml, x_len-1)
    i_mem = slice(0, npml-1)
    
    a_x_2d = a_x[x_len-npml:x_len-1, cp.newaxis]
    b_x_2d = b_x[x_len-npml:x_len-1, cp.newaxis]
    k_x_2d = k_x[x_len-npml:x_len-1, cp.newaxis]
    
    memory_dHz_dx[i_mem, j_middle] = (b_x_2d * memory_dHz_dx[i_mem, j_middle] + 
                                      a_x_2d * dHz_dx[i_right, j_middle])
    dHz_dx_pml = dHz_dx[i_right, j_middle] / k_x_2d + memory_dHz_dx[i_mem, j_middle]
    Ey[i_right, j_middle] = (ca[i_right, j_middle] * Ey[i_right, j_middle] + 
                             cb[i_right, j_middle] * (dHz_dx_pml - dHx_dz[i_right, j_middle]) * dt)
    
    # 顶部PML
    i_middle = slice(npml, x_len-npml)
    j_top = slice(1, npml)
    
    a_z_2d = a_z[cp.newaxis, 1:npml]
    b_z_2d = b_z[cp.newaxis, 1:npml]
    k_z_2d = k_z[cp.newaxis, 1:npml]
    
    memory_dHx_dz[i_middle, j_top] = (b_z_2d * memory_dHx_dz[i_middle, j_top] + 
                                      a_z_2d * dHx_dz[i_middle, j_top])
    dHx_dz_pml = dHx_dz[i_middle, j_top] / k_z_2d + memory_dHx_dz[i_middle, j_top]
    Ey[i_middle, j_top] = (ca[i_middle, j_top] * Ey[i_middle, j_top] + 
                           cb[i_middle, j_top] * (dHz_dx[i_middle, j_top] - dHx_dz_pml) * dt)
    
    # 底部PML
    j_bottom = slice(z_len-npml, z_len-1)
    j_mem = slice(0, npml-1)
    
    a_z_2d = a_z[cp.newaxis, z_len-npml:z_len-1]
    b_z_2d = b_z[cp.newaxis, z_len-npml:z_len-1]
    k_z_2d = k_z[cp.newaxis, z_len-npml:z_len-1]
    
    memory_dHx_dz[i_middle, j_mem] = (b_z_2d * memory_dHx_dz[i_middle, j_mem] + 
                                      a_z_2d * dHx_dz[i_middle, j_bottom])
    dHx_dz_pml = dHx_dz[i_middle, j_bottom] / k_z_2d + memory_dHx_dz[i_middle, j_mem]
    Ey[i_middle, j_bottom] = (ca[i_middle, j_bottom] * Ey[i_middle, j_bottom] + 
                              cb[i_middle, j_bottom] * (dHz_dx[i_middle, j_bottom] - dHx_dz_pml) * dt)
    
    # 四个角的PML区域（简化处理）
    if npml > 0:
        i_lt = slice(1, npml)
        j_lt = slice(1, npml)
        Ey[i_lt, j_lt] = (ca[i_lt, j_lt] * Ey[i_lt, j_lt] + 
                         cb[i_lt, j_lt] * (dHz_dx[i_lt, j_lt] - dHx_dz[i_lt, j_lt]) * dt)
    
    return Ey


def time_loop_gpu(xl, zl, dx, dz, dt, sigma, epsilon, mu, CPML_Params,
                 f, k_max, source_site, ref_pos, device_id=0):
    """
    高性能GPU版本的time_loop
    
    主要优化：
    1. 完全向量化，无Python循环
    2. 最小化CPU-GPU传输（只在最后传输）
    3. 在GPU上累积接收器数据
    4. 支持直接接收cupy数组（零拷贝）
    """
    if not GPU_AVAILABLE:
        from .Time_loop import time_loop
        return time_loop(xl, zl, dx, dz, dt, sigma, epsilon, mu, CPML_Params,
                        f, k_max, source_site, ref_pos)
    
    with cp.cuda.Device(device_id):
        # 智能转换：如果已经是cupy数组，直接使用；否则转换
        if isinstance(sigma, cp.ndarray):
            # ✓ 已在GPU上，零拷贝
            sigma_gpu = sigma.astype(cp.float32) if sigma.dtype != cp.float32 else sigma
            epsilon_gpu = epsilon.astype(cp.float32) if epsilon.dtype != cp.float32 else epsilon
            mu_gpu = mu.astype(cp.float32) if mu.dtype != cp.float32 else mu
        else:
            # 从CPU转到GPU
            sigma_gpu = cp.asarray(sigma, dtype=cp.float32)
            epsilon_gpu = cp.asarray(epsilon, dtype=cp.float32)
            mu_gpu = cp.asarray(mu, dtype=cp.float32)
        
        f_gpu = cp.asarray(f, dtype=cp.float32)
        
        # CPML参数转换
        a_x = cp.asarray(CPML_Params.a_x, dtype=cp.float32)
        b_x = cp.asarray(CPML_Params.b_x, dtype=cp.float32)
        k_x = cp.asarray(CPML_Params.k_x, dtype=cp.float32)
        a_z = cp.asarray(CPML_Params.a_z, dtype=cp.float32)
        b_z = cp.asarray(CPML_Params.b_z, dtype=cp.float32)
        k_z = cp.asarray(CPML_Params.k_z, dtype=cp.float32)
        a_x_half = cp.asarray(CPML_Params.a_x_half, dtype=cp.float32)
        b_x_half = cp.asarray(CPML_Params.b_x_half, dtype=cp.float32)
        k_x_half = cp.asarray(CPML_Params.k_x_half, dtype=cp.float32)
        a_z_half = cp.asarray(CPML_Params.a_z_half, dtype=cp.float32)
        b_z_half = cp.asarray(CPML_Params.b_z_half, dtype=cp.float32)
        k_z_half = cp.asarray(CPML_Params.k_z_half, dtype=cp.float32)
        ca = cp.asarray(CPML_Params.ca, dtype=cp.float32)
        cb = cp.asarray(CPML_Params.cb, dtype=cp.float32)
        
        ep0 = 8.841941282883074e-12
        mu0 = 1.2566370614359173e-06
        epsilon_gpu = epsilon_gpu * ep0
        mu_gpu = mu_gpu * mu0
        
        npml = (sigma_gpu.shape[0] - xl) // 2
        x_len = xl + 2*npml
        z_len = zl + 2*npml
        
        # 初始化场变量（在GPU上）
        Ey = cp.zeros((x_len, z_len), dtype=cp.float32)
        Hz = cp.zeros((x_len, z_len), dtype=cp.float32)
        Hx = cp.zeros((x_len, z_len), dtype=cp.float32)
        
        memory_dEy_dx = cp.zeros((2*npml, z_len), dtype=cp.float32)
        memory_dEy_dz = cp.zeros((x_len, 2*npml), dtype=cp.float32)
        memory_dHz_dx = cp.zeros((2*npml, z_len), dtype=cp.float32)
        memory_dHx_dz = cp.zeros((x_len, 2*npml), dtype=cp.float32)
        
        # 在GPU上预分配接收器数据数组（关键优化！）
        receiver_data = cp.zeros(k_max, dtype=cp.float32)
        
        # 时间循环 - 全部在GPU上（逐步产出当前帧，确保正演波场随时间变化被正确保存）
        dEy_dx_buf = cp.zeros_like(Hz)
        dEy_dz_buf = cp.zeros_like(Hx)
        for tt in range(k_max):
            # 1. 更新磁场（向量化）
            Hz, Hx = update_H_vectorized(xl, zl, dx, dz, dt, sigma_gpu, epsilon_gpu, mu_gpu, npml,
                                        a_x_half, a_z_half, b_x_half, b_z_half, k_x_half, k_z_half,
                                        Hz, Hx, Ey, memory_dEy_dx, memory_dEy_dz,
                                        dEy_dx_buf, dEy_dz_buf)
            
            # 2. 更新电场（向量化）
            # 预分配导数缓冲区
            dHz_dx_buf = cp.zeros_like(Ey)
            dHx_dz_buf = cp.zeros_like(Ey)
            Ey = update_E_vectorized(xl, zl, dx, dz, dt, ca, cb, npml,
                                    a_x, a_z, b_x, b_z, k_x, k_z,
                                    Hz, Hx, Ey, memory_dHz_dx, memory_dHx_dz,
                                    dHz_dx_buf, dHx_dz_buf)
            
            # 3. 添加震源
            Ey[source_site[0], source_site[1]] += (-cb[source_site[0], source_site[1]] * 
                                                    f_gpu[tt] * dt / dx / dz)
            
            # 4. 逐步传回当前帧（关键：必须复制当前Ey，否则所有帧相同）
            ey_cpu = cp.asnumpy(Ey.copy())
            rec_val = float(Ey[ref_pos[0], ref_pos[1]].get())
            yield ey_cpu, rec_val


def reverse_time_loop_gpu(xl, zl, dx, dz, dt, sigma, epsilon, mu, CPML_Params,
                          k_max, ref_pos, rhs_data, device_id=0):
    """GPU加速的反向时间循环（向量化版本）"""
    if not GPU_AVAILABLE:
        from .Time_loop import reverse_time_loop
        return reverse_time_loop(xl, zl, dx, dz, dt, sigma, epsilon, mu, CPML_Params,
                                k_max, ref_pos, rhs_data)
    
    with cp.cuda.Device(device_id):
        sigma_gpu = cp.asarray(sigma, dtype=cp.float32)
        epsilon_gpu = cp.asarray(epsilon, dtype=cp.float32)
        mu_gpu = cp.asarray(mu, dtype=cp.float32)
        rhs_data_gpu = cp.asarray(rhs_data, dtype=cp.float32)
        
        # CPML参数
        a_x = cp.asarray(CPML_Params.a_x, dtype=cp.float32)
        b_x = cp.asarray(CPML_Params.b_x, dtype=cp.float32)
        k_x = cp.asarray(CPML_Params.k_x, dtype=cp.float32)
        a_z = cp.asarray(CPML_Params.a_z, dtype=cp.float32)
        b_z = cp.asarray(CPML_Params.b_z, dtype=cp.float32)
        k_z = cp.asarray(CPML_Params.k_z, dtype=cp.float32)
        a_x_half = cp.asarray(CPML_Params.a_x_half, dtype=cp.float32)
        b_x_half = cp.asarray(CPML_Params.b_x_half, dtype=cp.float32)
        k_x_half = cp.asarray(CPML_Params.k_x_half, dtype=cp.float32)
        a_z_half = cp.asarray(CPML_Params.a_z_half, dtype=cp.float32)
        b_z_half = cp.asarray(CPML_Params.b_z_half, dtype=cp.float32)
        k_z_half = cp.asarray(CPML_Params.k_z_half, dtype=cp.float32)
        ca = cp.asarray(CPML_Params.ca_r, dtype=cp.float32)
        cb = cp.asarray(CPML_Params.cb, dtype=cp.float32)
        
        ep0 = 8.841941282883074e-12
        mu0 = 1.2566370614359173e-06
        epsilon_gpu = epsilon_gpu * ep0
        mu_gpu = mu_gpu * mu0
        
        npml = (sigma_gpu.shape[0] - xl) // 2
        x_len = xl + 2*npml
        z_len = zl + 2*npml
        
        Ey = cp.zeros((x_len, z_len), dtype=cp.float32)
        Hz = cp.zeros((x_len, z_len), dtype=cp.float32)
        Hx = cp.zeros((x_len, z_len), dtype=cp.float32)
        
        memory_dEy_dx = cp.zeros((2*npml, z_len), dtype=cp.float32)
        memory_dEy_dz = cp.zeros((x_len, 2*npml), dtype=cp.float32)
        memory_dHz_dx = cp.zeros((2*npml, z_len), dtype=cp.float32)
        memory_dHx_dz = cp.zeros((x_len, 2*npml), dtype=cp.float32)
        
        # 预分配导数缓冲区（减少每步分配）
        dEy_dx_buf = cp.zeros_like(Hz)
        dEy_dz_buf = cp.zeros_like(Hx)
        dHz_dx_buf = cp.zeros_like(Ey)
        dHx_dz_buf = cp.zeros_like(Ey)
        # 流式生成伴随波场（不一次性缓存全部）
        for tt in range(k_max):
            # 反向传播source term（与CPU版本一致：不乘cb与dt/dx/dz）
            Ey[ref_pos[0], ref_pos[1]] -= rhs_data_gpu[k_max-tt-1]
            
            # 更新磁场
            Hz, Hx = update_H_vectorized(xl, zl, dx, dz, dt, sigma_gpu, epsilon_gpu, mu_gpu, npml,
                                        a_x_half, a_z_half, b_x_half, b_z_half, k_x_half, k_z_half,
                                        Hz, Hx, Ey, memory_dEy_dx, memory_dEy_dz,
                                        dEy_dx_buf, dEy_dz_buf)
            
            # 更新电场
            Ey = update_E_vectorized(xl, zl, dx, dz, dt, ca, cb, npml,
                                    a_x, a_z, b_x, b_z, k_x, k_z,
                                    Hz, Hx, Ey, memory_dHz_dx, memory_dHx_dz,
                                    dHz_dx_buf, dHx_dz_buf)
            
            # 直接yield当前时间步（传输到CPU），由上层按需降采样
            yield cp.asnumpy(Ey)
