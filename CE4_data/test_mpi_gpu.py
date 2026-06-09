#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPI多GPU配置测试脚本

使用方法:
    # 单进程测试
    python test_mpi_gpu.py
    
    # 多进程测试
    mpirun -np 4 python test_mpi_gpu.py
"""

import numpy as np
import os
import sys

# MPI初始化
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    MPI_AVAILABLE = True
except ImportError:
    rank, size = 0, 1
    MPI_AVAILABLE = False
    print("⚠ MPI不可用")

# 测试PyTorch CUDA
print(f"\n[Rank {rank}] ========== PyTorch CUDA测试 ==========")
try:
    import torch
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        process_gpu_id = rank % n_gpus
        
        print(f"[Rank {rank}] ✓ PyTorch CUDA可用")
        print(f"[Rank {rank}]   检测到 {n_gpus} 个GPU")
        print(f"[Rank {rank}]   分配GPU: {process_gpu_id}")
        
        # 设置并测试GPU
        torch.cuda.set_device(process_gpu_id)
        device = torch.device(f'cuda:{process_gpu_id}')
        
        # 简单计算测试
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        print(f"[Rank {rank}]   ✓ GPU {process_gpu_id} 计算测试通过")
        
        # 显示GPU信息
        props = torch.cuda.get_device_properties(process_gpu_id)
        mem_total = props.total_memory / 1024**3
        print(f"[Rank {rank}]   GPU名称: {props.name}")
        print(f"[Rank {rank}]   总显存: {mem_total:.1f}GB")
    else:
        print(f"[Rank {rank}] ✗ PyTorch CUDA不可用")
except ImportError:
    print(f"[Rank {rank}] ✗ PyTorch未安装")

# 测试CuPy
print(f"\n[Rank {rank}] ========== CuPy测试 ==========")
try:
    import cupy as cp
    
    n_devices = cp.cuda.runtime.getDeviceCount()
    process_gpu_id = rank % n_devices
    
    print(f"[Rank {rank}] ✓ CuPy已安装 (版本: {cp.__version__})")
    print(f"[Rank {rank}]   检测到 {n_devices} 个CUDA设备")
    print(f"[Rank {rank}]   分配GPU: {process_gpu_id}")
    
    # 设置并测试GPU
    with cp.cuda.Device(process_gpu_id):
        # 显示GPU内存
        mem_info = cp.cuda.Device().mem_info
        free_gb = mem_info[0] / 1024**3
        total_gb = mem_info[1] / 1024**3
        print(f"[Rank {rank}]   GPU内存: 空闲 {free_gb:.1f}GB / 总共 {total_gb:.1f}GB")
        
        # 简单计算测试
        a_gpu = cp.random.randn(1000, 1000, dtype=cp.float32)
        b_gpu = cp.random.randn(1000, 1000, dtype=cp.float32)
        c_gpu = cp.matmul(a_gpu, b_gpu)
        cp.cuda.Stream.null.synchronize()
        
        print(f"[Rank {rank}]   ✓ GPU {process_gpu_id} CuPy计算测试通过")
        
except ImportError as e:
    print(f"[Rank {rank}] ✗ CuPy未安装: {e}")
    print(f"[Rank {rank}]   请安装: pip install cupy-cuda11x 或 cupy-cuda12x")

# 测试FDTD正演（如果可用）
print(f"\n[Rank {rank}] ========== FDTD正演测试 ==========")
try:
    from utils.forward_gpu import forward_model_gpu
    from utils.Add_CPML import Add_CPML
    
    print(f"[Rank {rank}] ✓ GPU版forward模块可用")
    
    # 小规模测试
    xl, zl = 100, 200
    npml = 10
    dx, dz = 0.06, 0.06
    dt = 1.4e-10
    freq = 4e8
    steps = 100  # 少量步数快速测试
    
    # 创建测试模型
    epsilon = np.ones((xl, zl), dtype=np.float32) * 4.0
    sigma = np.ones((xl, zl), dtype=np.float32) * 1e-3
    
    # 单炮测试
    source_list = [(0, zl//2)]
    receiver_list = [(0, zl//2)]
    
    # 合成子波
    t = np.arange(steps) * dt
    wavelet = np.sin(2*np.pi*freq*t).astype(np.float32) * 0.1
    
    print(f"[Rank {rank}]   运行小规模正演测试...")
    print(f"[Rank {rank}]   网格: {xl}x{zl}, 步数: {steps}")
    
    import time
    start = time.time()
    
    # GPU正演
    data_gpu = forward_model_gpu(
        epsilon, sigma, source_list, receiver_list,
        dt, dx, dz, npml, freq, steps,
        save_wavefield=False,
        wavelet=wavelet,
        device_id=process_gpu_id,
        verbose=False
    )
    
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - start
    
    print(f"[Rank {rank}]   ✓ GPU正演测试通过")
    print(f"[Rank {rank}]   耗时: {gpu_time:.3f}秒")
    print(f"[Rank {rank}]   数据形状: {data_gpu.shape}")
    print(f"[Rank {rank}]   数据范围: [{data_gpu.min():.2e}, {data_gpu.max():.2e}]")
    
except ImportError as e:
    print(f"[Rank {rank}] ⚠ GPU正演模块不可用: {e}")
except Exception as e:
    print(f"[Rank {rank}] ✗ GPU正演测试失败: {e}")

# 同步所有进程
if MPI_AVAILABLE:
    comm.Barrier()

# 汇总报告（仅rank 0输出）
if rank == 0:
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    if MPI_AVAILABLE:
        print(f"✓ MPI可用，进程数: {size}")
    else:
        print(f"⚠ MPI不可用（单进程模式）")
    
    # 检查必要组件
    components = {
        'PyTorch': False,
        'PyTorch CUDA': False,
        'CuPy': False,
        'GPU正演': False
    }
    
    try:
        import torch
        components['PyTorch'] = True
        if torch.cuda.is_available():
            components['PyTorch CUDA'] = True
    except:
        pass
    
    try:
        import cupy as cp
        components['CuPy'] = True
    except:
        pass
    
    try:
        from utils.forward_gpu import forward_model_gpu
        components['GPU正演'] = True
    except:
        pass
    
    print("\n组件状态:")
    for name, available in components.items():
        status = "✓" if available else "✗"
        print(f"  {status} {name}")
    
    print("\n建议:")
    if not components['CuPy']:
        print("  ⚠ 请安装CuPy以启用GPU加速:")
        print("    pip install cupy-cuda11x  # 或 cupy-cuda12x")
    
    if components['CuPy'] and components['GPU正演']:
        print("  ✓ 所有组件就绪，可以运行GPU加速训练！")
        print("\n运行命令:")
        if MPI_AVAILABLE:
            print(f"    mpirun -np {size} python main.py --mode generate")
            print(f"    mpirun -np {size} python main.py --mode inversion")
        else:
            print("    python main.py --mode generate")
            print("    python main.py --mode inversion")
    
    print("="*70)

print(f"\n[Rank {rank}] 测试完成！")

