#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速迁移到GPU加速版本的辅助脚本

使用方法:
    python migrate_to_gpu.py --check  # 检查GPU环境
    python migrate_to_gpu.py --test   # 测试GPU性能
    python migrate_to_gpu.py --benchmark  # 完整性能对比
"""

import argparse
import sys
import os

def check_gpu_environment():
    """检查GPU环境是否正确配置"""
    print("="*70)
    print("GPU环境检查")
    print("="*70)
    
    # 1. 检查CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            n_gpus = torch.cuda.device_count()
            print(f"✓ PyTorch CUDA可用")
            print(f"  GPU数量: {n_gpus}")
            for i in range(n_gpus):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}, 显存: {props.total_memory/1024**3:.1f}GB")
        else:
            print("✗ PyTorch CUDA不可用")
    except ImportError:
        print("✗ PyTorch未安装")
    
    # 2. 检查CuPy
    try:
        import cupy as cp
        print(f"\n✓ CuPy已安装 (版本: {cp.__version__})")
        
        n_devices = cp.cuda.runtime.getDeviceCount()
        print(f"  检测到 {n_devices} 个CUDA设备")
        
        for i in range(n_devices):
            with cp.cuda.Device(i):
                mem_info = cp.cuda.Device().mem_info
                free_gb = mem_info[0] / 1024**3
                total_gb = mem_info[1] / 1024**3
                print(f"  GPU {i}: 空闲 {free_gb:.1f}GB / 总共 {total_gb:.1f}GB")
        
        # 简单测试
        print("\n测试CuPy基本功能...")
        a = cp.random.randn(1000, 1000, dtype=cp.float32)
        b = cp.random.randn(1000, 1000, dtype=cp.float32)
        c = cp.matmul(a, b)
        cp.cuda.Stream.null.synchronize()
        print("✓ CuPy测试通过")
        
    except ImportError:
        print("\n✗ CuPy未安装")
        print("  安装命令:")
        print("    pip install cupy-cuda11x  # CUDA 11.x")
        print("    pip install cupy-cuda12x  # CUDA 12.x")
        return False
    except Exception as e:
        print(f"\n✗ CuPy测试失败: {e}")
        return False
    
    # 3. 检查MPI
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        print(f"\n✓ MPI可用 (进程数: {comm.Get_size()})")
    except ImportError:
        print("\n⚠ MPI不可用 (可选)")
    
    print("\n" + "="*70)
    print("环境检查完成！")
    print("="*70)
    return True


def test_gpu_speed():
    """测试GPU加速效果"""
    import numpy as np
    import time
    
    print("\n" + "="*70)
    print("GPU加速性能测试")
    print("="*70)
    
    try:
        import cupy as cp
    except ImportError:
        print("✗ 需要安装CuPy才能进行GPU测试")
        return
    
    # 测试1: 简单矩阵运算
    print("\n测试1: 矩阵乘法 (5000x5000)")
    size = 5000
    
    # CPU测试
    print("  CPU测试中...")
    a_cpu = np.random.randn(size, size).astype(np.float32)
    b_cpu = np.random.randn(size, size).astype(np.float32)
    
    start = time.time()
    c_cpu = np.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start
    print(f"  CPU耗时: {cpu_time:.3f}秒")
    
    # GPU测试
    print("  GPU测试中...")
    a_gpu = cp.asarray(a_cpu)
    b_gpu = cp.asarray(b_cpu)
    
    # 预热
    _ = cp.matmul(a_gpu, b_gpu)
    cp.cuda.Stream.null.synchronize()
    
    start = time.time()
    c_gpu = cp.matmul(a_gpu, b_gpu)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - start
    print(f"  GPU耗时: {gpu_time:.3f}秒")
    print(f"  加速比: {cpu_time/gpu_time:.1f}x")
    
    # 测试2: FDTD相关的操作
    print("\n测试2: FDTD类型计算 (410x800网格, 1000步)")
    xl, zl = 410, 800
    steps = 1000
    
    # CPU版本
    print("  CPU测试中...")
    field_cpu = np.zeros((xl, zl), dtype=np.float32)
    
    start = time.time()
    for _ in range(steps):
        # 模拟FDTD更新
        field_cpu[1:-1, 1:-1] += 0.5 * (
            field_cpu[2:, 1:-1] + field_cpu[:-2, 1:-1] +
            field_cpu[1:-1, 2:] + field_cpu[1:-1, :-2] -
            4 * field_cpu[1:-1, 1:-1]
        )
    cpu_time = time.time() - start
    print(f"  CPU耗时: {cpu_time:.3f}秒")
    
    # GPU版本
    print("  GPU测试中...")
    field_gpu = cp.zeros((xl, zl), dtype=cp.float32)
    
    # 预热
    for _ in range(10):
        field_gpu[1:-1, 1:-1] += 0.5 * (
            field_gpu[2:, 1:-1] + field_gpu[:-2, 1:-1] +
            field_gpu[1:-1, 2:] + field_gpu[1:-1, :-2] -
            4 * field_gpu[1:-1, 1:-1]
        )
    cp.cuda.Stream.null.synchronize()
    
    field_gpu = cp.zeros((xl, zl), dtype=cp.float32)
    start = time.time()
    for _ in range(steps):
        field_gpu[1:-1, 1:-1] += 0.5 * (
            field_gpu[2:, 1:-1] + field_gpu[:-2, 1:-1] +
            field_gpu[1:-1, 2:] + field_gpu[1:-1, :-2] -
            4 * field_gpu[1:-1, 1:-1]
        )
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - start
    print(f"  GPU耗时: {gpu_time:.3f}秒")
    print(f"  加速比: {cpu_time/gpu_time:.1f}x")
    
    print("\n" + "="*70)
    print("性能测试完成！")
    print("预期FDTD完整正演加速比: 15-50x")
    print("="*70)


def benchmark_forward():
    """完整的forward模型性能对比"""
    import numpy as np
    import time
    
    print("\n" + "="*70)
    print("Forward模型完整性能测试")
    print("="*70)
    
    # 检查是否能导入所需模块
    try:
        from utils.forward import forward_model
        from utils.Add_CPML import Add_CPML
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        print("  请确保在正确的目录下运行此脚本")
        return
    
    # 测试参数（小规模测试）
    xl, zl = 410, 800
    npml = 10
    dx, dz = 0.06, 0.06
    dt = 1.4e-10
    freq = 4e8
    steps = 500  # 使用较少的步数以加快测试
    n_shots = 5  # 测试5炮
    
    print(f"\n测试配置:")
    print(f"  网格: {xl}x{zl}")
    print(f"  时间步数: {steps}")
    print(f"  炮点数: {n_shots}")
    print(f"  PML层数: {npml}")
    
    # 准备模型
    epsilon = np.ones((xl, zl), dtype=np.float32) * 4.0
    sigma = np.ones((xl, zl), dtype=np.float32) * 1e-3
    
    # 准备炮点
    source_list = [(0, i*160) for i in range(n_shots)]
    receiver_list = source_list.copy()
    
    # 加载子波
    try:
        from utils.forward import load_filtered_wavelet
        wavelet = load_filtered_wavelet(expected_steps=steps, verbose=False)
    except:
        print("  使用合成子波")
        wavelet = np.sin(2*np.pi*freq*np.arange(steps)*dt).astype(np.float32) * 0.1
    
    # CPU版本测试
    print(f"\n[1/2] CPU版本测试...")
    start = time.time()
    data_cpu = forward_model(epsilon, sigma, source_list, receiver_list,
                            dt, dx, dz, npml, freq, steps,
                            save_wavefield=False, wavelet=wavelet)
    cpu_time = time.time() - start
    print(f"  完成! 耗时: {cpu_time:.2f}秒")
    print(f"  单炮平均: {cpu_time/n_shots:.2f}秒")
    
    # GPU版本测试（如果可用）
    try:
        import cupy as cp
        from utils.forward_gpu import forward_model_gpu
        
        print(f"\n[2/2] GPU版本测试...")
        
        # 预热
        _ = forward_model_gpu(epsilon, sigma, [source_list[0]], [receiver_list[0]],
                             dt, dx, dz, npml, freq, 100,
                             save_wavefield=False, wavelet=wavelet[:100],
                             device_id=0, verbose=False)
        
        # 正式测试
        start = time.time()
        data_gpu = forward_model_gpu(epsilon, sigma, source_list, receiver_list,
                                    dt, dx, dz, npml, freq, steps,
                                    save_wavefield=False, wavelet=wavelet,
                                    device_id=0, verbose=False)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.time() - start
        print(f"  完成! 耗时: {gpu_time:.2f}秒")
        print(f"  单炮平均: {gpu_time/n_shots:.2f}秒")
        
        print(f"\n性能对比:")
        print(f"  CPU总耗时: {cpu_time:.2f}秒")
        print(f"  GPU总耗时: {gpu_time:.2f}秒")
        print(f"  加速比: {cpu_time/gpu_time:.1f}x")
        
        # 估算完整训练时间
        print(f"\n完整训练时间估算 (2000 epochs, 每epoch 100炮):")
        full_cpu_time = (cpu_time / n_shots) * 100 * 2000
        full_gpu_time = (gpu_time / n_shots) * 100 * 2000
        print(f"  CPU: {full_cpu_time/3600:.1f}小时 ({full_cpu_time/3600/24:.1f}天)")
        print(f"  GPU: {full_gpu_time/3600:.1f}小时 ({full_gpu_time/3600/24:.1f}天)")
        
        # 验证结果一致性
        diff = np.abs(data_cpu - data_gpu).max()
        rel_diff = diff / (np.abs(data_cpu).max() + 1e-10)
        print(f"\n结果一致性检查:")
        print(f"  最大绝对误差: {diff:.6e}")
        print(f"  最大相对误差: {rel_diff:.6e}")
        if rel_diff < 1e-4:
            print("  ✓ GPU和CPU结果一致")
        else:
            print("  ⚠ GPU和CPU结果存在较大差异")
        
    except ImportError:
        print(f"\n[2/2] GPU版本跳过 (CuPy未安装)")
    except Exception as e:
        print(f"\n[2/2] GPU测试失败: {e}")
    
    print("\n" + "="*70)
    print("完整性能测试结束！")
    print("="*70)


def print_usage_guide():
    """打印使用指南"""
    print("\n" + "="*70)
    print("GPU加速使用指南")
    print("="*70)
    print("""
步骤1: 安装CuPy
    pip install cupy-cuda11x  # 或 cupy-cuda12x

步骤2: 检查环境
    python migrate_to_gpu.py --check

步骤3: 测试性能
    python migrate_to_gpu.py --test

步骤4: 修改main.py使用GPU
    方法A - 最简单的修改:
        在main.py开头添加:
            USE_GPU = True  # 设置为True启用GPU
        
        然后在forward_model调用前添加:
            if USE_GPU:
                from utils.forward_gpu import forward_model_gpu as forward_model
            else:
                from utils.forward import forward_model
    
    方法B - 使用多GPU:
        from utils.forward_gpu import forward_model_multigpu
        
        # 在正演时:
        data = forward_model_multigpu(
            epsilon, sigma, source_list, receiver_list,
            dt, dx, dz, npml, freq, steps,
            gpu_ids=[0, 1, 2, 3],  # 使用4张GPU
            verbose=True
        )

步骤5: 运行训练
    # 单GPU
    python main.py --mode inversion
    
    # 多GPU (需要修改main.py)
    python main.py --mode inversion

详细文档请查看: GPU加速使用指南.md
""")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='GPU加速迁移工具')
    parser.add_argument('--check', action='store_true', help='检查GPU环境')
    parser.add_argument('--test', action='store_true', help='测试GPU基本性能')
    parser.add_argument('--benchmark', action='store_true', help='完整forward模型性能对比')
    parser.add_argument('--guide', action='store_true', help='显示使用指南')
    
    args = parser.parse_args()
    
    # 如果没有参数，显示帮助
    if not any(vars(args).values()):
        parser.print_help()
        print_usage_guide()
        return
    
    if args.check:
        success = check_gpu_environment()
        if success:
            print("\n✓ GPU环境配置正确，可以开始使用GPU加速")
        else:
            print("\n✗ GPU环境配置有问题，请按照提示修复")
    
    if args.test:
        test_gpu_speed()
    
    if args.benchmark:
        benchmark_forward()
    
    if args.guide:
        print_usage_guide()


if __name__ == "__main__":
    main()

