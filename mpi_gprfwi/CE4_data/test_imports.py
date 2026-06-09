#!/usr/bin/env python3
"""
测试脚本：验证所有导入和函数定义是否正确
"""

import sys
import os

print("测试导入...")

# 测试基础导入
try:
    import numpy as np
    print("✓ numpy 导入成功")
except ImportError as e:
    print(f"✗ numpy 导入失败: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib 导入成功")
except ImportError as e:
    print(f"✗ matplotlib 导入失败: {e}")
    sys.exit(1)

try:
    import psutil
    print("✓ psutil 导入成功")
except ImportError as e:
    print(f"✗ psutil 导入失败: {e}")
    print("提示：运行 'pip install psutil' 来安装")
    sys.exit(1)

try:
    import gc
    print("✓ gc 导入成功")
except ImportError as e:
    print(f"✗ gc 导入失败: {e}")
    sys.exit(1)

# 测试MPI导入
try:
    from mpi4py import MPI
    print("✓ mpi4py 导入成功")
    print(f"  MPI 版本: {MPI.Get_version()}")
except ImportError as e:
    print(f"⚠ mpi4py 未安装（将使用串行模式）")

# 测试内存监控函数
print("\n测试内存监控函数...")
try:
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    memory_gb = memory_mb / 1024
    print(f"✓ 当前进程内存使用: {memory_mb:.1f} MB ({memory_gb:.2f} GB)")
except Exception as e:
    print(f"✗ 内存监控测试失败: {e}")
    sys.exit(1)

# 测试系统内存信息
print("\n测试系统内存信息...")
try:
    total_memory = psutil.virtual_memory()
    print(f"✓ 系统总内存: {total_memory.total/1024/1024/1024:.1f} GB")
    print(f"  可用内存: {total_memory.available/1024/1024/1024:.1f} GB")
    print(f"  使用率: {total_memory.percent:.1f}%")
except Exception as e:
    print(f"✗ 系统内存信息获取失败: {e}")
    sys.exit(1)

# 测试文件访问
print("\n测试数据文件...")
required_files = [
    'test_model.npy',
    'wavelet_resampled.npy',
    'data_ist.npy'
]

for file_path in required_files:
    if os.path.exists(file_path):
        print(f"✓ {file_path} 存在")
    else:
        print(f"✗ {file_path} 不存在")
        print(f"  请确保该文件在当前目录: {os.getcwd()}")

print("\n" + "="*60)
print("所有基础测试完成！")
print("="*60)
print("\n建议的运行命令:")
print("  - 串行运行: python main.py")
print("  - MPI运行 (2进程): mpirun -n 2 python main.py")
print("  - MPI运行 (4进程): mpirun -n 4 python main.py")
print("\n内存监控工具:")
print("  - 运行: python memory_monitor.py")


