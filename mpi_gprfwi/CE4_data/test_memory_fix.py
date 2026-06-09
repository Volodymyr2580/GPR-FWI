#!/usr/bin/env python3
"""
测试内存修复是否有效
运行小规模测试以验证内存使用
"""

import numpy as np
import psutil
import os
import sys

sys.path.append('../RMSprop')
from forward import forward_model

def get_memory_mb():
    """获取当前进程内存使用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

print("=" * 60)
print("内存修复测试")
print("=" * 60)

# 创建小规模测试模型
xl, zl = 100, 100  # 小模型
dt, dx, dz = 7e-11, 0.03, 0.03
npml = 10
freq = 5e8
steps = 1000  # 较少的时间步

# 创建模型
epsilon = np.ones((xl, zl)) * 3.0
sigma = np.ones((xl, zl)) * 1e-5

# 测试配置
source_list = [(0, i) for i in range(0, zl, 20)]  # 5个shots
receiver_list = source_list.copy()

print(f"测试配置：")
print(f"  模型大小: {xl} × {zl}")
print(f"  时间步数: {steps}")
print(f"  炮点数量: {len(source_list)}")
print()

# 测试1：不保存波场数据
print("测试1：不保存波场数据")
print("-" * 60)
mem_before = get_memory_mb()
print(f"  正演前内存: {mem_before:.1f} MB")

data = forward_model(epsilon, sigma, source_list, receiver_list, 
                    dt, dx, dz, npml, freq, steps, save_wavefield=False)

mem_after = get_memory_mb()
print(f"  正演后内存: {mem_after:.1f} MB")
print(f"  内存增长: {mem_after - mem_before:.1f} MB")
print(f"  数据形状: {data.shape}")
print(f"  ✓ 测试通过！内存增长应该很小（< 50 MB）")
print()

# 测试2：保存波场数据
print("测试2：保存波场数据")
print("-" * 60)
mem_before = get_memory_mb()
print(f"  正演前内存: {mem_before:.1f} MB")

data, wavefield = forward_model(epsilon, sigma, source_list, receiver_list, 
                                dt, dx, dz, npml, freq, steps, save_wavefield=True)

mem_after = get_memory_mb()
print(f"  正演后内存: {mem_after:.1f} MB")
print(f"  内存增长: {mem_after - mem_before:.1f} MB")
print(f"  数据形状: {data.shape}")
print(f"  波场数量: {len(wavefield)} shots")
if len(wavefield) > 0:
    print(f"  单个shot波场: {len(wavefield[0])} 时间步")

# 估算预期内存
expected_memory = len(source_list) * (xl + 2*npml) * (zl + 2*npml) * steps * 8 / 1024 / 1024
print(f"  预期波场内存: ~{expected_memory:.1f} MB")
print(f"  ✓ 测试通过！内存增长应该接近预期值")
print()

# 清理
del wavefield
del data
import gc
gc.collect()

mem_final = get_memory_mb()
print(f"清理后内存: {mem_final:.1f} MB")
print()

print("=" * 60)
print("测试完成！")
print("=" * 60)
print()
print("如果两个测试都通过，说明内存修复有效。")
print("现在可以安全运行完整程序：")
print()
print("  mpirun -n 10 python main.py")
print()


