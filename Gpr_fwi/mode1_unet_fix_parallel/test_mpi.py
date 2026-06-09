#!/usr/bin/env python3
"""
MPI功能测试脚本
用于验证MPI并行计算是否正常工作
"""

import numpy as np
import sys
import os
from mpi4py import MPI

def test_mpi_basic():
    """测试基本MPI功能"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print(f"Process {rank}/{size}: Hello from MPI!")
    
    # 测试通信
    if rank == 0:
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        print(f"Process {rank}: Sending data: {data}")
    else:
        data = None
    
    # 广播数据
    data = comm.bcast(data, root=0)
    print(f"Process {rank}: Received data: {data}")
    
    # 测试收集
    local_data = np.array([rank + 1, rank + 2, rank + 3])
    all_data = comm.gather(local_data, root=0)
    
    if rank == 0:
        print(f"Process {rank}: Collected data: {all_data}")
    
    return True

def test_mpi_parallel():
    """测试并行计算功能"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # 模拟并行计算
    n_tasks = 20
    tasks_per_proc = n_tasks // size
    remainder = n_tasks % size
    
    # 分配任务
    if rank < remainder:
        start_idx = rank * (tasks_per_proc + 1)
        end_idx = start_idx + tasks_per_proc + 1
    else:
        start_idx = rank * tasks_per_proc + remainder
        end_idx = start_idx + tasks_per_proc
    
    local_tasks = list(range(start_idx, end_idx))
    print(f"Process {rank}: Processing tasks {local_tasks}")
    
    # 模拟计算
    local_result = sum(local_tasks)
    
    # 收集结果
    all_results = comm.gather(local_result, root=0)
    
    if rank == 0:
        total_result = sum(all_results)
        expected_result = sum(range(n_tasks))
        print(f"Process {rank}: Total result: {total_result}")
        print(f"Process {rank}: Expected result: {expected_result}")
        print(f"Process {rank}: Test {'PASSED' if total_result == expected_result else 'FAILED'}")
    
    return True

def test_mpi_array():
    """测试MPI数组操作"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # 创建本地数组
    local_array = np.ones((5, 5)) * (rank + 1)
    
    # 收集所有进程的数组
    all_arrays = comm.gather(local_array, root=0)
    
    if rank == 0:
        print(f"Process {rank}: Collected {len(all_arrays)} arrays")
        for i, arr in enumerate(all_arrays):
            print(f"Process {rank}: Array from process {i}: sum = {np.sum(arr)}")
    
    return True

def main():
    """主测试函数"""
    print("=" * 60)
    print("MPI功能测试开始")
    print("=" * 60)
    
    # 测试1: 基本MPI功能
    print("\n1. 测试基本MPI功能...")
    test_mpi_basic()
    
    # 测试2: 并行计算
    print("\n2. 测试并行计算功能...")
    test_mpi_parallel()
    
    # 测试3: 数组操作
    print("\n3. 测试MPI数组操作...")
    test_mpi_array()
    
    print("\n" + "=" * 60)
    print("MPI功能测试完成")
    print("=" * 60)
    
    # 同步所有进程
    comm = MPI.COMM_WORLD
    comm.Barrier()
    
    if comm.Get_rank() == 0:
        print("\n所有测试完成！如果看到所有进程的输出，说明MPI工作正常。")

if __name__ == "__main__":
    main()
