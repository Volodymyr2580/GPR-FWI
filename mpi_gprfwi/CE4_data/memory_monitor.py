#!/usr/bin/env python3
"""
内存监控脚本
用于分析MPI GPR FWI程序的内存使用情况
"""

import psutil
import os
import time
import numpy as np

def get_system_memory_info():
    """获取系统内存信息"""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percent': memory.percent
    }

def get_process_memory_info(pid):
    """获取指定进程的内存信息"""
    try:
        process = psutil.Process(pid)
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / (1024**2),  # 物理内存
            'vms_mb': memory_info.vms / (1024**2),  # 虚拟内存
            'rss_gb': memory_info.rss / (1024**3),
            'vms_gb': memory_info.vms / (1024**3)
        }
    except psutil.NoSuchProcess:
        return None

def estimate_memory_requirements():
    """估算内存需求"""
    # 基于代码中的参数
    xl, zl = 1000, 1000  # 网格大小
    npml = 10  # PML层厚度
    steps = 8000  # 时间步数
    n_shots = 500  # 炮点数量
    
    # 扩展网格大小
    xl_ext = xl + 2 * npml
    zl_ext = zl + 2 * npml
    
    # 估算各种数组的内存需求
    model_size = xl * zl * 8  # 8 bytes per float64
    extended_model_size = xl_ext * zl_ext * 8
    wavefield_size = xl_ext * zl_ext * steps * 8  # 每个shot的波场数据
    shot_data_size = n_shots * steps * 8
    
    print("=== 内存需求估算 ===")
    print(f"模型大小: {xl} x {zl} = {xl*zl:,} 个元素")
    print(f"扩展模型大小: {xl_ext} x {zl_ext} = {xl_ext*zl_ext:,} 个元素")
    print(f"时间步数: {steps}")
    print(f"炮点数量: {n_shots}")
    print()
    
    print("=== 各组件内存需求 ===")
    print(f"基础模型数组 (2个): {model_size * 2 / (1024**3):.2f} GB")
    print(f"扩展模型数组 (3个): {extended_model_size * 3 / (1024**3):.2f} GB")
    print(f"单个shot波场数据: {wavefield_size / (1024**3):.2f} GB")
    print(f"所有shot数据: {shot_data_size / (1024**3):.2f} GB")
    
    # 估算总内存需求
    base_memory = model_size * 2 + extended_model_size * 3 + shot_data_size
    wavefield_memory = wavefield_size * n_shots
    
    print()
    print("=== 总内存需求估算 ===")
    print(f"不保存波场数据: {base_memory / (1024**3):.2f} GB")
    print(f"保存所有波场数据: {(base_memory + wavefield_memory) / (1024**3):.2f} GB")
    print(f"保存单个shot波场数据: {(base_memory + wavefield_size) / (1024**3):.2f} GB")

def monitor_memory_usage():
    """实时监控内存使用"""
    print("=== 实时内存监控 ===")
    print("按 Ctrl+C 停止监控")
    print()
    
    try:
        while True:
            # 系统内存
            sys_mem = get_system_memory_info()
            print(f"系统内存: {sys_mem['used_gb']:.1f}/{sys_mem['total_gb']:.1f} GB "
                  f"({sys_mem['percent']:.1f}%) - 可用: {sys_mem['available_gb']:.1f} GB")
            
            # 查找MPI进程
            mpi_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['cmdline'] and any('mpi' in arg.lower() or 'python' in arg.lower() for arg in proc.info['cmdline']):
                        if 'main.py' in ' '.join(proc.info['cmdline']):
                            mpi_processes.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if mpi_processes:
                total_process_memory = 0
                for pid in mpi_processes:
                    proc_mem = get_process_memory_info(pid)
                    if proc_mem:
                        print(f"  进程 {pid}: {proc_mem['rss_gb']:.2f} GB (物理), {proc_mem['vms_gb']:.2f} GB (虚拟)")
                        total_process_memory += proc_mem['rss_gb']
                print(f"  总进程内存: {total_process_memory:.2f} GB")
            else:
                print("  未发现运行中的MPI进程")
            
            print("-" * 50)
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n监控已停止")

def main():
    """主函数"""
    print("MPI GPR FWI 内存分析工具")
    print("=" * 50)
    
    # 显示系统信息
    sys_mem = get_system_memory_info()
    print(f"系统总内存: {sys_mem['total_gb']:.1f} GB")
    print(f"当前可用内存: {sys_mem['available_gb']:.1f} GB")
    print(f"内存使用率: {sys_mem['percent']:.1f}%")
    print()
    
    # 估算内存需求
    estimate_memory_requirements()
    print()
    
    # 检查是否有足够的可用内存
    estimated_base = 2.0  # GB
    estimated_wavefield = 8.0  # GB
    if sys_mem['available_gb'] < estimated_base:
        print(f"[警告] 可用内存 ({sys_mem['available_gb']:.1f} GB) 可能不足以运行基础计算")
    if sys_mem['available_gb'] < estimated_base + estimated_wavefield:
        print(f"[警告] 可用内存可能不足以保存波场数据，建议设置 save_wavefield=False")
    
    print()
    
    # 询问是否开始实时监控
    choice = input("是否开始实时内存监控？(y/n): ").lower().strip()
    if choice == 'y':
        monitor_memory_usage()

if __name__ == "__main__":
    main()
