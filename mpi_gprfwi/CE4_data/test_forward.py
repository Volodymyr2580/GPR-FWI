import numpy as np
import matplotlib.pyplot as plt
from forward import forward_model
import os
import shutil
import time
import psutil
import gc

freq = 5e8

# 内存监控函数
def get_memory_usage():
    """获取当前进程的内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    memory_gb = memory_mb / 1024
    return memory_mb, memory_gb

def print_memory_status(location=""):
    """打印内存使用状态"""
    memory_mb, memory_gb = get_memory_usage()
    print(f"[内存监控 {location}] 进程内存: {memory_mb:.1f} MB ({memory_gb:.2f} GB)")
    
    total_memory = psutil.virtual_memory()
    print(f"[系统内存] 总内存: {total_memory.total/1024/1024/1024:.1f} GB, "
          f"可用: {total_memory.available/1024/1024/1024:.1f} GB, "
          f"使用率: {total_memory.percent:.1f}%")
    gc.collect()
    print("-" * 60)
def eps_to_sig(epsilon_r):
    ep0 = 8.841941282883074e-12
    epsilon=epsilon_r * ep0
    index = 0.440 * np.log(epsilon_r) / np.log(1.93) - 2.943
    sig = 2*np.pi*freq*epsilon*10**(index)
    return sig
# ==== MPI 初始化 ====
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except Exception:
    # 兼容无 MPI 环境（串行）
    class _DummyComm:
        def Barrier(self):
            pass
        def bcast(self, x, root=0):
            return x
    comm = _DummyComm()
    rank, size = 0, 1

# 创建results文件夹在当前目录下（仅在rank 0 执行）
results_dir = "test_forward_results"
if rank == 0:
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    print("=" * 70)
    print(" " * 20 + "正演测试程序")
    print("=" * 70)
    print(f"结果将保存到: {results_dir}")
    print(f"MPI 进程数: {size}")
    print_memory_status("程序开始")
comm.Barrier()

# ========== 1. 参数设置 ==========
# 网格参数
xl, zl = 1000, 1000 #纵向和横向网格数
k_max = 8000 #最大迭代次数
dx, dz = 0.03, 0.03 #网格间距
dt = 7e-11 #时间步长
npml = 10 #PML层厚度
steps = 8000 #时间步数

# 真实模型参数
test_model = np.load('test_model.npy')[:,4000:5000]
epsilon_true = test_model.copy()
sigma_true = eps_to_sig(epsilon_true)

# 画图保存
plt.figure()
plt.imshow(epsilon_true, aspect='auto', cmap='jet',vmin=0,vmax=6)
plt.title('Epsilon True')
plt.colorbar()
plt.savefig(f'{results_dir}/epsilon_true.png')
plt.close()

plt.figure()
plt.imshow(sigma_true, aspect='auto', cmap='jet')
plt.title('Sigma True')
plt.colorbar()
plt.savefig(f'{results_dir}/sigma_true.png')
plt.close()

# 源点和接收点设置 - 注意：这些坐标是相对于内部网格的，不包括PML边界
source_list = [(0, i) for i in range(0, zl, 2)]
receiver_list = source_list.copy()

if rank == 0:
    print("\n" + "=" * 70)
    print("参数设置")
    print("=" * 70)
    print(f"网格大小: {xl} × {zl}")
    print(f"网格间距: dx={dx} m, dz={dz} m")
    print(f"时间步长: dt={dt} s")
    print(f"时间步数: {steps}")
    print(f"PML厚度: {npml}")
    print(f"源点/接收点数量: {len(source_list)}")
    print(f"采样间隔: 每2个网格点")
    print(f"真实模型范围: epsilon=[{epsilon_true.min():.2f}, {epsilon_true.max():.2f}]")
    print(f"真实模型范围: sigma=[{sigma_true.min():.3e}, {sigma_true.max():.3e}]")
    
# ========== 2. 生成观测数据 ==========
if rank == 0:
    print("\n" + "=" * 70)
    print("开始正演模拟")
    print("=" * 70)
    print(f"正在计算 {len(source_list)} 个炮点的数据...")
    print(f"⚠️ 注意: 不保存波场数据以节省内存")
    print_memory_status("正演开始前")

# 前向模拟（forward_model 内部支持 MPI 并行且会在需要时广播数据）
start_time = time.time()
d_obs = forward_model(
    epsilon_true,
    sigma_true,
    source_list,
    receiver_list,
    dt,
    dx,
    dz,
    npml,
    freq,
    steps,
    save_wavefield=False,
)
elapsed = time.time() - start_time

comm.Barrier()
if rank == 0:
    print_memory_status("正演完成后")
    
    print("\n" + "=" * 70)
    print("正演结果检查")
    print("=" * 70)
    print(f"✓ 正演完成，用时: {elapsed:.2f} 秒")
    
    if d_obs is None:
        print("✗ [错误] d_obs 为 None！")
    else:
        print(f"✓ d_obs 类型: {d_obs.dtype}")
        print(f"✓ d_obs 形状: {d_obs.shape}")
        print(f"  预期形状: ({len(source_list)}, {steps})")
        
        has_nan = np.isnan(d_obs).any()
        has_inf = np.isinf(d_obs).any()
        
        if has_nan:
            print("✗ [警告] d_obs 包含 NaN 值！")
        else:
            print("✓ d_obs 不包含 NaN 值")
            
        if has_inf:
            print("✗ [警告] d_obs 包含 Inf 值！")
        else:
            print("✓ d_obs 不包含 Inf 值")
            
        try:
            vmin, vmax = float(np.nanmin(d_obs)), float(np.nanmax(d_obs))
            vmean = float(np.nanmean(d_obs))
            vstd = float(np.nanstd(d_obs))
            print(f"✓ d_obs 数值范围:")
            print(f"  最小值: {vmin:.6e}")
            print(f"  最大值: {vmax:.6e}")
            print(f"  平均值: {vmean:.6e}")
            print(f"  标准差: {vstd:.6e}")
        except Exception as e:
            print(f"✗ 无法计算 d_obs 统计信息: {e}")

    # 保存数据和可视化
    print("\n" + "=" * 70)
    print("保存结果")
    print("=" * 70)
    
    # 保存numpy数组
    np.save(f"{results_dir}/d_obs.npy", d_obs)
    print(f"✓ 观测数据已保存: {results_dir}/d_obs.npy")

    # 创建B-scan图像
    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 图1: 完整B-scan
        im1 = axes[0].imshow(d_obs.T, aspect='auto', cmap='seismic', 
                            vmin=-np.max(np.abs(d_obs)), vmax=np.max(np.abs(d_obs)))
        axes[0].set_title(f'B-scan: 完整观测数据 ({d_obs.shape[0]} traces × {d_obs.shape[1]} samples)', 
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Trace Number (炮点索引)', fontsize=10)
        axes[0].set_ylabel('Time Sample (时间采样点)', fontsize=10)
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('Amplitude', fontsize=10)
        
        # 图2: 单个trace示例
        trace_idx = len(source_list) // 2  # 中间的trace
        axes[1].plot(d_obs[trace_idx], linewidth=0.5)
        axes[1].set_title(f'单个Trace示例 (Trace #{trace_idx})', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Time Sample (时间采样点)', fontsize=10)
        axes[1].set_ylabel('Amplitude', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/obs_data_bscan.png', dpi=150, bbox_inches='tight')
        print(f"✓ B-scan图像已保存: {results_dir}/obs_data_bscan.png")
        plt.close()
        
        # 额外创建一个更详细的单trace图
        fig, ax = plt.subplots(figsize=(14, 6))
        time_axis = np.arange(steps) * dt * 1e9  # 转换为纳秒
        ax.plot(time_axis, d_obs[trace_idx], linewidth=1)
        ax.set_title(f'单Trace详细视图 (Trace #{trace_idx})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (ns)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/single_trace_detail.png', dpi=150, bbox_inches='tight')
        print(f"✓ 单trace详图已保存: {results_dir}/single_trace_detail.png")
        plt.close()
        
    except Exception as e:
        print(f"✗ 绘制/保存观测数据图失败: {e}")
        import traceback
        traceback.print_exc()

    # 记录运行信息
    try:
        with open(f"{results_dir}/run_info.txt", "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("正演测试运行报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("运行信息:\n")
            f.write("-" * 70 + "\n")
            f.write(f"正演用时: {elapsed:.2f} 秒\n")
            f.write(f"MPI进程数: {size}\n")
            f.write(f"结果保存目录: {results_dir}\n\n")
            
            f.write("模型参数:\n")
            f.write("-" * 70 + "\n")
            f.write(f"网格大小: xl={xl}, zl={zl}\n")
            f.write(f"网格间距: dx={dx} m, dz={dz} m\n")
            f.write(f"时间步长: dt={dt} s\n")
            f.write(f"时间步数: {steps}\n")
            f.write(f"PML层厚度: {npml}\n")
            f.write(f"源频率: {freq} Hz\n")
            f.write(f"炮点数量: {len(source_list)}\n\n")
            
            f.write("输出数据:\n")
            f.write("-" * 70 + "\n")
            if d_obs is None:
                f.write("错误: d_obs 为 None\n")
            else:
                f.write(f"数据类型: {d_obs.dtype}\n")
                f.write(f"数据形状: {d_obs.shape}\n")
                f.write(f"包含NaN: {bool(np.isnan(d_obs).any())}\n")
                f.write(f"包含Inf: {bool(np.isinf(d_obs).any())}\n")
                try:
                    f.write(f"最小值: {float(np.nanmin(d_obs)):.6e}\n")
                    f.write(f"最大值: {float(np.nanmax(d_obs)):.6e}\n")
                    f.write(f"平均值: {float(np.nanmean(d_obs)):.6e}\n")
                    f.write(f"标准差: {float(np.nanstd(d_obs)):.6e}\n")
                except Exception as e:
                    f.write(f"统计信息计算失败: {e}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("测试完成\n")
            f.write("=" * 70 + "\n")
            
        print(f"✓ 运行日志已保存: {results_dir}/run_info.txt")
    except Exception as e:
        print(f"✗ 保存运行日志失败: {e}")
    
    # 最终总结
    print("\n" + "=" * 70)
    print("测试完成总结")
    print("=" * 70)
    print(f"✓ 正演模拟成功完成")
    print(f"✓ 用时: {elapsed:.2f} 秒")
    print(f"✓ 生成数据: {d_obs.shape[0]} 个traces × {d_obs.shape[1]} 个时间采样点")
    print(f"✓ 所有结果已保存到: {results_dir}/")
    print("\n生成的文件:")
    print(f"  - epsilon_true.png        : 真实介电常数模型")
    print(f"  - sigma_true.png          : 真实电导率模型")
    print(f"  - d_obs.npy               : 观测数据（numpy数组）")
    print(f"  - obs_data_bscan.png      : B-scan图像（完整+单trace）")
    print(f"  - single_trace_detail.png : 单trace详细视图")
    print(f"  - run_info.txt            : 运行报告")
    print("=" * 70)
    
    # 内存使用最终报告
    print_memory_status("测试结束")




