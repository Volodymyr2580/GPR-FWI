import numpy as np
import matplotlib.pyplot as plt
from forward import forward_model
from gradient import compute_gradient, compute_tikhonov_gradient
import os
import shutil
import time
from tqdm import tqdm
import psutil
import gc

freq = 5e8

# ==== 工具函数定义 ====
def get_memory_usage():
    """获取当前进程的内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024  # 转换为MB
    memory_gb = memory_mb / 1024  # 转换为GB
    return memory_mb, memory_gb

def print_memory_status(location="", rank=0):
    """打印内存使用状态"""
    if rank == 0:
        memory_mb, memory_gb = get_memory_usage()
        print(f"[内存监控 {location}] 进程内存使用: {memory_mb:.1f} MB ({memory_gb:.2f} GB)")
        
        # 获取系统总内存信息
        total_memory = psutil.virtual_memory()
        print(f"[系统内存] 总内存: {total_memory.total/1024/1024/1024:.1f} GB, "
              f"可用: {total_memory.available/1024/1024/1024:.1f} GB, "
              f"使用率: {total_memory.percent:.1f}%")
        
        # 强制垃圾回收
        gc.collect()
        memory_mb_after, memory_gb_after = get_memory_usage()
        print(f"[内存监控 {location}] 垃圾回收后: {memory_mb_after:.1f} MB ({memory_gb_after:.2f} GB)")
        print("-" * 60)

def cleanup_memory():
    """清理内存，删除不需要的大对象"""
    global local_wavefield_data
    if 'local_wavefield_data' in globals():
        del local_wavefield_data
    gc.collect()

def check_nan_inf(arr, name):
    """检查数组中的nan和inf值"""
    if np.any(np.isnan(arr)):
        print(f"[警告] {name} 出现nan!")
    if np.any(np.isinf(arr)):
        print(f"[警告] {name} 出现inf!")
    print(f"[{name}] min={np.nanmin(arr)}, max={np.nanmax(arr)}, mean={np.nanmean(arr)}")

def should_save_epoch(epoch):
    """判断是否应该保存当前epoch的结果"""
    save_epochs = [0, 1, 2, 5, 10, 20, 50]
    return epoch in save_epochs or (epoch >= 50 and epoch % 50 == 0)

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

# 创建results文件夹在用户主目录下（仅在rank 0 执行）
results_dir = os.path.expanduser("~/swz/mpi_gprfwi/CE4_data/results")
if rank == 0:
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    print(f"结果将保存到: {results_dir}")
comm.Barrier()

# ========== 1. 参数设置 ==========
# 网格参数
xl, zl = 1000, 1000 #纵向和横向网格数
k_max = 8000 #最大迭代次数
dx, dz = 0.03, 0.03 #网格间距
dt = 7e-11 #时间步长
npml = 10 #PML层厚度
steps = 8000 #时间步数

# 初始模型参数
test_model = np.load('test_model.npy')
epsilon0 = test_model.copy()[:,4000:5000]
sigma0 = eps_to_sig(epsilon0)

# 画图保存
plt.figure()
plt.imshow(epsilon0, aspect='auto', cmap='jet',vmin=0,vmax=6)
plt.title('Epsilon Initial')
plt.colorbar()
plt.savefig(f'{results_dir}/epsilon_initial.png')
plt.close()

plt.figure()
plt.imshow(sigma0, aspect='auto', cmap='jet',vmin=0,vmax=1e-3)
plt.title('Sigma Initial')
plt.colorbar()
plt.savefig(f'{results_dir}/sigma_initial.png')
plt.close()


# 检查CFL条件
c = 3e8  # 光速
epsilon_max = epsilon0.max()
cfl_condition = dt < 1 / (c * np.sqrt(1/dx**2 + 1/dz**2) / np.sqrt(epsilon_max))
if rank == 0:
    print(f"MPI 进程数: {size}")
    print(f"CFL条件检查: {cfl_condition}")
    print(f"当前dt: {dt}, 最大允许dt: {1 / (c * np.sqrt(1/dx**2 + 1/dz**2) / np.sqrt(epsilon_max))}")
    
    # 内存警告
    total_memory = psutil.virtual_memory()
    if total_memory.available < 8 * 1024 * 1024 * 1024:  # 小于8GB可用内存
        print(f"[警告] 可用内存不足8GB，建议减少MPI进程数或减少模型大小")
        print(f"[建议] 当前可用内存: {total_memory.available/1024/1024/1024:.1f} GB")
    
    print_memory_status("初始化完成后", rank)


# 迭代参数
epochs = 101 #迭代次数
sigma_required_gradient = False  # 用户自定义key

# Tikhonov正则化参数
alpha_tikhonov_eps = 0.015  # epsilon的Tikhonov正则化参数
alpha_tikhonov_sig = 0  # sigma的Tikhonov正则化参数

source_list = [(0, i) for i in range(0, zl, 2)]  # 间隔改为1，覆盖更多区域
receiver_list = source_list.copy()



# ========== 2. 生成观测数据 ==========
if rank == 0:
    print("source_list shape:", np.array(source_list).shape)
    print("生成观测数据...")

d_obs = np.load('data_ist.npy')[2000:2500,:]
print('shape of d_obs:', d_obs.shape) #应该是(5000,2048)



# 把d_obs插值到8000的长度
if d_obs.shape[1] != 8000:
    from scipy.interpolate import interp1d
    n_shots, old_len = d_obs.shape
    new_len = 8000
    # 原始采样率: old_dt = 0.3125e-9
    # 目标采样率: new_dt = 7e-11
    old_dt = 0.3125e-9
    new_dt = 7e-11
    # 生成绝对时间刻度
    t_old = np.arange(old_len) * old_dt
    t_new = np.arange(new_len) * new_dt
    # 截断t_new超出t_old范围（仅外插一点也ok）
    t_new_max = t_new[-1]
    t_old_max = t_old[-1]
    # 但填充值这里允许外推，fill_value="extrapolate"
    d_obs_interp = np.zeros((n_shots, new_len))
    for i in range(n_shots):
        f_interp = interp1d(t_old, d_obs[i], kind='linear', fill_value="extrapolate")
        d_obs_interp[i] = f_interp(t_new)
        # 确认对齐点完全对应（严格验证绝对时间点插值）：原第j点在目标的round(j*old_dt/new_dt)点
    d_obs = d_obs_interp
    print(f"d_obs 已根据绝对时间插值至 shape: {d_obs.shape}, old_dt: {old_dt}, new_dt: {new_dt}")


if rank == 0:
    plt.figure()
    plt.imshow(d_obs.T, aspect='auto', cmap='seismic',vmin=-0.01,vmax=0.01)
    plt.title('B-scan d_obs')
    plt.colorbar()
    plt.savefig(f'{results_dir}/obs_data.png')
    plt.close()

# ========== 3. FWI主循环 ==========
model_eps = epsilon0.copy()
model_sig = sigma0.copy()
loss_list = []

#设置RMSprop超参数
beta_eps = 0.9
beta_sigma = 0.9
alpha_eps = 0.03
alpha_sigma = 0.01
V = np.zeros_like(model_eps.flatten())
U = np.zeros_like(model_sig.flatten())

# 工具函数已在文件开头定义

for epoch in range(epochs):
    epoch_start_time = time.time()  # 记录每个epoch的开始时间
    if rank == 0:
        print(f"Epoch {epoch+1}/{epochs}")
        print_memory_status(f"Epoch {epoch+1} 开始", rank)

    selected_source_list = source_list
    selected_receiver_list = receiver_list
    selected_d_obs = d_obs

    if rank == 0:
        # print('current selected_source_list:', selected_source_list)
        # print('current selected_receiver_list:', selected_receiver_list)
        print('current selected_d_obs shape:', selected_d_obs.shape)
    
    # 梯度计算 - 每个进程分别进行对应source的正演、计算residual和梯度
    
    
    # 计算每个进程处理的shot数量
    n_shots = len(selected_source_list)
    shots_per_proc = n_shots // size
    remainder = n_shots % size
    
    # 分配shot索引给每个进程
    if rank < remainder:
        start_idx = rank * (shots_per_proc + 1)
        end_idx = start_idx + shots_per_proc + 1
    else:
        start_idx = rank * shots_per_proc + remainder
        end_idx = start_idx + shots_per_proc
    
    local_source_list = selected_source_list[start_idx:end_idx]
    local_receiver_list = selected_receiver_list[start_idx:end_idx]
    local_d_obs = selected_d_obs[start_idx:end_idx]
    
    if rank == 0:
        print(f"  Total shots: {n_shots}, shots per proc: {shots_per_proc}, remainder: {remainder}")
        print(f"  Process {rank}: shots {start_idx}-{end_idx-1} (indices {start_idx}:{end_idx})")
        print(f"  Local sources: {local_source_list}")
        print(f"  Local d_obs shape: {local_d_obs.shape}")
    
        # 本地梯度计算
    local_grad_eps = np.zeros_like(model_eps)
    local_grad_sig = np.zeros_like(model_sig) if sigma_required_gradient else None
    local_residual = None  # 初始化local_residual变量
    
    if len(local_source_list) > 0:
        # 对本地shots进行正演，保存波场数据
        # 现在每个进程只处理自己的本地shots，返回本地数据
        if rank == 0:
            print("  开始正演...")
            print_memory_status("正演开始前", rank)
        # 根据epoch决定是否保存波场数据以减少内存使用
        save_wavefield = (epoch == 0)  # 只在第一个epoch保存波场数据用于调试
        
        # forward_model根据save_wavefield返回不同数量的值
        result = forward_model(
            model_eps, model_sig, local_source_list, local_receiver_list, 
            dt, dx, dz, npml, freq, steps, save_wavefield=save_wavefield
        )
        
        if save_wavefield:
            local_d_syn, local_wavefield_data = result
        else:
            local_d_syn = result
            local_wavefield_data = None  # 不保存波场数据
        
        if rank == 0:
            print_memory_status("正演完成后", rank)
        if rank == 0:
            print(f"  Local d_syn shape: {local_d_syn.shape}")
            if epoch == 0:
                print('saving test_d_syn')
                np.save('results/test_d_syn.npy', local_d_syn)
                print('test_wavefield_data saved')
        # 计算本地residual
        local_residual = local_d_syn - local_d_obs
        
        # 计算本地梯度
        # 传递全局索引信息，确保梯度计算正确
        if rank == 0:
            print("  计算梯度...")
            print_memory_status("梯度计算开始前", rank)
        
        if save_wavefield:
            # 如果保存了波场数据，直接使用
            grad_eps, grad_sig = compute_gradient(
                model_eps, model_sig, local_residual, local_source_list, local_receiver_list,
                dt, dx, dz, npml, freq, steps, sigma_required_gradient, local_wavefield_data,
                global_start_idx=start_idx  # 传递全局起始索引
            )
        else:
            # 如果没有保存波场数据，梯度计算函数会重新进行正演
            grad_eps, grad_sig = compute_gradient(
                model_eps, model_sig, local_residual, local_source_list, local_receiver_list,
                dt, dx, dz, npml, freq, steps, sigma_required_gradient, None,
                global_start_idx=start_idx  # 传递全局起始索引
            )
        
        if rank == 0:
            print_memory_status("梯度计算完成后", rank)
            # 清理波场数据以释放内存
            if not save_wavefield and 'local_wavefield_data' in locals():
                del local_wavefield_data
                gc.collect()
        print(f"grad_eps shape: {grad_eps.shape}, grad_sig shape: {grad_sig.shape if grad_sig is not None else None}")
        assert grad_eps.shape == model_eps.shape, "grad_eps和model_eps的形状不一致"
        
        # 在rank 0保存和显示结果（用于调试和可视化）
        if rank == 0 and should_save_epoch(epoch):
             # 保存模拟数据
            plt.figure()
            plt.imshow(local_d_syn.T, aspect='auto', cmap='seismic')
            plt.title(f'Local simulated data (epoch {epoch})')
            plt.colorbar()
            plt.savefig(f'{results_dir}/local_syn_data_epoch{epoch}.png')
            plt.close()
     
     # 合并所有进程的local_d_syn得到总的d_syn
    if rank == 0:
         print("  合并所有进程的模拟数据...")
     
     # 收集所有进程的local_d_syn
    all_local_d_syn = comm.gather(local_d_syn, root=0)
     
    if rank == 0:
         # 重新组织总的模拟数据
        d_syn = np.zeros((n_shots, steps))
        data_idx = 0
        for i, proc_d_syn in enumerate(all_local_d_syn):
            if proc_d_syn is not None and proc_d_syn.shape[0] > 0:
                n_local = proc_d_syn.shape[0]
                d_syn[data_idx:data_idx+n_local] = proc_d_syn
                data_idx += n_local
         
        print(f"  Total d_syn shape: {d_syn.shape}")
        print(f"  Total d_obs shape: {selected_d_obs.shape}")
        assert d_syn.shape == selected_d_obs.shape, "d_syn和d_obs的形状不一致"
         
         # 计算总的残差
        residual = d_syn - selected_d_obs
        print(f"  Total residual shape: {residual.shape}")
        print(f"  Total residual range: [{np.min(residual)}, {np.max(residual)}]")
         
         # 保存总的模拟数据
        if should_save_epoch(epoch):
            plt.figure()
            plt.imshow(d_syn.T, aspect='auto', cmap='seismic')
            plt.title(f'Total simulated data (epoch {epoch})')
            plt.colorbar()
            plt.savefig(f'{results_dir}/total_syn_data_epoch{epoch}.png')
            plt.close()
             
             # 保存总的残差数据
            plt.figure()
            plt.imshow(residual.T, aspect='auto', cmap='bwr')
            plt.title(f'Total residual (epoch {epoch})')
            plt.colorbar()
            plt.savefig(f'{results_dir}/total_residual_epoch{epoch}.png')
            plt.close()
        
        # 如果是初始模型（epoch==0），额外保存.npy文件
        if epoch == 0:
            # 保存初始模型正演输出
            np.save(f'{results_dir}/d_syn_initial.npy', d_syn)
            print(f"初始模型正演输出已保存到: {results_dir}/d_syn_initial.npy")
            
            # 保存初始残差
            np.save(f'{results_dir}/residual_initial.npy', residual)
            print(f"初始模型残差已保存到: {results_dir}/residual_initial.npy")
            
            # 计算初始模型损失
            loss_initial = np.linalg.norm(residual)
            print(f"初始模型损失: {loss_initial:.6f}")
     
     # 广播总的d_syn和residual到所有进程
    d_syn = comm.bcast(d_syn if rank == 0 else None, root=0)
    residual = comm.bcast(residual if rank == 0 else None, root=0)
    
    if rank == 0:
        check_nan_inf(grad_eps, f"grad_eps (epoch {epoch})")
    
    # 只在特定epoch保存结果
    if should_save_epoch(epoch):
        #np.save(f'results/grad_eps_epoch{epoch}.npy', grad_eps)
        plt.figure()
        plt.imshow(grad_eps, cmap='jet')
        plt.title(f'epsilon gradient (epoch {epoch})')
        plt.colorbar()
        plt.savefig(f'{results_dir}/grad_eps_epoch{epoch}.png')
        plt.close()
    
    if sigma_required_gradient and grad_sig is not None:
        # 只在特定epoch保存结果
        if should_save_epoch(epoch):
            #np.save(f'results/grad_sig_epoch{epoch}.npy', grad_sig)
            plt.figure()
            plt.imshow(grad_sig, cmap='jet')
            plt.title(f'sigma gradient (epoch {epoch})')
            plt.colorbar()
            plt.savefig(f'{results_dir}/grad_sig_epoch{epoch}.png')
            plt.close()
    
    #得到总的残差residual
    # 添加二阶Tikhonov正则项梯度
    if rank == 0:
        print("计算Tikhonov正则项梯度...")
    tikhonov_grad_eps = compute_tikhonov_gradient(model_eps.copy(), dx, dz, alpha_tikhonov_eps)
    tikhonov_grad_eps[:20,:] = 0
    if rank == 0:
        print(f"Tikhonov epsilon梯度: max={np.max(tikhonov_grad_eps)}, min={np.min(tikhonov_grad_eps)}, mean={np.mean(tikhonov_grad_eps)}")

    # 模型更新
    if rank == 0:
        print("  更新模型...")
    g_eps = grad_eps.flatten()
    V = beta_eps * V + (1 - beta_eps) * g_eps**2
    update_eps = g_eps/np.sqrt(V + 1e-8)
    max_update_eps = np.max(np.abs(update_eps))
    update_eps_norm = update_eps/max_update_eps
    if rank == 0:
        print(f"归一化更新量update_eps_norm: max={np.max(update_eps_norm)}, min={np.min(update_eps_norm)}, mean={np.mean(update_eps_norm)}")
    
    # 只在特定epoch保存结果
    if should_save_epoch(epoch):
        #np.save(f'results/update_eps_norm_epoch{epoch}.npy', update_eps_norm)
        plt.figure()
        plt.imshow(update_eps_norm.reshape(model_eps.shape), cmap='viridis')
        plt.title(f'epsilon update norm (epoch {epoch})')
        plt.colorbar()
        plt.savefig(f'{results_dir}/update_eps_norm_epoch{epoch}.png')
        plt.close()
    
    d_eps = -alpha_eps * update_eps_norm-tikhonov_grad_eps.flatten()
    
    d_eps = d_eps.reshape(model_eps.shape)
    model_eps = model_eps + d_eps
    #限制范围>=1
    model_eps = np.maximum(model_eps, 1)
    check_nan_inf(model_eps, f"model_eps (epoch {epoch})")
    
    # 只在特定epoch保存结果
    #if should_save_epoch(epoch):
        #np.save(f'results/model_eps_epoch{epoch}.npy', model_eps)
    
    if sigma_required_gradient and grad_sig is not None:
        g_sig = grad_sig.flatten()
        U = beta_sigma * U + (1 - beta_sigma) * g_sig**2
        update_sig = g_sig/np.sqrt(U + 1e-8)
        max_update_sig = np.max(np.abs(update_sig))
        update_sig_norm = update_sig/max_update_sig
        print(f"归一化更新量update_sig_norm: max={np.max(update_sig_norm)}, min={np.min(update_sig_norm)}, mean={np.mean(update_sig_norm)}")
        
        # 只在特定epoch保存结果
        if should_save_epoch(epoch):
            #np.save(f'results/update_sig_norm_epoch{epoch}.npy', update_sig_norm)
            plt.figure()
            plt.imshow(update_sig_norm.copy().reshape(model_sig.shape), cmap='viridis')
            plt.title(f'sigma update norm (epoch {epoch})')
            plt.colorbar()
            plt.savefig(f'{results_dir}/update_sig_norm_epoch{epoch}.png')
            plt.close()

        tikhonov_grad_sig = compute_tikhonov_gradient(model_sig.copy(), dx, dz, alpha_tikhonov_sig)
        tikhonov_grad_sig[:20,0] = 0
        print(f"Tikhonov sigma梯度: max={np.max(tikhonov_grad_sig)}, min={np.min(tikhonov_grad_sig)}, mean={np.mean(tikhonov_grad_sig)}")

        d_sig = -alpha_sigma * update_sig_norm-tikhonov_grad_sig.flatten()
        d_sig = d_sig.reshape(model_sig.shape)
        model_sig = model_sig + d_sig
        #限制范围>=0
        model_sig = np.maximum(model_sig, 0)
        
        # 只在特定epoch保存结果
        #if should_save_epoch(epoch):
            #np.save(f'results/model_sig_epoch{epoch}.npy', model_sig)
    
    # 只在特定epoch保存结果
    if should_save_epoch(epoch):
        plt.figure()
        plt.imshow(model_eps, cmap='jet',vmin=0,vmax=10)
        plt.title(f'epsilon model (epoch {epoch})')
        plt.colorbar()
        plt.savefig(f'{results_dir}/model_eps_epoch{epoch}.png')
        plt.close()
    
    if sigma_required_gradient and grad_sig is not None:
        # 只在特定epoch保存结果
        if should_save_epoch(epoch):
            plt.figure()
            plt.imshow(model_sig, cmap='jet',vmin=0,vmax=1e-3)
            plt.title(f'sigma model (epoch {epoch})')
            plt.colorbar()
            plt.savefig(f'{results_dir}/model_sig_epoch{epoch}.png')
            plt.close()

    loss = np.linalg.norm(residual)
    loss_list.append(loss)

    epoch_end_time = time.time()  # 记录每个epoch的结束时间
    epoch_duration = epoch_end_time - epoch_start_time  # 计算每个epoch的耗时
    if rank == 0:
        print(f"  Epoch {epoch+1}/{epochs} 耗时: {epoch_duration:.2f}秒")

    # ========== 4. 可视化与保存 ==========
    # 每个epoch都保存损失曲线
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('FWI Loss Curve')
    plt.savefig(f'{results_dir}/loss_curve.png')
    plt.close()
    #np.save('results/loss_curve.npy', np.array(loss_list))

    # 保存最终模型
    #np.save('results/final_model_eps.npy', model_eps)
    #np.save('results/final_model_sig.npy', model_sig)
    plt.figure()
    plt.imshow(model_eps, cmap='jet')
    plt.title('Final epsilon model')
    plt.colorbar()
    plt.savefig(f'{results_dir}/final_model_eps.png')
    plt.close()

    if sigma_required_gradient and grad_sig is not None:
        plt.figure()
        plt.imshow(model_sig, cmap='jet')
        plt.title('Final sigma model')
        plt.colorbar()
        plt.savefig('results/final_model_sig.png')
        plt.close()

if rank == 0:
    print("FWI完成！结果保存在results文件夹中。") 