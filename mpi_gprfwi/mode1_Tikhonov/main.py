import numpy as np
import matplotlib.pyplot as plt
from forward import forward_model
from gradient import compute_gradient, compute_tikhonov_gradient
import os
import shutil
import time
from tqdm import tqdm

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
results_dir = os.path.expanduser("~/swz/mpi_gprfwi/mode1_Tikhonov/results")
if rank == 0:
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    print(f"结果将保存到: {results_dir}")
comm.Barrier()

# ========== 1. 参数设置 ==========
# 网格参数
xl, zl = 100, 200 #纵向和横向网格数
k_max = 1000 #最大迭代次数
dx, dz = 0.02, 0.02 #网格间距
dt = 4e-11 #时间步长
npml = 10 #PML层厚度
steps = 1000 #时间步数

# 震源参数
freq = 4e8 #震源主频

# 真实模型参数
epsilon_true = np.ones((xl, zl))*3
epsilon_true[40:60,90:110]=1.5
sigma_true = np.ones((xl, zl)) * 1e-4
epsilon_true[85:,:]=6

# 保存真实模型参数
#np.save('~/results/epsilon_true.npy', epsilon_true)
#np.save('~/results/sigma_true.npy', sigma_true)

# 画图保存
plt.figure()
plt.imshow(epsilon_true, aspect='auto', cmap='jet',vmin=0,vmax=10)
plt.title('Epsilon True')
plt.colorbar()
plt.savefig(f'{results_dir}/epsilon_true.png')
plt.close()

plt.figure()
plt.imshow(sigma_true, aspect='auto', cmap='jet',vmin=0,vmax=1e-3)
plt.title('Sigma True')
plt.colorbar()
plt.savefig(f'{results_dir}/sigma_true.png')
plt.close()



# 初始模型参数
epsilon0 = np.ones((xl, zl)) * 3
epsilon0[85:,:]=6
sigma0 = np.ones((xl, zl)) * 1e-4

# 画图保存
plt.figure()
plt.imshow(epsilon0, aspect='auto', cmap='viridis',vmin=0,vmax=10)
plt.title('Epsilon Initial')
plt.colorbar()
plt.savefig(f'{results_dir}/epsilon_initial.png')
plt.close()

plt.figure()
plt.imshow(sigma0, aspect='auto', cmap='viridis',vmin=0,vmax=1e-3)
plt.title('Sigma Initial')
plt.colorbar()
plt.savefig(f'{results_dir}/sigma_initial.png')
plt.close()
# 检查CFL条件
c = 3e8  # 光速
epsilon_max = epsilon_true.max()
cfl_condition = dt < 1 / (c * np.sqrt(1/dx**2 + 1/dz**2) / np.sqrt(epsilon_max))
if rank == 0:
    print(f"MPI 进程数: {size}")
    print(f"CFL条件检查: {cfl_condition}")
    print(f"当前dt: {dt}, 最大允许dt: {1 / (c * np.sqrt(1/dx**2 + 1/dz**2) / np.sqrt(epsilon_max))}")
# 迭代参数
epochs = 2001 #迭代次数
sigma_required_gradient = False  # 用户自定义key

# Tikhonov正则化参数
alpha_tikhonov_eps = 0.015  # epsilon的Tikhonov正则化参数
alpha_tikhonov_sig = 0  # sigma的Tikhonov正则化参数

# 源点和接收点设置 - 注意：这些坐标是相对于内部网格的，不包括PML边界
source_list = [(0, i) for i in range(10, zl-10, 2)]

receiver_list = source_list.copy()

# ========== 2. 生成观测数据 ==========
if rank == 0:
    print('source_list:', source_list)
    print("生成观测数据...")

# 前向模拟（forward_model 内部支持 MPI 并行且会在需要时广播数据）
d_obs = forward_model(epsilon_true, sigma_true, source_list, receiver_list, dt, dx, dz, npml, freq, steps, save_wavefield=False)
#注意，forward_model中d_obs是合并过的全部数据，而返回的正演波场是没有合并过的。

if rank == 0:
    #np.save('~/results/d_obs.npy', d_obs)
    plt.figure()
    plt.imshow(d_obs.T, aspect='auto', cmap='seismic')
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

def check_nan_inf(arr, name):
    if rank == 0:
        if np.any(np.isnan(arr)):
            print(f"[警告] {name} 出现nan!")
        if np.any(np.isinf(arr)):
            print(f"[警告] {name} 出现inf!")
        print(f"[{name}] min={np.nanmin(arr)}, max={np.nanmax(arr)}, mean={np.nanmean(arr)}")

def should_save_epoch(epoch):
    """判断是否应该保存当前epoch的结果"""
    save_epochs = [0, 1, 2, 5, 10, 20, 50]
    return epoch in save_epochs or (epoch >= 50 and epoch % 50 == 0)

for epoch in range(epochs):
    epoch_start_time = time.time()  # 记录每个epoch的开始时间
    if rank == 0:
        print(f"Epoch {epoch+1}/{epochs}")

    selected_source_list = source_list
    selected_receiver_list = receiver_list
    selected_d_obs = d_obs

    if rank == 0:
        print('current selected_source_list:', selected_source_list)
        print('current selected_receiver_list:', selected_receiver_list)
        print('current selected_d_obs shape:', selected_d_obs.shape)
    
    # 梯度计算 - 每个进程分别进行对应source的正演、计算residual和梯度
    if rank == 0:
        print("  计算梯度...")
    
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
        local_d_syn, local_wavefield_data = forward_model(
            model_eps, model_sig, local_source_list, local_receiver_list, 
            dt, dx, dz, npml, freq, steps, save_wavefield=True
        )
        if rank == 0:
            print(f"  Local d_syn shape: {local_d_syn.shape}")
            print(f"  Local d_obs shape: {local_d_obs.shape}")
            assert local_d_syn.shape == local_d_obs.shape, "local_d_syn和local_d_obs的形状不一致"
            print(f"  Local d_syn range: [{np.min(local_d_syn)}, {np.max(local_d_syn)}]")
            print(f"  Local d_obs range: [{np.min(local_d_obs)}, {np.max(local_d_obs)}]")
        
        # 计算本地residual
        local_residual = local_d_syn - local_d_obs
        
        # 计算本地梯度
        # 传递全局索引信息，确保梯度计算正确
        grad_eps, grad_sig = compute_gradient(
            model_eps, model_sig, local_residual, local_source_list, local_receiver_list,
            dt, dx, dz, npml, freq, steps, sigma_required_gradient, local_wavefield_data,
            global_start_idx=start_idx  # 传递全局起始索引
        )
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