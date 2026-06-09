import numpy as np
import matplotlib.pyplot as plt
from forward_parallel import forward_model_parallel
from gradient_parallel import compute_gradient_parallel
import os
import shutil
import time
import multiprocessing

# 删除results文件夹
if os.path.exists('results'):
    shutil.rmtree('results')
os.makedirs('results', exist_ok=True)

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
epsilon_true[30:70,80:120]=1.5
sigma_true = np.ones((xl, zl)) * 1e-4
epsilon_true[85:,:]=6

# 保存真实模型参数
np.save('results/epsilon_true.npy', epsilon_true)
np.save('results/sigma_true.npy', sigma_true)

# 画图保存
plt.figure()
plt.imshow(epsilon_true, aspect='auto', cmap='viridis',vmin=0,vmax=10)
plt.title('Epsilon True')
plt.colorbar()
plt.savefig('results/epsilon_true.png')
plt.close()

plt.figure()
plt.imshow(sigma_true, aspect='auto', cmap='viridis',vmin=0,vmax=1e-3)
plt.title('Sigma True')
plt.colorbar()
plt.savefig('results/sigma_true.png')
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
plt.savefig('results/epsilon_initial.png')
plt.close()

plt.figure()
plt.imshow(sigma0, aspect='auto', cmap='viridis',vmin=0,vmax=1e-3)
plt.title('Sigma Initial')
plt.colorbar()
plt.savefig('results/sigma_initial.png')
plt.close()

# 检查CFL条件
c = 3e8  # 光速
epsilon_max = epsilon_true.max()
cfl_condition = dt < 1 / (c * np.sqrt(1/dx**2 + 1/dz**2) / np.sqrt(epsilon_max))
print(f"CFL条件检查: {cfl_condition}")
print(f"当前dt: {dt}, 最大允许dt: {1 / (c * np.sqrt(1/dx**2 + 1/dz**2) / np.sqrt(epsilon_max))}")

# 迭代参数
epochs = 10 #迭代次数
sigma_required_gradient = False  # 用户自定义key

# 源点和接收点设置 - 注意：这些坐标是相对于内部网格的，不包括PML边界
source_list = [(0, i) for i in range(10, zl-10, 5)]
print('source_list:', source_list)
receiver_list = source_list.copy()

# 并行计算参数
n_processes = min(multiprocessing.cpu_count(), len(source_list))  # 使用CPU核心数或炮点数中的较小值
print(f"将使用 {n_processes} 个进程进行并行计算")

os.makedirs('results', exist_ok=True)

# ========== 2. 生成观测数据 ==========
print("生成观测数据...")
start_time = time.time()

d_obs = forward_model_parallel(epsilon_true, sigma_true, source_list, receiver_list, dt, dx, dz, npml, freq, steps, save_wavefield=False, n_processes=n_processes)
np.save('results/d_obs.npy', d_obs)
plt.figure()
plt.imshow(d_obs.T, aspect='auto', cmap='seismic')
plt.title('B-scan d_obs')
plt.colorbar()
plt.savefig('results/obs_data.png')
plt.close()

print(f"观测数据生成完成，耗时: {time.time() - start_time:.2f} 秒")

# ========== 3. FWI主循环 ==========
model_eps = epsilon0.copy()
model_sig = sigma0.copy()
loss_list = []

#设置RMSprop超参数
beta_eps = 0.9
beta_sigma = 0.9
alpha_eps = 0.01
alpha_sigma = 0.01
V = np.zeros_like(model_eps.flatten())
U = np.zeros_like(model_sig.flatten())

def check_nan_inf(arr, name):
    if np.any(np.isnan(arr)):
        print(f"[警告] {name} 出现nan!")
    if np.any(np.isinf(arr)):
        print(f"[警告] {name} 出现inf!")
    print(f"[{name}] min={np.nanmin(arr)}, max={np.nanmax(arr)}, mean={np.nanmean(arr)}")

total_start_time = time.time()

for epoch in range(epochs):
    epoch_start_time = time.time()
    print(f"Epoch {epoch+1}/{epochs}")
    
    # 正演模拟（保存波场数据用于梯度计算）
    print("  正演模拟...")
    forward_start_time = time.time()
    d_syn, wavefield_data = forward_model_parallel(model_eps, model_sig, source_list, receiver_list, dt, dx, dz, npml, freq, steps, save_wavefield=True, n_processes=n_processes)
    print(f"  正演模拟完成，耗时: {time.time() - forward_start_time:.2f} 秒")
    
    check_nan_inf(d_syn.T, f"d_syn (epoch {epoch})")
    np.save(f'results/d_syn_epoch{epoch}.npy', d_syn)
    
    plt.figure()
    plt.imshow(d_syn, aspect='auto', cmap='seismic')
    plt.title(f'模拟数据 d_syn (epoch {epoch})')
    plt.colorbar()
    plt.savefig(f'results/syn_data_epoch{epoch}.png')
    plt.close()

    # 残差
    residual = d_syn - d_obs
    check_nan_inf(residual, f"residual (epoch {epoch})")
    np.save(f'results/residual_epoch{epoch}.npy', residual)
    plt.figure()
    plt.imshow(residual.T, aspect='auto', cmap='bwr')
    plt.title(f'残差 (epoch {epoch})')
    plt.colorbar()
    plt.savefig(f'results/residual_epoch{epoch}.png')
    plt.close()

    # 梯度计算
    print("  计算梯度...")
    gradient_start_time = time.time()
    grad_eps, grad_sig = compute_gradient_parallel(model_eps, model_sig, residual, source_list, receiver_list, dt, dx, dz, npml, freq, steps, sigma_required_gradient, wavefield_data, n_processes=n_processes)
    print(f"  梯度计算完成，耗时: {time.time() - gradient_start_time:.2f} 秒")
    
    check_nan_inf(grad_eps, f"grad_eps (epoch {epoch})")

    np.save(f'results/grad_eps_epoch{epoch}.npy', grad_eps)
    plt.figure()
    plt.imshow(grad_eps, cmap='viridis')
    plt.title(f'epsilon梯度 (epoch {epoch})')
    plt.colorbar()
    plt.savefig(f'results/grad_eps_epoch{epoch}.png')
    plt.close()
    
    if sigma_required_gradient and grad_sig is not None:
        np.save(f'results/grad_sig_epoch{epoch}.npy', grad_sig)
        plt.figure()
        plt.imshow(grad_sig, cmap='viridis')
        plt.title(f'sigma梯度 (epoch {epoch})')
        plt.colorbar()
        plt.savefig(f'results/grad_sig_epoch{epoch}.png')
        plt.close()

    # 模型更新
    print("  更新模型...")
    g_eps = grad_eps.flatten()
    V = beta_eps * V + (1 - beta_eps) * g_eps**2

    d_eps = -alpha_eps * g_eps/np.sqrt(V + 1e-8)
    
    d_eps = d_eps.reshape(model_eps.shape)
    model_eps = model_eps + d_eps
    check_nan_inf(model_eps, f"model_eps (epoch {epoch})")
    np.save(f'results/model_eps_epoch{epoch}.npy', model_eps)
    
    if sigma_required_gradient and grad_sig is not None:
        g_sig = grad_sig.flatten()
        U = beta_sigma * U + (1 - beta_sigma) * g_sig**2
        d_sig = -alpha_sigma * g_sig/np.sqrt(U + 1e-8)
        d_sig = d_sig.reshape(model_sig.shape)
        model_sig = model_sig + d_sig
        np.save(f'results/model_sig_epoch{epoch}.npy', model_sig)
    
    plt.figure()
    plt.imshow(model_eps, cmap='jet',vmin=0,vmax=10)
    plt.title(f'epsilon模型 (epoch {epoch})')
    plt.colorbar()
    plt.savefig(f'results/model_eps_epoch{epoch}.png')
    plt.close()
    
    if sigma_required_gradient and grad_sig is not None:
        plt.figure()
        plt.imshow(model_sig, cmap='jet',vmin=0,vmax=1e-3)
        plt.title(f'sigma模型 (epoch {epoch})')
        plt.colorbar()
        plt.savefig(f'results/model_sig_epoch{epoch}.png')
        plt.close()

    # 计算损失
    loss = np.linalg.norm(residual)
    loss_list.append(loss)
    print(f"  Loss: {loss:.6f}")
    print(f"  Epoch {epoch+1} 总耗时: {time.time() - epoch_start_time:.2f} 秒")

# ========== 4. 可视化与保存 ==========
plt.figure()
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('FWI Loss 曲线')
plt.savefig('results/loss_curve.png')
plt.close()
np.save('results/loss_curve.npy', np.array(loss_list))

# 保存最终模型
np.save('results/final_model_eps.npy', model_eps)
np.save('results/final_model_sig.npy', model_sig)
plt.figure()
plt.imshow(model_eps, cmap='jet')
plt.title('最终epsilon模型')
plt.colorbar()
plt.savefig('results/final_model_eps.png')
plt.close()

if sigma_required_gradient and grad_sig is not None:
    plt.figure()
    plt.imshow(model_sig, cmap='jet')
    plt.title('最终sigma模型')
    plt.colorbar()
    plt.savefig('results/final_model_sig.png')
    plt.close()

total_time = time.time() - total_start_time
print(f"FWI完成！总耗时: {total_time:.2f} 秒")
print("结果保存在results文件夹中。") 