import numpy as np
import matplotlib.pyplot as plt
from forward import forward_model
from gradient import compute_gradient, compute_tikhonov_gradient
import os
import shutil
import time
from tqdm import tqdm   

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
plt.imshow(epsilon_true, aspect='auto', cmap='jet',vmin=0,vmax=10)
plt.title('Epsilon True')
plt.colorbar()
plt.savefig('results/epsilon_true.png')
plt.close()

plt.figure()
plt.imshow(sigma_true, aspect='auto', cmap='jet',vmin=0,vmax=1e-3)
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
plt.imshow(epsilon0, aspect='auto', cmap='jet',vmin=0,vmax=10)
plt.title('Epsilon Initial')
plt.colorbar()
plt.savefig('results/epsilon_initial.png')
plt.close()

plt.figure()
plt.imshow(sigma0, aspect='auto', cmap='jet',vmin=0,vmax=1e-3)
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
epochs = 1000 #迭代次数
sigma_required_gradient = False  # 用户自定义key

# Tikhonov正则化参数
alpha_tikhonov_eps = 0.015  # epsilon的Tikhonov正则化参数
alpha_tikhonov_sig = 0 # sigma的Tikhonov正则化参数

# 源点和接收点设置 - mode2: 常规排列，每一炮对应相同数量的接收器
source_list = [(0, i) for i in range(10, zl-10, 10)]
receiver_list = [(0, i) for i in range(10, zl-10, 5)]  # 接收器位置固定
print('source_list:', source_list)
print('receiver_list:', receiver_list)

os.makedirs('results', exist_ok=True)

# ========== 2. 生成观测数据 ==========
print("生成观测数据...")

d_obs = forward_model(epsilon_true, sigma_true, source_list, receiver_list, dt, dx, dz, npml, freq, steps, save_wavefield=False)

# ========== 3. FWI主循环 ==========
model_eps = epsilon0.copy()
model_sig = sigma0.copy()
loss_list = []

#设置RMSprop超参数
beta_eps = 0.9
beta_sigma = 0.9
alpha_eps = 0.02
alpha_sigma = 0.01
V = np.zeros_like(model_eps.flatten())
U = np.zeros_like(model_sig.flatten())

def check_nan_inf(arr, name):
    if np.any(np.isnan(arr)):
        print(f"[警告] {name} 出现nan!")
    if np.any(np.isinf(arr)):
        print(f"[警告] {name} 出现inf!")
    print(f"[{name}] min={np.nanmin(arr)}, max={np.nanmax(arr)}, mean={np.nanmean(arr)}")

random_activate = True
for epoch in range(epochs):
    epoch_start_time = time.time()  # 记录每个epoch的开始时间
    print(f"Epoch {epoch+1}/{epochs}")
    
    if random_activate:
        # 先获取源的总数
        n = len(source_list)
        # 随机生成起始点k（1~n），注意Python索引从0开始
        k = np.random.randint(1, n+1)
        # 生成随机序列：k, k+3, k+6, ...，并且mod n循环直到回到k
        selected_indices = []
        idx = k - 1  # 转为0-based索引
        while True:
            if idx not in selected_indices:
                selected_indices.append(idx)
            idx = (idx + 3) % n
            if idx == (k - 1):
                break
        # 用selected_indices筛选source_list
        selected_source_list = [source_list[i] for i in selected_indices]
    else:
        selected_indices = range(len(source_list))
        selected_source_list = source_list

    print('current selected_source_list:', selected_source_list)
    selected_d_obs = d_obs[selected_indices,:,:]
    # 正演模拟（保存波场数据用于梯度计算）
    print("  正演模拟...")
    d_syn, wavefield_data = forward_model(model_eps, model_sig, selected_source_list, receiver_list, dt, dx, dz, npml, freq, steps, save_wavefield=True)

    # 残差
    residual = d_syn - selected_d_obs

    # 梯度计算
    print("  计算梯度...")
    grad_eps, grad_sig = compute_gradient(model_eps, model_sig, residual, selected_source_list, receiver_list, dt, dx, dz, npml, freq, steps, sigma_required_gradient, wavefield_data, alpha_tikhonov_eps,alpha_tikhonov_sig)
    check_nan_inf(grad_eps, f"grad_eps (epoch {epoch})")

    np.save(f'results/grad_eps_epoch{epoch}.npy', grad_eps)
    plt.figure()
    plt.imshow(grad_eps, cmap='viridis')
    plt.title(f'epsilon gradient (epoch {epoch})')
    plt.colorbar()
    plt.savefig(f'results/grad_eps_epoch{epoch}.png')
    plt.close()
    
    if sigma_required_gradient and grad_sig is not None:
        np.save(f'results/grad_sig_epoch{epoch}.npy', grad_sig)
        plt.figure()
        plt.imshow(grad_sig, cmap='viridis')
        plt.title(f'sigma gradient (epoch {epoch})')
        plt.colorbar()
        plt.savefig(f'results/grad_sig_epoch{epoch}.png')
        plt.close()
        
    # 添加二阶Tikhonov正则项梯度
    print("计算Tikhonov正则项梯度...")
    tikhonov_grad_eps = compute_tikhonov_gradient(model_eps.copy(), dx, dz, alpha_tikhonov_eps)
    tikhonov_grad_eps[:20,:] = 0
    print(f"Tikhonov epsilon梯度: max={np.max(tikhonov_grad_eps)}, min={np.min(tikhonov_grad_eps)}, mean={np.mean(tikhonov_grad_eps)}")

    # 模型更新
    print("  更新模型...")
    g_eps = grad_eps.flatten()
    V = beta_eps * V + (1 - beta_eps) * g_eps**2
    update_eps = g_eps/np.sqrt(V + 1e-8)
    max_update_eps = np.max(np.abs(update_eps))
    update_eps_norm = update_eps/max_update_eps
    print(f"归一化更新量update_eps_norm: max={np.max(update_eps_norm)}, min={np.min(update_eps_norm)}, mean={np.mean(update_eps_norm)}")
    
    np.save(f'results/update_eps_norm_epoch{epoch}.npy', update_eps_norm)
    plt.figure()
    plt.imshow(update_eps_norm.reshape(model_eps.shape), cmap='viridis')
    plt.title(f'epsilon update norm (epoch {epoch})')
    plt.colorbar()
    plt.savefig(f'results/update_eps_norm_epoch{epoch}.png')
    plt.close()

    d_eps = -alpha_eps * update_eps_norm-tikhonov_grad_eps.flatten()
    
    d_eps = d_eps.reshape(model_eps.shape)
    model_eps = model_eps + d_eps
    #限制范围>=1
    model_eps = np.maximum(model_eps, 1)
    check_nan_inf(model_eps, f"model_eps (epoch {epoch})")
    np.save(f'results/model_eps_epoch{epoch}.npy', model_eps)
    
    if sigma_required_gradient and grad_sig is not None:
        g_sig = grad_sig.flatten()
        U = beta_sigma * U + (1 - beta_sigma) * g_sig**2
        update_sig = g_sig/np.sqrt(U + 1e-8)
        max_update_sig = np.max(np.abs(update_sig))
        update_sig_norm = update_sig/max_update_sig
        print(f"归一化更新量update_sig_norm: max={np.max(update_sig_norm)}, min={np.min(update_sig_norm)}, mean={np.mean(update_sig_norm)}")
        np.save(f'results/update_sig_norm_epoch{epoch}.npy', update_sig_norm)
        plt.figure()
        plt.imshow(update_sig_norm.copy().reshape(model_sig.shape), cmap='viridis')
        plt.title(f'sigma update norm (epoch {epoch})')
        plt.colorbar()
        plt.savefig(f'results/update_sig_norm_epoch{epoch}.png')
        plt.close()

        tikhonov_grad_sig = compute_tikhonov_gradient(model_sig.copy(), dx, dz, alpha_tikhonov_sig)
        print(f"Tikhonov sigma梯度: max={np.max(tikhonov_grad_sig)}, min={np.min(tikhonov_grad_sig)}, mean={np.mean(tikhonov_grad_sig)}")

        d_sig = -alpha_sigma * update_sig_norm-tikhonov_grad_sig.flatten()
        d_sig = d_sig.reshape(model_sig.shape)
        model_sig = model_sig + d_sig
        #限制范围>=0
        model_sig = np.maximum(model_sig, 0)
        np.save(f'results/model_sig_epoch{epoch}.npy', model_sig)
    
    plt.figure()
    plt.imshow(model_eps, cmap='jet',vmin=0,vmax=10)
    plt.title(f'epsilon model (epoch {epoch})')
    plt.colorbar()
    plt.savefig(f'results/model_eps_epoch{epoch}.png')
    plt.close()
    
    if sigma_required_gradient and grad_sig is not None:
        plt.figure()
        plt.imshow(model_sig, cmap='jet',vmin=0,vmax=1e-3)
        plt.title(f'sigma model (epoch {epoch})')
        plt.colorbar()
        plt.savefig(f'results/model_sig_epoch{epoch}.png')
        plt.close()

    # 计算损失
    loss = np.linalg.norm(residual)
    loss_list.append(loss)
    print(f"  Loss: {loss:.6f}")

    epoch_end_time = time.time()  # 记录每个epoch的结束时间
    epoch_duration = epoch_end_time - epoch_start_time  # 计算每个epoch的耗时
    print(f"  Epoch {epoch+1}/{epochs} 耗时: {epoch_duration:.2f}秒")

# ========== 4. 可视化与保存 ==========
plt.figure()
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('FWI Loss Curve')
plt.savefig('results/loss_curve.png')
plt.close()
np.save('results/loss_curve.npy', np.array(loss_list))

# 保存最终模型
np.save('results/final_model_eps.npy', model_eps)
np.save('results/final_model_sig.npy', model_sig)
plt.figure()
plt.imshow(model_eps, cmap='jet')
plt.title('Final epsilon model')
plt.colorbar()
plt.savefig('results/final_model_eps.png')
plt.close()

if sigma_required_gradient and grad_sig is not None:
    plt.figure()
    plt.imshow(model_sig, cmap='jet')
    plt.title('Final sigma model')
    plt.colorbar()
    plt.savefig('results/final_model_sig.png')
    plt.close()

print("FWI完成！结果保存在results文件夹中。") 