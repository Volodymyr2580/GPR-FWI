import numpy as np
import matplotlib.pyplot as plt
from forward import forward_model
from gradient import compute_gradient, compute_tikhonov_gradient
import os
import shutil
import random
import time
from tqdm import tqdm

def main():
    # 删除results文件夹
    if os.path.exists('results'):
        shutil.rmtree('results')
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/update', exist_ok=True)


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
    epsilon_true = np.ones((xl, zl))*2
    epsilon_true[40:60,90:110]=8
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
    epsilon0 = np.ones((xl, zl)) * 2
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
    
    #设置初始温度和退火参数
    T0 = 10000 #初始温度
    T_min = 1e-6 #最低温度
    cooling_rate = 0.0001

    #模型限定范围
    eps_max = 10
    eps_min = 1
    delta_eps = eps_max-eps_min

    sig_max = 1e-3
    sig_min = 1e-6
    delta_sig = sig_max-sig_min
    
    def get_current_temperature(epoch):
        """
        计算当前温度，提供多种温度递减策略
        策略1: 指数衰减 (推荐)
        策略2: 线性衰减
        策略3: 对数衰减
        """
        # 策略1: 指数衰减 (推荐)
        return T0 * np.exp(-cooling_rate * epoch)

        # 策略2: 线性衰减 (取消注释使用)
        # return max(T0 * (1 - epoch / epochs), T_min)

        # 策略3: 对数衰减 (取消注释使用)
        # return T0 / (1 + cooling_factor * epoch)

    # 迭代参数
    epochs = 5000 #迭代次数
    sigma_required_gradient = False  # 用户自定义key

    # Tikhonov正则化参数
    alpha_tikhonov_eps = 0.015 # epsilon的Tikhonov正则化参数
    alpha_tikhonov_sig = 0  # sigma的Tikhonov正则化参数
    alpha_SA = 0.01 #模拟退火参数

    # 源点和接收点设置 - 注意：这些坐标是相对于内部网格的，不包括PML边界
    source_list = [(0, i) for i in range(10, zl-10, 5)]
    print('source_list:', source_list)
    receiver_list = source_list.copy()

    os.makedirs('results', exist_ok=True)

    # ========== 2. 生成观测数据 ==========
    print("生成观测数据...")

    d_obs = forward_model(epsilon_true, sigma_true, source_list, receiver_list, dt, dx, dz, npml, freq, steps, save_wavefield=False)
    np.save('results/d_obs.npy', d_obs)
    plt.figure()
    plt.imshow(d_obs.T, aspect='auto', cmap='seismic')
    plt.title('B-scan d_obs')
    plt.colorbar()
    plt.savefig('results/obs_data.png')
    plt.close()

    # ========== 3. FWI主循环 ==========
    model_eps = epsilon0.copy()
    model_sig = sigma0.copy()
    loss_list = []

    #设置Adam超参数
    beta_1 = 0.9 #一阶矩
    beta_2 = 0.999 #二阶矩
    alpha_eps = 0.02
    alpha_sig = 1e-4
    eps = 1e-8 #数值稳定性常数

    #初始化Moment vector
    w_eps = np.zeros_like(model_eps.flatten())
    w_sig = np.zeros_like(model_sig.flatten())
    v_eps = np.zeros_like(model_eps.flatten())
    v_sig = np.zeros_like(model_sig.flatten())

    def acceptance_probability(old_loss, new_loss, temperature):
        if new_loss < old_loss:
            return 1.0
        else:
            ### ?如果做幅值修改呢？
            delta = (old_loss - new_loss)
            return np.exp(delta / temperature)

    def check_nan_inf(arr, name):
        if np.any(np.isnan(arr)):
            print(f"[警告] {name} 出现nan!")
        if np.any(np.isinf(arr)):
            print(f"[警告] {name} 出现inf!")
        print(f"[{name}] min={np.nanmin(arr)}, max={np.nanmax(arr)}, mean={np.nanmean(arr)}")

    random_activate = True
    for epoch in range(epochs):
        epoch_start_time = time.time()  # 记录每个epoch的开始时间
        T=get_current_temperature(epoch)
        print(f"Epoch {epoch+1}/{epochs}, 当前温度: {T:.6f}")
        w_eps_new = w_eps.copy()
        w_sig_new = w_sig.copy()
        v_eps_new = v_eps.copy()
        v_sig_new = v_sig.copy()
        
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
        selected_receiver_list = receiver_list[selected_indices]
        print('current selected_receiver_list:', selected_receiver_list)
        selected_d_obs = d_obs[selected_indices,:,:]
        # 正演模拟（保存波场数据用于梯度计算）
        print("  正演模拟...")
        d_syn, wavefield_data = forward_model(model_eps, model_sig, selected_source_list, selected_receiver_list, dt, dx, dz, npml, freq, steps, save_wavefield=True)
        #check_nan_inf(d_syn.T, f"d_syn (epoch {epoch})")
        # np.save(f'results/d_syn_epoch{epoch}.npy', d_syn)
        
        # plt.figure()
        # plt.imshow(d_syn, aspect='auto', cmap='seismic')
        # plt.title(f'Simulated data d_syn (epoch {epoch})')
        # plt.colorbar()
        # plt.savefig(f'results/syn_data_epoch{epoch}.png')
        # plt.close()

        # 残差
        residual = d_syn - selected_d_obs
        # check_nan_inf(residual, f"residual (epoch {epoch})")
        np.save(f'results/residual_epoch{epoch}.npy', residual)
        # plt.figure()
        # plt.imshow(residual.T, aspect='auto', cmap='bwr')
        # plt.title(f'Residual (epoch {epoch})')
        # plt.colorbar()
        # plt.savefig(f'results/residual_epoch{epoch}.png')
        # plt.close()

        # 计算损失
        loss = np.linalg.norm(residual)
        loss_list.append(loss)
        print(f"  Loss: {loss:.6f}")

        # 梯度计算
        print("  计算梯度...")
        grad_eps, grad_sig= compute_gradient(model_eps, model_sig, residual, selected_source_list, selected_receiver_list, dt, dx, dz, npml, freq, steps, sigma_required_gradient, wavefield_data, alpha_tikhonov_eps, alpha_tikhonov_sig)
        check_nan_inf(grad_eps, f"grad_eps (epoch {epoch})")

        #np.save(f'results/grad_eps_epoch{epoch}.npy', grad_eps)
        plt.figure()
        plt.imshow(grad_eps, cmap='viridis')
        plt.title(f'epsilon gradient (epoch {epoch})')
        plt.colorbar()
        plt.savefig(f'results/update/grad_eps_epoch{epoch}.png')
        plt.close()
        
        if sigma_required_gradient and grad_sig is not None:
            np.save(f'results/grad_sig_epoch{epoch}.npy', grad_sig)
            plt.figure()
            plt.imshow(grad_sig, cmap='viridis')
            plt.title(f'sigma gradient (epoch {epoch})')
            plt.colorbar()
            plt.savefig(f'results/grad_sig_epoch{epoch}.png')
            plt.close()

        # adam矩更新
        print("  adam矩更新和adam更新量计算中...")
        g_eps = grad_eps.flatten()
        w_eps_new = beta_1 * w_eps+(1-beta_1)*g_eps
        v_eps_new = beta_2 * v_eps+(1-beta_2)*g_eps**2
        w_eps_hat = w_eps_new/(1-beta_1**(epoch+1))
        v_eps_hat = v_eps_new/(1-beta_2**(epoch+1))
        update_eps_grad = alpha_eps*w_eps_hat/(np.sqrt (v_eps_hat)+eps)
        print(f"Adam梯度eps更新量update_grad: max={np.max(update_eps_grad)}, min={np.min(update_eps_grad)}, mean={np.mean(update_eps_grad)}")

        #np.save(f'results/update_eps_grad_epoch{epoch}.npy', update_eps_grad)
        plt.figure()
        plt.imshow(update_eps_grad.reshape(model_eps.shape), cmap='jet')
        plt.title(f'epsilon update grad (epoch {epoch})')
        plt.colorbar()
        plt.savefig(f'results/update/update_eps_grad_epoch{epoch}.png')
        plt.close()
            
        ########################################
        #计算正则化梯度部分
        ########################################

        print('Tikhonov正则化更新量计算中...')
        # 添加二阶Tikhonov正则项梯度
        tikhonov_grad_eps = compute_tikhonov_gradient(model_eps.copy(), dx, dz, alpha_tikhonov_eps)
        tikhonov_grad_eps[:20,:] = 0
        print(f"Tikhonov epsilon梯度: max={np.max(tikhonov_grad_eps)}, min={np.min(tikhonov_grad_eps)}, mean={np.mean(tikhonov_grad_eps)}")
        #np.save(f'results/tikhonov_grad_eps_epoch{epoch}.npy', tikhonov_grad_eps)
        plt.figure()
        plt.imshow(tikhonov_grad_eps, cmap='jet')
        plt.title(f'epsilon tikhonov grad (epoch {epoch})')
        plt.colorbar()
        plt.savefig(f'results/update/tikhonov_grad_eps_epoch{epoch}.png')
        plt.close()
        
        if sigma_required_gradient:
            g_sig = grad_sig.flatten()
            w_sig_new = beta_1 * w_sig+(1-beta_1)*g_sig
            v_sig_new = beta_2 * v_sig+(1-beta_2)*g_sig**2
            w_sig_hat = w_sig_new/(1-beta_1**(epoch+1))
            v_sig_hat = v_sig_new/(1-beta_2**(epoch+1))
            update_sig_grad = w_sig_hat/(np.sqrt(v_sig_hat)+eps)
            print(f"Adam梯度sig更新量update_grad: max={np.max(update_sig_grad)}, min={np.min(update_sig_grad)}, mean={np.mean(update_sig_grad)}")
            np.save(f'results/update_sig_grad_epoch{epoch}.npy', update_sig_grad)
            plt.figure()
            plt.imshow(update_sig_grad.reshape(model_sig.shape), cmap='viridis')
            plt.title(f'sigma update grad (epoch {epoch})')
            plt.colorbar()
            plt.savefig(f'results/update_sig_grad_epoch{epoch}.png')
            plt.close()
            print('Tikhonov正则化sigma更新量计算中...')
            tikhonov_grad_sig = compute_tikhonov_gradient(model_sig.copy(), dx, dz, alpha_tikhonov_sig)
            tikhonov_grad_sig[:20,0] = 0
            print(f"Tikhonov sigma梯度: max={np.max(tikhonov_grad_sig)}, min={np.min(tikhonov_grad_sig)}, mean={np.mean(tikhonov_grad_sig)}")
            np.save(f'results/tikhonov_grad_sig_epoch{epoch}.npy', tikhonov_grad_sig)
            plt.figure()
            plt.imshow(tikhonov_grad_sig, cmap='viridis')
            plt.title(f'sigma tikhonov grad (epoch {epoch})')
            plt.colorbar()
            plt.savefig(f'results/tikhonov_grad_sig_epoch{epoch}.png')
            plt.close()

        #########################################
        # 模拟退火随机扰动部分
        #########################################

        print('模拟退火随机扰动部分...')
        update_eps = update_eps_grad.copy().flatten()
        new_eps = model_eps.copy().flatten()
        tikhonov_grad_eps = tikhonov_grad_eps.flatten()

        u = np.random.uniform(0, 1, size=model_eps.flatten().shape)
        sign = T * np.sign(u-0.5)
        abs = (1+1/T)**(np.abs(u-0.5))-1
        y = sign * abs
        
        update_SA = alpha_SA * y * delta_eps
        print(f"update_SA: max={np.max(update_SA)}, min={np.min(update_SA)}, mean={np.mean(update_SA)}")
        
        # 将update_SA重塑为2D数组，固定浅层部分为0，然后重新展平
        update_SA_2d = update_SA.reshape(model_eps.shape)
        update_SA_2d[:20, :] = 0  # 固定浅层部分为0

        plt.figure(figsize=(15, 5))  # 设置图像大小
        plt.subplot(1, 3, 1)  # 修改为1行3列的子图布局
        plt.imshow(update_SA_2d, cmap='jet')
        plt.title(f'epsilon update SA (epoch {epoch})')
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.imshow(-update_eps_grad.reshape(model_eps.shape) + update_SA_2d, cmap='jet')
        plt.title(f'epsilon update grad+SA (epoch {epoch})')
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.imshow(update_eps_grad.reshape(model_eps.shape) + update_SA_2d - tikhonov_grad_eps.reshape(model_eps.shape), cmap='jet')
        plt.title(f'epsilon update grad+SA+tikhonov (epoch {epoch})')
        plt.colorbar()
        plt.tight_layout()  # 自动调整子图参数以填充整个图像区域
        plt.savefig(f'results/update/update_SA_epoch{epoch}_3s.png')
        plt.close()
        

        update_SA = update_SA_2d.flatten()
        
        new_eps = new_eps - update_eps + update_SA - tikhonov_grad_eps
        
        # 将new_eps中小于1的值调整为1
        new_eps = np.maximum(new_eps, 1)
        
        print('计算新模型loss')
        d_syn_new = forward_model(new_eps.reshape(model_eps.shape), model_sig, source_list, receiver_list, dt, dx, dz, npml, freq, steps, save_wavefield=False)

        loss_new = np.linalg.norm((d_syn_new-d_obs).flatten())
        print(f'新模型loss: {loss_new}')
        delta_loss = loss_new - loss
        print(f'delta_loss: {delta_loss}')
        if delta_loss <=0:
            model_eps = new_eps.reshape(model_eps.shape)
            w_eps = w_eps_new.copy()
            v_eps = v_eps_new.copy()
            print(f"新Loss较小，更新模型参数")
        else:
            Probability = acceptance_probability(loss, loss_new, T)
            r = np.random.uniform(0,1)
            print(f'当前接受概率: {Probability}, 随机数: {r}')
            if Probability > r:
                model_eps = new_eps.reshape(model_eps.shape)
                w_eps = w_eps_new.copy()
                v_eps = v_eps_new.copy()
                print(f"新Loss较大, 依照概率接受新参数")
            else:
                print(f"新Loss较大, 依照概率不接受新参数")
        #限制范围>=1
        model_eps = np.maximum(model_eps, 1)
        check_nan_inf(model_eps, f"model_eps (epoch {epoch})")
        np.save(f'results/model_eps_epoch{epoch}.npy', model_eps)
        
        #画图保存
        plt.figure()
        plt.imshow(model_eps, cmap='jet',vmin=0,vmax=10)
        plt.title(f'epsilon model (epoch {epoch})')
        plt.colorbar()
        plt.savefig(f'results/model_eps_epoch{epoch}.png')
        plt.close()

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

if __name__ == '__main__':
    main() 
    import numpy as np
    import os

    # 获取所有模型数据文件
    model_files = [f for f in os.listdir('results') if f.startswith('model_eps_epoch') and f.endswith('.npy')]
    model_files.sort(key=lambda x: int(x.split('epoch')[1].split('.npy')[0]))

    # 读取最后一千次迭代的模型数据文件
    last_1000_files = model_files[-1000:]
    models = [np.load(os.path.join('results', f)) for f in last_1000_files]

    # 计算平均值
    average_model = np.mean(models, axis=0)

    # 保存平均模型
    np.save('results/average_model_eps.npy', average_model)

    # 可视化平均模型
    plt.figure()
    plt.imshow(average_model, cmap='jet', vmin=0, vmax=10)
    plt.title('Average epsilon model (last 1000 epochs)')
    plt.colorbar()
    plt.savefig('results/average_model_eps.png')
    plt.close()