# GPR电磁波双参数FWI训练脚本
# 基于声波train.py改编的GPR版本

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.utils.data as data_utils
import os
import shutil
import sys
import random
import time
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from forward import forward_model_simul
from gradient import compute_gradient_simul, compute_tikhonov_gradient,compute_laplacian_2d
from unet import UNet

# 设置PyTorch默认数据类型
torch.set_default_dtype(torch.float32)

# ========== 自动微分算子定义 ==========
class ForwardModelFunction(Function):
    """
    自定义自动微分函数，包装FDTD正演模拟
    """
    @staticmethod
    def forward(ctx, epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps):
        """
        前向传播：执行FDTD正演模拟
        """
        # 将torch张量转换为numpy数组用于FDTD计算
        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()
        
        # 执行正演模拟
        data, wave_field = forward_model_simul(
            epsilon_np, sigma_np, source_list, receiver_list,
            dt, dx, dz, npml, freq, steps, save_wavefield=True
        )
        
        # 将结果转换回torch张量，确保是float32类型
        data_tensor = torch.from_numpy(data).to(epsilon.device).float()
        
        # 保存用于反向传播的信息
        ctx.save_for_backward(epsilon, sigma)
        ctx.source_list = source_list
        ctx.receiver_list = receiver_list
        ctx.dt = dt
        ctx.dx = dx
        ctx.dz = dz
        ctx.npml = npml
        ctx.freq = freq
        ctx.steps = steps
        ctx.wave_field = wave_field

        return data_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：计算梯度
        """
        epsilon, sigma = ctx.saved_tensors
        source_list = ctx.source_list
        receiver_list = ctx.receiver_list
        dt = ctx.dt
        dx = ctx.dx
        dz = ctx.dz
        npml = ctx.npml
        freq = ctx.freq
        steps = ctx.steps
        wave_field = ctx.wave_field
        
        # 将torch张量转换为numpy数组
        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()
        grad_output_np = grad_output.detach().cpu().numpy()
        
        # 计算梯度
        grad_eps, grad_sig = compute_gradient_simul(
            epsilon_np, sigma_np, grad_output_np,
            source_list, receiver_list, dt, dx, dz, npml,
            freq, steps, sigma_required_gradient=False, wavefield_data=wave_field
        )
        
        # 转换回torch张量
        grad_eps_tensor = torch.from_numpy(grad_eps).to(epsilon.device)
        grad_sig_tensor = torch.from_numpy(grad_sig).to(sigma.device) if grad_sig is not None else None
        
        return grad_eps_tensor, grad_sig_tensor, None, None, None, None, None, None, None, None

class TikhonovRegularizationFunction(Function):
    """
    自定义Tikhonov正则化函数，利用gradient.py中的compute_tikhonov_gradient
    """
    @staticmethod
    def forward(ctx, epsilon, sigma, alpha_tikhonov_eps, alpha_tikhonov_sig, dx, dz):
        """
        前向传播：计算Tikhonov正则化损失
        """
        # 将torch张量转换为numpy数组以使用gradient.py中的函数
        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()
        
        # 使用gradient.py中的函数计算拉普拉斯算子
        laplacian_eps_np = compute_laplacian_2d(epsilon_np, dx, dz)
        laplacian_sig_np = compute_laplacian_2d(sigma_np, dx, dz)
        
        # 转换回torch张量
        laplacian_eps = torch.from_numpy(laplacian_eps_np).to(epsilon.device)
        laplacian_sig = torch.from_numpy(laplacian_sig_np).to(sigma.device)
        
        # 计算正则化损失
        reg_loss_eps = 1e-10 * alpha_tikhonov_eps * torch.sum(laplacian_eps**2)
        reg_loss_sig = 1e-10 * alpha_tikhonov_sig * torch.sum(laplacian_sig**2)
        
        # 总的正则化损失
        reg_loss = reg_loss_eps + reg_loss_sig
        
        # 保存用于反向传播的信息
        ctx.save_for_backward(epsilon, sigma, laplacian_eps, laplacian_sig)
        ctx.alpha_tikhonov_eps = alpha_tikhonov_eps
        ctx.alpha_tikhonov_sig = alpha_tikhonov_sig
        ctx.dx = dx
        ctx.dz = dz
        
        return reg_loss
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：使用gradient.py中的compute_tikhonov_gradient
        """
        epsilon, sigma, laplacian_eps, laplacian_sig = ctx.saved_tensors
        alpha_tikhonov_eps = ctx.alpha_tikhonov_eps
        alpha_tikhonov_sig = ctx.alpha_tikhonov_sig
        dx = ctx.dx
        dz = ctx.dz
        
        # 将torch张量转换为numpy数组
        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()
        
        # 使用gradient.py中的函数计算Tikhonov梯度
        tikhonov_grad_eps = compute_tikhonov_gradient(epsilon_np, dx, dz, alpha_tikhonov_eps)
        tikhonov_grad_sig = compute_tikhonov_gradient(sigma_np, dx, dz, alpha_tikhonov_sig)
        
        # 转换回torch张量
        grad_eps = torch.from_numpy(tikhonov_grad_eps).to(epsilon.device) * grad_output
        grad_sig = torch.from_numpy(tikhonov_grad_sig).to(sigma.device) * grad_output
        
        return grad_eps, grad_sig, None, None, None, None

def tikhonov_reg(epsilon, sigma, alpha_tikhonov_eps, alpha_tikhonov_sig, dx=0.02, dz=0.02):
    """
    计算Tikhonov正则化损失项，使用自定义自动微分函数
    """
    return TikhonovRegularizationFunction.apply(epsilon, sigma, alpha_tikhonov_eps, alpha_tikhonov_sig, dx, dz)

# ========== 辅助函数 ==========
def grad_hook(grad):
    """梯度处理钩子函数"""
    # 检查梯度是否包含nan或inf
    if torch.isnan(grad).any() or torch.isinf(grad).any():
        print("警告：梯度包含nan或inf值！")
        grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 浅层掩码：前20行梯度设为0
    grad[:20, :] = grad[:20, :] * 0
    
    # 梯度归一化
    max_abs_value = torch.max(torch.abs(grad)).item()
    if max_abs_value != 0:
        grad /= max_abs_value
    
    # 梯度裁剪：限制梯度范围
    grad = torch.clamp(grad, -1.0, 1.0)
    
    return grad

def get_parameter_number(net):
    """获取网络参数数量"""
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def plot_loss_curves(loss_history, eps_loss_history, save_path, current_epoch):
    """绘制损失曲线"""
    plt.figure(figsize=(15, 5))
    
    # 完整损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.title('Full Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 最近100个epoch的损失趋势
    plt.subplot(1, 3, 2)
    recent_epochs = min(100, len(loss_history))
    if recent_epochs > 0:
        plt.plot(range(len(loss_history)-recent_epochs, len(loss_history)), 
                loss_history[-recent_epochs:], 'r-', linewidth=2)
        plt.title(f'Recent {recent_epochs} Epochs Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    # Epsilon模型损失
    plt.subplot(1, 3, 3)
    plt.plot(eps_loss_history, 'g-', linewidth=2)
    plt.title('Epsilon Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}loss_curves_epoch_{current_epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()



def same_seeds(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ========== 数据加载器 ==========
class MyDataset(data_utils.Dataset):
    def __init__(self, seismic_data, source_locations, receiver_locations):
        self.seismic_data = seismic_data
        self.source_locations = source_locations
        self.receiver_locations = receiver_locations
    
    def __getitem__(self, item):
        return self.seismic_data[item], self.source_locations[item], self.receiver_locations[item]
    
    def __len__(self):
        return len(self.seismic_data)


# ========== 主程序 ==========
def main():
    # 设置随机种子
    same_seeds(2025)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # 删除并创建结果文件夹
    if os.path.exists('result_gpr'):
        shutil.rmtree('result_gpr')
    os.makedirs('result_gpr', exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")


    # ========== 训练设置 ==========
    learning_rate = 1e-3  # 学习率
    alpha_tikhonov_eps = 0.015  # Tikhonov正则化参数
    alpha_tikhonov_sig = 0
    offset=0
    # 构造结果文件夹名，包含学习率和alpha_tikhonov_eps
    result_dir = f'result_gpr_{offset}_lr{learning_rate:.0e}_alpha{alpha_tikhonov_eps:.3g}_mode1_obs/'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    path = result_dir  # 后续保存都用 path 变量

    # ========== 参数设置 ==========
    # 网格参数
    xl, zl = 100, 200  # 深度x横向长度
    dx, dz = 0.02, 0.02
    dt = 4e-11
    npml = 10
    steps = 1000
    
    # 震源参数
    freq = 4e8
    
    # 真实模型参数
    epsilon_true = np.ones((xl, zl)) * 3
    epsilon_true[40:60, 90:110] = 1.5
    epsilon_true[85:, :] = 6
    sigma_true = np.ones((xl, zl)) * 0
    
    # 转换为torch张量
    epsilon_true = torch.tensor(epsilon_true, dtype=torch.float32, device=device)
    sigma_true = torch.tensor(sigma_true, dtype=torch.float32, device=device)
    
    # 确保所有张量都是float32类型
    torch.set_default_dtype(torch.float32)
    
    # 参数范围
    min_eps, max_eps = 1.0, 6.0
    min_sig, max_sig = 1e-6, 1e-3
    
    # 移除未使用的浅层固定参数
    l_size = 20  # 浅层固定深度
    
    # 源点和接收点设置
    source_list = [(0, i) for i in range(10, zl-10, 2)]  
    receiver_list = source_list.copy()
    
    print(f"源点数量: {len(source_list)}")
    print(f"接收点数量: {len(receiver_list)}")
    print(f"源点位置: {source_list}")
    print(f"接收点位置: {receiver_list}")
    
    # ========== 读取观测数据 ==========
    print("读取观测数据文件 d_obs_mode1.npy...")
    obs_path = os.path.join(os.path.dirname(__file__), 'd_obs_mode1.npy')
    d_obs_np = np.load(obs_path)
    d_obs = torch.tensor(d_obs_np, dtype=torch.float32, device=device)
    print(f"观测数据形状: {d_obs.shape}")
    print(f"观测数据类型: {d_obs.dtype}")
    plt.figure(figsize=(12, 4))
    plt.imshow(d_obs.detach().cpu().numpy().T, aspect='auto', cmap='seismic')
    plt.title('Observed Data (d_obs)')
    plt.colorbar(); plt.tight_layout()
    plt.savefig(f"{path}d_obs.png"); plt.close()

    
    # ========== 数据加载器设置 ==========
    # 创建数据集 - 确保观测数据是3D张量 (n_shots, n_receivers, n_steps)
    if d_obs.dim() == 4:
        d_obs = d_obs.squeeze(0)  # 移除可能的batch维度
    
    dataset = MyDataset([d_obs], [source_list], [receiver_list])
    train_loader = data_utils.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # ========== UNet网络设置 ==========
    print("初始化UNet网络...")
    net_1 = UNet(in_channels=1).to(device)  # 介电常数网络
    # net_2 = UNet(in_channels=1).to(device)  # 电导率网络
    print(f"网络参数数量: {get_parameter_number(net_1)}")
    
    # ========== 网络输入处理 ==========
    # 对真实模型进行高斯滤波处理
    eps_input = torch.tensor(1/gaussian_filter(1/epsilon_true.cpu().numpy(), 5), dtype=torch.float32)
    
    # 计算padding
    target_height, target_width = 100, 200
    required_height = ((target_height + 31) // 32) * 32
    required_width = ((target_width + 31) // 32) * 32
    
    pad_height = required_height - target_height
    pad_width = required_width - target_width
    
    if pad_height % 2 == 1:
        pad_height += 1
    if pad_width % 2 == 1:
        pad_width += 1
    
    print(f"padding: height={pad_height}, width={pad_width}")
    
    # 对称padding
    eps_input_padded = F.pad(eps_input, (pad_width//2, pad_width//2, pad_height//2, pad_height//2))
    
    # 扩展维度
    net1_input = eps_input_padded.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, height+pad_height, width+pad_width]
    
    # ========== 迁移项设置 ==========
    epsilon_mig = torch.zeros_like(epsilon_true, requires_grad=True)
    sigma_mig = torch.zeros_like(sigma_true, requires_grad=True)
    
    # 移除未使用的浅层固定模型变量
    epsilon_layer = epsilon_true[:l_size, :].clone()
    sigma_layer = sigma_true[:l_size, :].clone()
    
    # ========== 训练设置 ==========
    optimizer = torch.optim.Adam([
        {'params': net_1.parameters(), 'lr': learning_rate},
    ])
    # #scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[2000, 4000, 6000, 8000, 10000], gamma=0.5
    # )
    loss_fn = torch.nn.MSELoss()
    num_epochs = 10001
    
    
    # 损失记录
    loss_history = []
    eps_loss_history = []
    
    # 训练循环
    print("开始训练...")
    random_activate = False
    for epoch in range(num_epochs):
        net_1.train()
        epoch_loss = 0
        total_samples = 0
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
        selected_receiver_list = selected_source_list.copy()
        print('current selected_receiver_list:', selected_receiver_list)
        selected_d_obs = d_obs[selected_indices,:]
        print('current selected_d_obs shape:', selected_d_obs.shape)

        
        for i, (s_d_i, x_s_i, x_r_i) in enumerate(train_loader):
            batch_size_now = s_d_i.size(0)
            optimizer.zero_grad()
            
            # 网络前向传播
            batch_net1_input = net1_input[:, i:i+1, :, :].to(device)
            _, _, net1_output_model = net_1(batch_net1_input)
            
            # Sigmoid归一化
            max_value = net1_output_model.max().item()
            min_value = net1_output_model.min().item()
            mean_value = net1_output_model.mean().item()
            print(f"net1_output_model 统计量: 最大值={max_value}, 最小值={min_value}, 平均值={mean_value}")

            net1_ = torch.sigmoid(net1_output_model+offset)
            
            # 缩放到物理范围
            net1_o = net1_ * (max_eps - min_eps) + min_eps
            
            # 连接浅层固定模型和网络输出
            epsilon = torch.cat((epsilon_layer, net1_o), dim=0)  # 在深度维度连接
            
            # 添加迁移项
            outputs_eps = epsilon + epsilon_mig
            # # 保持浅层=真实模型
            outputs_eps[:l_size, :] = epsilon_true[:l_size, :]
            # 应用clamp但保持梯度流
            outputs_eps = torch.clamp(outputs_eps, min=min_eps, max=max_eps)
            # 确保梯度能够传播（在clamp之后）
            outputs_eps.retain_grad()
            
            # 使用自动微分正演模拟
            print(f"Epoch {epoch}, Batch {i}: 开始正演模拟...")
        
            d_syn = ForwardModelFunction.apply(
                outputs_eps, sigma_true, selected_source_list, selected_receiver_list, 
                dt, dx, dz, npml, freq, steps
            )
            # 确保数据类型一致 - 都转换为float32
            d_syn = d_syn.float()
            selected_d_obs = selected_d_obs.float()
            
            
            print(f"合成数据形状: {d_syn.shape}")
            print(f"合成数据类型: {d_syn.dtype}")
            print(f"观测数据形状: {selected_d_obs.shape}")
            print(f"观测数据类型: {selected_d_obs.dtype}")
            
            # 计算数据损失
            data_loss = loss_fn(d_syn, selected_d_obs)
            
            # 计算正则化损失
            #reg_loss = tikhonov_reg(outputs_eps, sigma_true, alpha_tikhonov_eps, alpha_tikhonov_sig, dx, dz)
            
            # 总损失
            total_loss = data_loss #+ reg_loss
            
            if np.isnan(float(total_loss.item())):
                raise ValueError('loss is nan while training')
            
            
            # 反向传播
            total_loss.backward(retain_graph=True)
            
            # 在参数更新前保存梯度信息
            if epoch % 50 == 0 or epoch in [1, 2, 3, 5, 10, 50, 100]:
                print(f"Epoch {epoch}: 触发可视化条件")
                
                # 可视化结果
                plt.figure(figsize=(15, 5))
                plt.imshow(outputs_eps.detach().cpu().numpy(), aspect='auto', 
                          cmap='jet', vmin=0, vmax=10)
                plt.title(f'Epsilon (epoch {epoch})')
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(f'{path}epoch_{epoch}_epsilon.png')
                plt.close()
                print(f"Epoch {epoch}: 已保存epsilon图")
                
                # 检查梯度状态
                print(f"Epoch {epoch}: outputs_eps.grad is None: {outputs_eps.grad is None}")
                print(f"Epoch {epoch}: outputs_eps.requires_grad: {outputs_eps.requires_grad}")
                
                # 可视化梯度
                if outputs_eps.grad is not None:
                    print(f"Epoch {epoch}: 开始绘制梯度...")
                    grad_np = outputs_eps.grad.detach().cpu().numpy()
                    print(f"Epoch {epoch}: 梯度形状: {grad_np.shape}")
                    print(f"Epoch {epoch}: 梯度范围: [{np.min(grad_np):.6f}, {np.max(grad_np):.6f}]")
                    
                    # 归一化梯度
                    max_abs_grad = np.max(np.abs(grad_np))
                    if max_abs_grad != 0:
                        grad_np_norm = grad_np / max_abs_grad
                    else:
                        grad_np_norm = grad_np.copy()
                    print(f"Epoch {epoch}: 归一化后梯度范围: [{np.min(grad_np_norm):.6f}, {np.max(grad_np_norm):.6f}]")
                    
                    # 梯度图（归一化后）
                    plt.figure(figsize=(15, 5))
                    plt.imshow(grad_np_norm, aspect='auto', cmap='RdBu_r', 
                              vmin=-1, vmax=1)
                    plt.title(f'Epsilon Gradient (Normed, epoch {epoch})')
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(f'{path}epoch_{epoch}_epsilon_grad.png')
                    plt.close()
                    print(f"Epoch {epoch}: 已保存归一化梯度图")
                    
                    # 保存归一化梯度数据
                    #np.save(f'{path}epoch_{epoch}_epsilon_grad.npy', grad_np_norm)
                    print(f"Epoch {epoch}: 已保存归一化梯度数据")
                else:
                    print(f"Epoch {epoch}: 梯度为None，无法绘制")
                
                # 保存数据
                #np.save(f'{path}epoch_{epoch}_epsilon.npy', outputs_eps.detach().cpu().numpy())
                print(f"Epoch {epoch}: 已保存epsilon数据")
            
            # 参数更新
            optimizer.step()
            #scheduler.step()
            
            epoch_loss += total_loss.item() * batch_size_now
            total_samples += batch_size_now
        
        # 记录损失
        if i == 0:
            epoch_loss = epoch_loss / total_samples
            loss_model_eps = loss_fn(outputs_eps, epsilon_true) / loss_fn(epsilon_true, torch.zeros_like(epsilon_true))
            
            # 记录到列表用于绘图
            loss_history.append(epoch_loss)
            eps_loss_history.append(loss_model_eps.item())
            
            print(f'Epoch {epoch}: Data Loss = {epoch_loss:.6f}, Eps Loss = {loss_model_eps:.6f}')
            
            # 每100个epoch绘制一次loss曲线
            if epoch % 100 == 0 and epoch > 0:
                plot_loss_curves(loss_history, eps_loss_history, path, epoch)
    
    # 训练结束后绘制最终的loss曲线
    print("绘制最终loss曲线...")
    plot_loss_curves(loss_history, eps_loss_history, path, num_epochs-1)
    
    # 保存最终的loss数据
    #np.save(f'{path}final_loss_history.npy', np.array(loss_history))
    #np.save(f'{path}final_eps_loss_history.npy', np.array(eps_loss_history))
    
    print("训练完成！最终loss曲线已保存。")

if __name__ == "__main__":
    main()
