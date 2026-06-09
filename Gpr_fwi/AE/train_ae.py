 # AutoEncoder训练脚本
# 用于训练AE模型还原输入的介电常数分布模型

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import os
import random
import time
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# 设置matplotlib中文字体
def setup_chinese_font():
    """设置中文字体显示"""
    import platform
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统字体设置
        font_list = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'SimSun']
    elif system == 'Darwin':  # macOS
        # macOS系统字体设置
        font_list = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux
        # Linux系统字体设置
        font_list = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'Liberation Sans']
    
    # 尝试设置字体
    for font in font_list:
        try:
            matplotlib.rcParams['font.sans-serif'] = [font] + matplotlib.rcParams['font.sans-serif']
            break
        except:
            continue
    
    matplotlib.rcParams['axes.unicode_minus'] = False
    print(f"已设置中文字体: {matplotlib.rcParams['font.sans-serif'][0]}")

# 调用字体设置函数
setup_chinese_font()

# 导入AutoEncoder模型
from unet import AutoEncoder

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

def generate_random_model(height=100, width=200, surface_layers=20, min_eps=1.0, max_eps=20.0):
    """
    生成随机的介电常数分布模型
    
    Args:
        height: 深度方向大小
        width: 横向大小
        surface_layers: 表层固定层数
        min_eps: 最小介电常数
        max_eps: 最大介电常数
    
    Returns:
        model: 介电常数分布模型 [height, width]
    """
    # 生成随机模型
    model = np.random.uniform(min_eps, max_eps, (height, width))
    
    # 表层20层固定为均匀介质（介电常数为3）
    model[:surface_layers, :] = 3.0
    
    # 添加一些地质结构特征
    # 1. 添加一些水平层状结构
    for i in range(3):
        layer_start = surface_layers + np.random.randint(10, height - surface_layers - 20)
        layer_thickness = np.random.randint(5, 15)
        layer_value = np.random.uniform(min_eps, max_eps)
        model[layer_start:layer_start+layer_thickness, :] = layer_value
    
    # 2. 添加一些异常体
    num_anomalies = np.random.randint(2, 6)
    for _ in range(num_anomalies):
        # 随机位置和大小
        center_x = np.random.randint(20, width-20)
        center_z = np.random.randint(surface_layers+10, height-20)
        size_x = np.random.randint(10, 30)
        size_z = np.random.randint(5, 15)
        
        # 确保不超出边界
        start_x = max(0, center_x - size_x//2)
        end_x = min(width, center_x + size_x//2)
        start_z = max(surface_layers, center_z - size_z//2)
        end_z = min(height, center_z + size_z//2)
        
        # 设置异常体值
        anomaly_value = np.random.uniform(min_eps, max_eps)
        model[start_z:end_z, start_x:end_x] = anomaly_value
    
    # 3. 添加高斯平滑（只对深层部分进行平滑）
    deep_part = model[surface_layers:, :]
    deep_part_smoothed = gaussian_filter(deep_part, sigma=1.0)
    model[surface_layers:, :] = deep_part_smoothed
    
    # 确保浅层部分保持固定值
    model[:surface_layers, :] = 3.0
    
    return model

def create_dataset(num_samples=1000, height=100, width=200, surface_layers=20):
    """
    创建训练数据集
    
    Args:
        num_samples: 样本数量
        height: 深度方向大小
        width: 横向大小
        surface_layers: 表层固定层数
    
    Returns:
        dataset: 数据集
    """
    print(f"正在生成 {num_samples} 个训练样本...")
    
    models = []
    for i in tqdm(range(num_samples)):
        model = generate_random_model(height, width, surface_layers)
        models.append(model)
    
    # 转换为torch张量
    models_tensor = torch.tensor(np.array(models), dtype=torch.float32)
    
    print(f"数据集创建完成，形状: {models_tensor.shape}")
    return models_tensor

class ModelDataset(data_utils.Dataset):
    def __init__(self, models):
        self.models = models
    
    def __len__(self):
        return len(self.models)
    
    def __getitem__(self, idx):
        return self.models[idx]

def plot_model_distribution(models, save_path, title="模型分布直方图"):
    """绘制模型分布直方图"""
    plt.figure(figsize=(10, 6))
    plt.hist(models.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('介电常数')
    plt.ylabel('频数')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_samples(models, save_path, num_samples=4):
    """绘制训练样本示例"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(models))):
        im = axes[i].imshow(models[i], aspect='auto', cmap='jet', vmin=1, vmax=20)
        axes[i].set_title(f'训练样本 {i+1}')
        axes[i].set_xlabel('横向位置')
        axes[i].set_ylabel('深度')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 设置随机种子
    same_seeds(2025)
    
    # 创建结果文件夹
    os.makedirs('results_ae', exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 参数设置
    height, width = 100, 200
    surface_layers = 20
    num_samples = 1000
    batch_size = 16
    num_epochs = 1000
    learning_rate = 1e-3
    
    # 创建数据集
    models = create_dataset(num_samples, height, width, surface_layers)
    
    # 绘制训练样本分布
    plot_model_distribution(models.numpy(), 'results_ae/training_models_distribution.png', 
                           "训练集模型分布直方图")
    
    # 绘制训练样本示例
    plot_training_samples(models.numpy(), 'results_ae/training_samples.png')
    
    # 创建数据加载器
    dataset = ModelDataset(models)
    train_loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化AutoEncoder
    print("初始化AutoEncoder...")
    model = AutoEncoder(in_channels=1, out_channels=1).to(device)
    
    # 计算padding
    target_height, target_width = height, width
    required_height = ((target_height + 31) // 32) * 32
    required_width = ((target_width + 31) // 32) * 32
    
    pad_height = required_height - target_height
    pad_width = required_width - target_width
    
    if pad_height % 2 == 1:
        pad_height += 1
    if pad_width % 2 == 1:
        pad_width += 1
    
    print(f"padding: height={pad_height}, width={pad_width}")
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    # 训练记录
    loss_history = []
    
    # 训练循环
    print("开始训练...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        for batch_idx, batch_models in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 数据预处理：padding和维度扩展
            batch_models = batch_models.to(device)  # [batch_size, height, width]
            
            # 对称padding
            batch_models_padded = F.pad(batch_models, (pad_width//2, pad_width//2, pad_height//2, pad_height//2))
            
            # 扩展维度
            batch_models_input = batch_models_padded.unsqueeze(1)  # [batch_size, 1, height+pad_height, width+pad_width]
            
            # 前向传播
            outputs = model(batch_models_input)  # [batch_size, 100, 200]
            
            # 应用sigmoid激活函数，然后乘10再加1
            outputs_processed = torch.sigmoid(outputs) * 10 + 1
            
            # 创建固定的浅层部分（介电常数为3）
            current_batch_size = batch_models.shape[0]
            uniform_layer = torch.ones((current_batch_size, 20, 200), device=device) * 3.0
            
            # 拼接浅层和深层
            outputs_processed = torch.cat([uniform_layer, outputs_processed], dim=1)
            
            # 计算损失
            loss = loss_fn(outputs_processed, batch_models)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Time: {elapsed_time:.2f}s')
        
        # 每20个epoch保存一次模型和可视化结果
        # if (epoch + 1) % 20 == 0:
        #     # 保存模型
        #     torch.save(model.state_dict(), f'results_ae/ae_model_epoch_{epoch+1}.pth')
            
        #     # 可视化一些结果
        #     model.eval()
        #     with torch.no_grad():
        #         # 选择第一个batch进行可视化
        #         test_models = batch_models[:4]  # 取前4个样本
        #         test_outputs = outputs_processed[:4]
                
        #         fig, axes = plt.subplots(4, 2, figsize=(12, 16))
                
        #         for i in range(4):
        #             # 原始模型
        #             im1 = axes[i, 0].imshow(test_models[i].cpu().numpy(), aspect='auto', cmap='jet', vmin=1, vmax=20)
        #             axes[i, 0].set_title(f'原始模型 {i+1}')
        #             axes[i, 0].set_xlabel('横向位置')
        #             axes[i, 0].set_ylabel('深度')
        #             plt.colorbar(im1, ax=axes[i, 0])
                    
        #             # 重建模型
        #             im2 = axes[i, 1].imshow(test_outputs[i].cpu().numpy(), aspect='auto', cmap='jet', vmin=1, vmax=20)
        #             axes[i, 1].set_title(f'重建模型 {i+1}')
        #             axes[i, 1].set_xlabel('横向位置')
        #             axes[i, 1].set_ylabel('深度')
        #             plt.colorbar(im2, ax=axes[i, 1])
                
        #         plt.tight_layout()
        #         plt.savefig(f'results_ae/reconstruction_epoch_{epoch+1}.png', dpi=300, bbox_inches='tight')
        #         plt.close()
    
    # 训练完成，保存最终模型
    torch.save(model.state_dict(), 'results_ae/ae_model_final.pth')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.title('AutoEncoder训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results_ae/training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存损失历史
    np.save('results_ae/loss_history.npy', np.array(loss_history))
    
    print("训练完成！")
    
    # 输出网络输出的数据分布直方图
    print("生成网络输出分布直方图...")
    model.eval()
    with torch.no_grad():
        # 使用所有训练数据进行测试
        all_outputs_raw = []
        for batch_models in train_loader:
            batch_models = batch_models.to(device)
            batch_models_padded = F.pad(batch_models, (pad_width//2, pad_width//2, pad_height//2, pad_height//2))
            batch_models_input = batch_models_padded.unsqueeze(1)
            deep_outputs_raw = model(batch_models_input)  # 只输出深层部分，未经过sigmoid
            all_outputs_raw.append(deep_outputs_raw.cpu().numpy())
        
        all_outputs_raw = np.concatenate(all_outputs_raw, axis=0)
        
        # 绘制分布直方图
        plt.figure(figsize=(12, 8))
        
        plt.hist(all_outputs_raw.flatten(), bins=50, alpha=0.7, color='blue', label='network output')
        plt.title('network output distribution')
        plt.xlabel('epsilon')
        plt.ylabel('frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results_ae/network_output_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    print("所有结果已保存到 results_ae/ 文件夹")

if __name__ == "__main__":
    main()