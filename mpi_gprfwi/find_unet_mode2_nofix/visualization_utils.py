#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNet网络层输出可视化工具
用于可视化UNet网络中每一层的输出特征图
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import os

def visualize_unet_layers(layer_outputs, save_path, epoch, max_channels_per_layer=16):
    """
    可视化UNet网络每一层的输出
    
    Args:
        layer_outputs: 字典，包含每层输出的张量
        save_path: 保存路径
        epoch: 当前epoch
        max_channels_per_layer: 每层最多显示的特征图数量
    """
    
    # 创建保存目录
    layer_vis_dir = os.path.join(save_path, f'epoch_{epoch}_layer_visualizations')
    os.makedirs(layer_vis_dir, exist_ok=True)
    
    # 定义层顺序和显示名称
    layer_order = ['input'] + [f'encoder_{i}' for i in range(5)] + [f'decoder_{i}' for i in range(5)] + ['final_conv', 'final_output']
    
    for layer_name in layer_order:
        if layer_name not in layer_outputs:
            continue
            
        tensor = layer_outputs[layer_name]
        
        # 处理不同维度的张量
        if tensor.dim() == 4:  # [B, C, H, W]
            batch_size, channels, height, width = tensor.shape
            tensor = tensor[0]  # 取第一个batch
        elif tensor.dim() == 3:  # [C, H, W]
            channels, height, width = tensor.shape
        elif tensor.dim() == 2:  # [H, W]
            height, width = tensor.shape
            channels = 1
            tensor = tensor.unsqueeze(0)
        else:
            print(f"警告: 无法处理 {layer_name} 的维度 {tensor.shape}")
            continue
        
        # 限制显示的特征图数量
        display_channels = min(channels, max_channels_per_layer)
        
        # 创建子图布局
        if display_channels <= 4:
            rows, cols = 2, 2
        elif display_channels <= 9:
            rows, cols = 3, 3
        elif display_channels <= 16:
            rows, cols = 4, 4
        else:
            rows, cols = 4, 4
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # 绘制特征图
        for i in range(rows * cols):
            if i < display_channels:
                # 获取第i个通道的特征图
                if channels > 1:
                    feature_map = tensor[i].detach().cpu().numpy()
                else:
                    feature_map = tensor[0].detach().cpu().numpy()
                
                # 归一化到[0,1]用于显示
                feature_map_norm = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
                
                im = axes[i].imshow(feature_map_norm, cmap='viridis', aspect='auto')
                axes[i].set_title(f'Channel {i}')
                axes[i].axis('off')
                
                # 添加颜色条
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            else:
                axes[i].axis('off')
        
        # 设置总标题
        fig.suptitle(f'{layer_name} - Shape: {tensor.shape}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 保存图像
        save_file = os.path.join(layer_vis_dir, f'{layer_name}_features.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存 {layer_name} 特征图: {save_file}")

def visualize_feature_statistics(layer_outputs, save_path, epoch):
    """
    可视化每层特征的统计信息
    
    Args:
        layer_outputs: 字典，包含每层输出的张量
        save_path: 保存路径
        epoch: 当前epoch
    """
    
    # 创建保存目录
    stats_dir = os.path.join(save_path, f'epoch_{epoch}_feature_statistics')
    os.makedirs(stats_dir, exist_ok=True)
    
    # 收集统计信息
    layer_names = []
    means = []
    stds = []
    mins = []
    maxs = []
    channel_counts = []
    
    for layer_name, tensor in layer_outputs.items():
        if tensor.dim() >= 2:
            layer_names.append(layer_name)
            
            # 计算统计信息
            tensor_flat = tensor.flatten()
            means.append(tensor_flat.mean().item())
            stds.append(tensor_flat.std().item())
            mins.append(tensor_flat.min().item())
            maxs.append(tensor_flat.max().item())
            
            # 计算通道数
            if tensor.dim() == 4:
                channel_counts.append(tensor.shape[1])
            elif tensor.dim() == 3:
                channel_counts.append(tensor.shape[0])
            else:
                channel_counts.append(1)
    
    # 创建统计图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 均值变化
    axes[0, 0].plot(range(len(layer_names)), means, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_title('Feature Mean Across Layers')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Mean Value')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 标准差变化
    axes[0, 1].plot(range(len(layer_names)), stds, 'r-o', linewidth=2, markersize=6)
    axes[0, 1].set_title('Feature Std Across Layers')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Std Value')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 值范围变化
    axes[1, 0].fill_between(range(len(layer_names)), mins, maxs, alpha=0.3, color='green', label='Value Range')
    axes[1, 0].plot(range(len(layer_names)), means, 'b-o', linewidth=2, markersize=6, label='Mean')
    axes[1, 0].set_title('Feature Value Range Across Layers')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. 通道数变化
    axes[1, 1].bar(range(len(layer_names)), channel_counts, color='purple', alpha=0.7)
    axes[1, 1].set_title('Channel Count Across Layers')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Number of Channels')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 设置x轴标签
    for ax in axes.flat:
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 保存统计图
    stats_file = os.path.join(stats_dir, 'feature_statistics.png')
    plt.savefig(stats_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存特征统计图: {stats_file}")
    
    # 保存统计数据到文件
    stats_data_file = os.path.join(stats_dir, 'feature_statistics.txt')
    with open(stats_data_file, 'w', encoding='utf-8') as f:
        f.write(f"UNet特征统计信息 - Epoch {epoch}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, layer_name in enumerate(layer_names):
            f.write(f"{layer_name}:\n")
            f.write(f"  通道数: {channel_counts[i]}\n")
            f.write(f"  均值: {means[i]:.6f}\n")
            f.write(f"  标准差: {stds[i]:.6f}\n")
            f.write(f"  最小值: {mins[i]:.6f}\n")
            f.write(f"  最大值: {maxs[i]:.6f}\n")
            f.write(f"  范围: {maxs[i] - mins[i]:.6f}\n\n")
    
    print(f"已保存特征统计数据: {stats_data_file}")

def create_layer_comparison_grid(layer_outputs, save_path, epoch, selected_layers=None):
    """
    创建选定层的对比网格图
    
    Args:
        layer_outputs: 字典，包含每层输出的张量
        save_path: 保存路径
        epoch: 当前epoch
        selected_layers: 要对比的层列表，如果为None则选择关键层
    """
    
    if selected_layers is None:
        selected_layers = ['input', 'encoder_0', 'encoder_2', 'decoder_0', 'decoder_2', 'final_output']
    
    # 过滤存在的层
    available_layers = [layer for layer in selected_layers if layer in layer_outputs]
    
    if not available_layers:
        print("没有找到可用的层进行对比")
        return
    
    # 创建保存目录
    comparison_dir = os.path.join(save_path, f'epoch_{epoch}_layer_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 计算网格布局
    n_layers = len(available_layers)
    cols = min(3, n_layers)
    rows = (n_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, layer_name in enumerate(available_layers):
        tensor = layer_outputs[layer_name]
        
        # 处理张量维度
        if tensor.dim() == 4:
            tensor = tensor[0]  # 取第一个batch
        if tensor.dim() == 3:
            # 对于多通道，取平均或第一个通道
            if tensor.shape[0] > 1:
                tensor = tensor.mean(dim=0)  # 取通道平均
            else:
                tensor = tensor[0]
        
        # 转换为numpy数组
        feature_map = tensor.detach().cpu().numpy()
        
        # 归一化
        feature_map_norm = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        # 绘制
        im = axes[i].imshow(feature_map_norm, cmap='viridis', aspect='auto')
        axes[i].set_title(f'{layer_name}\nShape: {tensor.shape}', fontsize=10)
        axes[i].axis('off')
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # 隐藏多余的子图
    for i in range(n_layers, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'UNet Layer Comparison - Epoch {epoch}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存对比图
    comparison_file = os.path.join(comparison_dir, 'layer_comparison.png')
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存层对比图: {comparison_file}")

def visualize_gradient_flow(net, save_path, epoch):
    """
    可视化梯度流
    
    Args:
        net: UNet网络
        save_path: 保存路径
        epoch: 当前epoch
    """
    
    # 创建保存目录
    grad_dir = os.path.join(save_path, f'epoch_{epoch}_gradient_flow')
    os.makedirs(grad_dir, exist_ok=True)
    
    # 收集梯度信息
    layer_names = []
    grad_norms = []
    param_counts = []
    
    for name, param in net.named_parameters():
        if param.grad is not None:
            layer_names.append(name)
            grad_norms.append(param.grad.norm().item())
            param_counts.append(param.numel())
    
    if not layer_names:
        print("没有找到梯度信息")
        return
    
    # 创建梯度可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 梯度范数
    axes[0, 0].bar(range(len(layer_names)), grad_norms, color='blue', alpha=0.7)
    axes[0, 0].set_title('Gradient Norm by Layer')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Gradient Norm')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 参数数量
    axes[0, 1].bar(range(len(layer_names)), param_counts, color='green', alpha=0.7)
    axes[0, 1].set_title('Parameter Count by Layer')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Number of Parameters')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 梯度范数 vs 参数数量
    axes[1, 0].scatter(param_counts, grad_norms, alpha=0.7, s=50)
    axes[1, 0].set_title('Gradient Norm vs Parameter Count')
    axes[1, 0].set_xlabel('Parameter Count')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 梯度范数分布
    axes[1, 1].hist(grad_norms, bins=20, alpha=0.7, color='red')
    axes[1, 1].set_title('Gradient Norm Distribution')
    axes[1, 1].set_xlabel('Gradient Norm')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 设置x轴标签
    for ax in [axes[0, 0], axes[0, 1]]:
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 保存梯度流图
    grad_file = os.path.join(grad_dir, 'gradient_flow.png')
    plt.savefig(grad_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存梯度流图: {grad_file}")
    
    # 保存梯度数据
    grad_data_file = os.path.join(grad_dir, 'gradient_data.txt')
    with open(grad_data_file, 'w', encoding='utf-8') as f:
        f.write(f"UNet梯度流信息 - Epoch {epoch}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, layer_name in enumerate(layer_names):
            f.write(f"{layer_name}:\n")
            f.write(f"  参数数量: {param_counts[i]}\n")
            f.write(f"  梯度范数: {grad_norms[i]:.6f}\n\n")
    
    print(f"已保存梯度数据: {grad_data_file}")
