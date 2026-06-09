#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试UNet层输出可视化功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from unet import UNet
from visualization_utils import visualize_unet_layers, visualize_feature_statistics, create_layer_comparison_grid, visualize_gradient_flow

def test_unet_visualization():
    """测试UNet可视化功能"""
    
    print("开始测试UNet层输出可视化功能...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建UNet网络
    net = UNet(in_channels=1).to(device)
    print(f"网络参数数量: {sum(p.numel() for p in net.parameters())}")
    
    # 启用调试模式
    net.set_debug_mode(True)
    print("已启用UNet调试模式")
    
    # 创建测试输入
    batch_size = 1
    height, width = 100, 200
    
    # 创建带padding的输入
    required_height = ((height + 31) // 32) * 32
    required_width = ((width + 31) // 32) * 32
    
    pad_height = required_height - height
    pad_width = required_width - width
    
    if pad_height % 2 == 1:
        pad_height += 1
    if pad_width % 2 == 1:
        pad_width += 1
    
    # 创建测试数据
    test_input = torch.randn(batch_size, 1, required_height, required_width).to(device)
    print(f"测试输入形状: {test_input.shape}")
    
    # 前向传播
    print("执行前向传播...")
    with torch.no_grad():
        encoder_outputs, decoder_outputs, final_output = net(test_input)
    
    print(f"编码器输出数量: {len(encoder_outputs)}")
    print(f"解码器输出数量: {len(decoder_outputs)}")
    print(f"最终输出形状: {final_output.shape}")
    
    # 检查调试信息
    print(f"调试信息键: {list(net.debug_info.keys())}")
    print(f"层输出键: {list(net.layer_outputs.keys())}")
    
    # 创建测试输出目录
    test_output_dir = "test_visualization_output"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # 测试可视化功能
    print("\n测试每层特征图可视化...")
    try:
        visualize_unet_layers(net.layer_outputs, test_output_dir, 0)
        print("✓ 每层特征图可视化测试通过")
    except Exception as e:
        print(f"✗ 每层特征图可视化测试失败: {e}")
    
    print("\n测试特征统计信息可视化...")
    try:
        visualize_feature_statistics(net.layer_outputs, test_output_dir, 0)
        print("✓ 特征统计信息可视化测试通过")
    except Exception as e:
        print(f"✗ 特征统计信息可视化测试失败: {e}")
    
    print("\n测试层对比网格可视化...")
    try:
        create_layer_comparison_grid(net.layer_outputs, test_output_dir, 0)
        print("✓ 层对比网格可视化测试通过")
    except Exception as e:
        print(f"✗ 层对比网格可视化测试失败: {e}")
    
    # 测试梯度流可视化（需要先计算梯度）
    print("\n测试梯度流可视化...")
    try:
        # 创建一个简单的损失函数
        target = torch.randn_like(final_output)
        loss = torch.nn.functional.mse_loss(final_output, target)
        
        # 计算梯度
        loss.backward()
        
        visualize_gradient_flow(net, test_output_dir, 0)
        print("✓ 梯度流可视化测试通过")
    except Exception as e:
        print(f"✗ 梯度流可视化测试失败: {e}")
    
    # 检查生成的文件
    print(f"\n检查生成的文件...")
    for root, dirs, files in os.walk(test_output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"  {file_path} ({file_size} bytes)")
    
    print(f"\n测试完成！输出文件保存在: {test_output_dir}")
    
    # 清理测试文件
    import shutil
    try:
        shutil.rmtree(test_output_dir)
        print("已清理测试文件")
    except:
        print("无法清理测试文件，请手动删除")

def test_layer_output_details():
    """测试层输出详细信息"""
    
    print("\n测试层输出详细信息...")
    
    # 创建UNet网络
    net = UNet(in_channels=1)
    net.set_debug_mode(True)
    
    # 创建测试输入
    test_input = torch.randn(1, 1, 128, 256)  # 使用32的倍数
    
    # 前向传播
    with torch.no_grad():
        encoder_outputs, decoder_outputs, final_output = net(test_input)
    
    # 打印每层的详细信息
    print("层输出详细信息:")
    for layer_name, tensor in net.layer_outputs.items():
        print(f"\n{layer_name}:")
        print(f"  形状: {tensor.shape}")
        print(f"  数据类型: {tensor.dtype}")
        print(f"  设备: {tensor.device}")
        print(f"  均值: {tensor.mean().item():.6f}")
        print(f"  标准差: {tensor.std().item():.6f}")
        print(f"  最小值: {tensor.min().item():.6f}")
        print(f"  最大值: {tensor.max().item():.6f}")
        print(f"  是否包含NaN: {torch.isnan(tensor).any().item()}")
        print(f"  是否包含Inf: {torch.isinf(tensor).any().item()}")

if __name__ == "__main__":
    print("=" * 60)
    print("UNet层输出可视化功能测试")
    print("=" * 60)
    
    # 测试基本功能
    test_unet_visualization()
    
    # 测试详细信息
    test_layer_output_details()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
