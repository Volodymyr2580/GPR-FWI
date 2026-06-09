#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试UNet调试输出功能
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet

def test_unet_debug_output():
    """测试UNet的调试输出功能"""
    print("开始测试UNet调试输出功能...")
    
    # 创建UNet网络
    net = UNet(in_channels=1)
    print(f"UNet网络创建成功，参数数量: {sum(p.numel() for p in net.parameters())}")
    
    # 创建测试输入
    batch_size = 1
    height, width = 100, 200
    test_input = torch.randn(batch_size, 1, height, width)
    print(f"测试输入形状: {test_input.shape}")
    
    # 测试正常模式
    print("\n=== 测试正常模式 ===")
    net.set_debug_mode(False)
    encoder_out, decoder_out, final_out = net(test_input)
    print(f"正常模式输出形状: {final_out.shape}")
    print(f"调试信息是否为空: {len(net.debug_info) == 0}")
    
    # 测试调试模式
    print("\n=== 测试调试模式 ===")
    net.set_debug_mode(True)
    encoder_out, decoder_out, final_out = net(test_input)
    print(f"调试模式输出形状: {final_out.shape}")
    print(f"调试信息数量: {len(net.debug_info)}")
    
    # 输出调试信息
    print("\n=== 调试信息详情 ===")
    for layer_name, info in net.debug_info.items():
        print(f"{layer_name}:")
        for key, value in info.items():
            if isinstance(value, (list, tuple)):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value:.6f}")
        print()
    
    # 测试梯度计算
    print("\n=== 测试梯度计算 ===")
    test_input.requires_grad_(True)
    encoder_out, decoder_out, final_out = net(test_input)
    
    # 计算一个简单的损失
    target = torch.randn_like(final_out)
    loss = nn.MSELoss()(final_out, target)
    print(f"损失值: {loss.item():.6f}")
    
    # 反向传播
    loss.backward()
    print(f"输入梯度形状: {test_input.grad.shape}")
    print(f"输入梯度范围: [{test_input.grad.min().item():.6f}, {test_input.grad.max().item():.6f}]")
    
    print("\n测试完成！UNet调试输出功能正常工作。")

if __name__ == "__main__":
    test_unet_debug_output()
