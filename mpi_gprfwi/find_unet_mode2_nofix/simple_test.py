#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的UNet可视化功能测试
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os

# 测试导入
try:
    from unet import UNet
    print("✓ UNet导入成功")
except Exception as e:
    print(f"✗ UNet导入失败: {e}")

try:
    from visualization_utils import visualize_unet_layers
    print("✓ visualization_utils导入成功")
except Exception as e:
    print(f"✗ visualization_utils导入失败: {e}")

# 测试UNet基本功能
print("\n测试UNet基本功能...")
try:
    net = UNet(in_channels=1)
    net.set_debug_mode(True)
    
    # 创建测试输入
    test_input = torch.randn(1, 1, 128, 256)
    print(f"测试输入形状: {test_input.shape}")
    
    # 前向传播
    with torch.no_grad():
        encoder_outputs, decoder_outputs, final_output = net(test_input)
    
    print(f"最终输出形状: {final_output.shape}")
    print(f"调试信息键: {list(net.debug_info.keys())}")
    print(f"层输出键: {list(net.layer_outputs.keys())}")
    
    print("✓ UNet基本功能测试通过")
    
except Exception as e:
    print(f"✗ UNet基本功能测试失败: {e}")

print("\n测试完成！")
