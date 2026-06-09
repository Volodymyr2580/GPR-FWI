# UNet调试输出功能说明

## 概述
为了在2950-3001这些epochs中输出详细的UNet网络信息，我对代码进行了以下修改：

## 修改内容

### 1. UNet网络修改 (`unet.py`)

#### 新增功能：
- **调试模式控制**：添加了 `debug_mode` 标志和 `set_debug_mode()` 方法
- **中间层信息记录**：在 `forward()` 方法中记录每层的输出统计信息
- **调试信息存储**：使用 `debug_info` 字典存储各层的详细信息
- **层输出张量存储**：使用 `layer_outputs` 字典存储每层的实际输出张量用于可视化

#### 记录的信息包括：
- 输入层：形状、均值、标准差、最小值、最大值
- 编码器层（encoder_0 到 encoder_4）：输出形状和统计信息
- 解码器层（decoder_0 到 decoder_4）：输出形状和统计信息
- 最终卷积层：输出形状和统计信息
- 最终输出：形状和统计信息

### 2. 可视化工具 (`visualization_utils.py`)

#### 新增功能：
- **每层特征图可视化**：`visualize_unet_layers()` - 生成每层输出的特征图网格
- **特征统计信息可视化**：`visualize_feature_statistics()` - 生成特征统计图表
- **层对比网格**：`create_layer_comparison_grid()` - 创建选定层的对比图
- **梯度流可视化**：`visualize_gradient_flow()` - 可视化网络梯度流信息

### 3. 主程序修改 (`gpr_train_mpi.py`)

#### 新增功能：
- **条件调试模式**：在2950-3001 epochs中自动启用UNet调试模式
- **详细输出信息**：每个epoch输出以下信息：

#### 1. Epsilon输出信息
- 形状、范围、平均值、标准差
- 保存高分辨率图像：`epoch_{epoch}_epsilon_detailed.png`
- 保存数据文件：`epoch_{epoch}_epsilon.npy`

#### 2. Epsilon梯度信息
- 梯度形状、范围、平均值、标准差、L2范数
- 保存归一化梯度图像：`epoch_{epoch}_epsilon_grad_detailed.png`
- 保存梯度数据：`epoch_{epoch}_epsilon_grad.npy` 和 `epoch_{epoch}_epsilon_grad_norm.npy`

#### 3. UNet网络层信息
- 输入层、编码器层、解码器层、最终输出的详细信息
- 保存调试信息文件：`epoch_{epoch}_unet_debug_info.txt`
- 控制台输出关键层信息

#### 4. UNet网络层可视化
- 每层特征图网格可视化
- 特征统计信息图表
- 层对比网格图
- 梯度流可视化

#### 5. 网络参数更新信息
- 总参数数量和可训练参数数量
- 各层参数的梯度L2范数（显示前10个）
- 参数更新量统计

## 输出文件说明

在2950-3001 epochs期间，每个epoch会生成以下文件：

### 图像文件
- `epoch_{epoch}_epsilon_detailed.png`：高分辨率epsilon输出图像
- `epoch_{epoch}_epsilon_grad_detailed.png`：归一化梯度图像

### 数据文件
- `epoch_{epoch}_epsilon.npy`：epsilon输出数据
- `epoch_{epoch}_epsilon_grad.npy`：原始梯度数据
- `epoch_{epoch}_epsilon_grad_norm.npy`：归一化梯度数据

### 文本文件
- `epoch_{epoch}_unet_debug_info.txt`：UNet网络各层详细信息

### 可视化目录
- `epoch_{epoch}_layer_visualizations/`：每层特征图可视化目录
  - `input_features.png`：输入层特征图
  - `encoder_0_features.png`：编码器第0层特征图
  - `encoder_1_features.png`：编码器第1层特征图
  - ...（其他编码器和解码器层）
  - `final_output_features.png`：最终输出特征图
- `epoch_{epoch}_feature_statistics/`：特征统计信息目录
  - `feature_statistics.png`：特征统计图表
  - `feature_statistics.txt`：特征统计数据
- `epoch_{epoch}_layer_comparison/`：层对比目录
  - `layer_comparison.png`：层对比网格图
- `epoch_{epoch}_gradient_flow/`：梯度流目录
  - `gradient_flow.png`：梯度流图表
  - `gradient_data.txt`：梯度数据

## 使用方法

1. 正常运行训练程序：`python gpr_train_mpi.py`
2. 程序会在2950-3001 epochs自动启用详细输出
3. 在3002 epoch后自动关闭调试模式
4. 所有输出文件保存在结果目录中

## 控制台输出示例

```
=== Epoch 2950 详细输出信息 ===
1. Epsilon输出信息:
   形状: (100, 200)
   范围: [1.000000, 6.000000]
   平均值: 3.245678
   标准差: 1.234567
   已保存epsilon图像: epoch_2950_epsilon_detailed.png
   已保存epsilon数据: epoch_2950_epsilon.npy

2. Epsilon梯度信息:
   梯度形状: (100, 200)
   梯度范围: [-0.123456, 0.234567]
   梯度平均值: 0.000123
   梯度标准差: 0.045678
   梯度L2范数: 0.123456
   已保存梯度图像: epoch_2950_epsilon_grad_detailed.png
   已保存梯度数据: epoch_2950_epsilon_grad.npy

3. UNet网络层信息:
   输入层信息:
     形状: torch.Size([1, 1, 100, 200])
     范围: [-2.123456, 2.345678]
     平均值: 0.123456
   编码器层信息:
     encoder_0: 形状=torch.Size([1, 36, 50, 100]), 范围=[-1.234567, 1.345678]
     encoder_1: 形状=torch.Size([1, 72, 25, 50]), 范围=[-0.987654, 0.876543]
     ...
   解码器层信息:
     decoder_0: 形状=torch.Size([1, 288, 25, 50]), 范围=[-0.765432, 0.654321]
     ...
   最终输出信息:
     形状: torch.Size([100, 200])
     范围: [-1.234567, 1.345678]
     平均值: 0.123456
   已保存UNet调试信息: epoch_2950_unet_debug_info.txt

4. 网络参数更新信息:
   总参数数量: 1234567
   可训练参数数量: 1234567
   参数更新量 (梯度L2范数):
     encoders.0.conv1.conv.weight: 0.001234
     encoders.0.conv1.conv.bias: 0.000567
     ...
=== Epoch 2950 详细输出信息结束 ===
```

## 注意事项

1. 调试模式只在2950-3001 epochs中启用，不会影响其他epochs的性能
2. 所有输出文件都保存在结果目录中，文件名包含epoch编号
3. 调试信息文件使用UTF-8编码，包含完整的网络层信息
4. 图像文件使用高分辨率（300 DPI）保存，便于后续分析
5. 梯度数据同时保存原始值和归一化值，便于不同用途的分析

## 测试

可以使用 `test_debug_output.py` 脚本测试UNet调试功能是否正常工作：

```bash
python test_debug_output.py
```

这个测试脚本会验证：
- UNet调试模式的开启和关闭
- 中间层信息的记录
- 梯度计算的正确性
- 调试信息的完整性
