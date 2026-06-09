# GPR双参数全波形反演UNet网络重参数化

本项目实现了基于UNet网络的GPR（探地雷达）电磁波双参数全波形反演，参考了声波FWI的实现方法，但适配了电磁波模型。

## 项目结构

```
mode2_unet/
├── gpr_data_loader.py      # GPR数据加载器
├── gpr_train.py           # GPR双参数训练脚本
├── test_gpr_workflow.py   # 工作流程测试脚本
├── unet.py                # UNet网络定义
├── forward.py             # FDTD正演模拟
├── gradient.py            # 梯度计算
├── Time_loop.py           # 时间循环
├── Add_CPML.py            # CPML边界条件
├── Wavelet.py             # 子波生成
├── main_autograd.py       # 原始自动微分版本
└── README_GPR_FWI.md      # 本说明文件
```

## 核心特性

### 1. 双参数反演
- **介电常数 (ε)**：控制电磁波传播速度
- **电导率 (σ)**：控制电磁波衰减

### 2. UNet网络重参数化
- 使用两个独立的UNet网络分别参数化介电常数和电导率
- 网络输出通过sigmoid函数归一化到物理范围
- 支持浅层固定模型和深层网络输出的拼接

### 3. 电磁波FDTD正演模拟
- 基于Maxwell方程的2D FDTD方法
- 支持CPML边界条件
- 400MHz Ricker子波震源

## 工作流程

### 1. 数据准备 (`gpr_data_loader.py`)

```python
# 加载GPR数据
observed_data, data_dim, model_dim, source_amplitudes, \
source_locations, receiver_locations, epsilon_true, sigma_true = GPR_DataLoad_Train(n_shots)
```

**功能：**
- 创建真实电磁参数模型（介电常数和电导率）
- 设置震源和接收点位置
- 生成400MHz Ricker子波并进行高通滤波
- 使用FDTD进行正演模拟生成观测数据

**参数设置：**
- 模型尺寸：100×200网格点
- 网格间距：0.02m
- 时间采样：4e-11s
- 主频：400MHz

### 2. 网络训练 (`gpr_train.py`)

```python
# 双UNet网络
net_1 = UNet(in_channels=1)  # 介电常数网络
net_2 = UNet(in_channels=1)  # 电导率网络

# 参数转换
net1_o = torch.sigmoid(net1) * (max_eps - min_eps) + min_eps
net2_o = torch.sigmoid(net2) * (max_sig - min_sig) + min_sig

# 模型拼接
epsilon = torch.cat((epsilon_layer, net1_o), dim=1)
sigma = torch.cat((sigma_layer, net2_o), dim=1)
```

**训练策略：**
- 介电常数网络学习率：1e-3
- 电导率网络学习率：1e-2
- 使用Adam优化器
- 多步学习率调度

### 3. 正演模拟和损失计算

```python
# 正演模拟
d_syn = forward_model(eps_np, sig_np, source_list, receiver_list, 
                     dt, dx, dz, npml, freq, steps)

# 损失计算
loss = loss_fn(d_syn, s_d_i.to(device))
```

## 使用方法

### 1. 环境准备

确保安装以下依赖：
```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib scipy
pip install pytorch-msssim
```

### 2. 测试工作流程

```bash
cd Gpr_fwi/mode2_unet
python test_gpr_workflow.py
```

这将测试整个工作流程的各个组件是否正常工作。

### 3. 开始训练

```bash
python gpr_train.py
```

训练过程将：
- 自动创建结果目录
- 保存网络参数和可视化结果
- 记录损失函数变化
- 生成中间结果图像

### 4. 结果分析

训练完成后，结果保存在 `result_gpr/` 目录下：
- `loss_data.txt`：数据损失记录
- `loss_model_eps.txt`：介电常数模型损失
- `loss_model_sig.txt`：电导率模型损失
- `batch*/`：各批次的结果图像和数据

## 关键参数说明

### 物理参数
- `dx, dz = 0.02m`：网格间距
- `dt = 4e-11s`：时间采样间隔
- `freq = 4e8Hz`：震源主频
- `npml = 10`：CPML边界层数

### 网络参数
- `in_channels = 1`：输入通道数
- `encoder_channels = [36, 72, 144, 288, 576]`：编码器通道数
- `decoder_channels = [576, 288, 144, 72, 36]`：解码器通道数

### 训练参数
- `BatchSize = 4`：批次大小
- `number_of_shots = 20`：炮数
- `num_epochs = 3001`：训练轮数
- `l_size = 30`：浅层固定层数

## 与声波FWI的对比

| 特性 | 声波FWI | GPR FWI |
|------|---------|---------|
| 物理模型 | 声波方程 | Maxwell方程 |
| 参数 | 速度、密度 | 介电常数、电导率 |
| 波速 | 1500-5500 m/s | 光速/√ε |
| 频率 | 7Hz | 400MHz |
| 网格间距 | 20m | 0.02m |
| 时间采样 | 0.002s | 4e-11s |

## 注意事项

1. **CFL条件**：确保时间采样间隔满足CFL稳定性条件
2. **梯度处理**：使用梯度钩子函数处理梯度异常值
3. **参数范围**：介电常数1-10，电导率1e-6到1e-3 S/m
4. **内存使用**：大模型可能需要GPU加速

## 故障排除

### 常见问题

1. **梯度爆炸**：检查学习率设置和梯度裁剪
2. **内存不足**：减少批次大小或模型尺寸
3. **收敛缓慢**：调整学习率或网络结构
4. **数值不稳定**：检查CFL条件和边界条件

### 调试建议

1. 先运行测试脚本验证工作流程
2. 使用小规模模型进行调试
3. 监控损失函数变化
4. 检查中间结果的可视化

## 扩展功能

- 支持3D模型
- 添加正则化项
- 多尺度策略
- 不确定性量化
- 实时可视化

## 参考文献

1. Zhao, T., et al. "A Hybrid Optimization Framework for Seismic Full Waveform Inversion"
2. UNet: Convolutional Networks for Biomedical Image Segmentation
3. FDTD Method for Electromagnetic Wave Propagation 