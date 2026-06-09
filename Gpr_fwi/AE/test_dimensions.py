import torch
import torch.nn.functional as F
from unet import AutoEncoder

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型
model = AutoEncoder(in_channels=1, out_channels=1).to(device)

# 测试输入
batch_size = 4
height, width = 100, 200
surface_layers = 20

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

# 创建测试输入
test_input = torch.randn(batch_size, height, width).to(device)
test_input_padded = F.pad(test_input, (pad_width//2, pad_width//2, pad_height//2, pad_height//2))
test_input_padded = test_input_padded.unsqueeze(1)  # [batch_size, 1, height+pad_height, width+pad_width]

print(f"输入形状: {test_input_padded.shape}")

# 前向传播
with torch.no_grad():
    output = model(test_input_padded)

print(f"网络输出形状: {output.shape}")

# 测试拼接
deep_output_processed = torch.sigmoid(output) * 10 + 1
uniform_layer = torch.ones((batch_size, surface_layers, width), device=device) * 3.0
full_output = torch.cat([uniform_layer, deep_output_processed], dim=1)

print(f"深层输出形状: {deep_output_processed.shape}")
print(f"浅层形状: {uniform_layer.shape}")
print(f"拼接后形状: {full_output.shape}")
print(f"期望形状: [{batch_size}, {height}, {width}]")

# 验证形状是否匹配
assert full_output.shape == (batch_size, height, width), f"形状不匹配: {full_output.shape} vs ({batch_size}, {height}, {width})"
print("✓ 形状匹配正确！")