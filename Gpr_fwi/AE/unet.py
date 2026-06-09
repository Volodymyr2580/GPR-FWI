import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.init as init
 
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_block = nn.Sequential(
            ConvBlock(out_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        return self.conv_block(x)

class AutoEncoder(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1,
                 encoder_channels=[36, 72, 144, 288, 576],
                 decoder_channels=[576, 288, 144, 72, 36]):
        super().__init__()
        
        assert len(encoder_channels) == len(decoder_channels), "Encoder and decoder stages mismatch"
        assert decoder_channels[0] == encoder_channels[-1], "First decoder channel should match last encoder channel"
        
        # 编码器
        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in encoder_channels:
            self.encoders.append(EncoderBlock(prev_ch, ch))
            prev_ch = ch
        
        # 解码器
        self.decoders = nn.ModuleList()
        for i in range(len(decoder_channels)):
            if i == 0:
                # 第一个解码器：从编码器最后一层的通道数开始
                self.decoders.append(
                    DecoderBlock(
                        in_channels=encoder_channels[-1],
                        out_channels=decoder_channels[i]
                    )
                )
            else:
                # 后续解码器：从上一个解码器的输出通道数开始
                self.decoders.append(
                    DecoderBlock(
                        in_channels=decoder_channels[i-1],
                        out_channels=decoder_channels[i]
                    )
                )

        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)
        )

    def forward(self, x):
        # 编码过程
        for encoder in self.encoders:
            x = encoder(x)
        
        # 解码过程
        for decoder in self.decoders:
            x = decoder(x)
        
        # 获取最终输出
        final_output = self.final_conv(x)
        
        # 动态裁剪到原始尺寸（去掉padding）
        # 假设输入是经过padding的，需要裁剪回原始尺寸
        original_height, original_width = 80, 200  # 原始目标尺寸
        final_output = final_output[:, :, :original_height, :original_width]
        
        return final_output.squeeze()  # 返回 [80, 200]