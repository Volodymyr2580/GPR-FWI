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
        skip = x
        x = self.pool(x)
        return x, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, skipbool=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.skipbool = skipbool

        if self.skipbool:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(skip_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
            self.conv_block = nn.Sequential(
                ConvBlock(out_channels * 2, out_channels),
                ConvBlock(out_channels, out_channels)
            )
        else:
            self.conv_block = nn.Sequential(
                ConvBlock(out_channels, out_channels),
                ConvBlock(out_channels, out_channels)
            )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        x = self.conv1(x)

        if self.skipbool:
            if skip is None:
                raise ValueError("skip must be provided when skipbool=True")
            skip = self.skip_conv(skip)
            x = torch.cat([x, skip], dim=1)
        
        return self.conv_block(x)

class UNet(nn.Module):
    def __init__(self, 
                 in_channels=20, 
                 out_channels=1,
                 encoder_channels=[36, 72, 144, 288, 576],
                 decoder_channels=[576, 288, 144, 72, 36]):
        super().__init__()
        

        assert len(encoder_channels) == len(decoder_channels), "Encoder and decoder stages mismatch"
        assert decoder_channels[0] == encoder_channels[-1], "First decoder channel should match last encoder channel"
        
        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in encoder_channels:
            self.encoders.append(EncoderBlock(prev_ch, ch))
            prev_ch = ch
        self.decoders = nn.ModuleList()
        for i in range(len(decoder_channels)-3):
            self.decoders.append(
                DecoderBlock(
                    in_channels=decoder_channels[i],
                    skip_channels=encoder_channels[-(i+2)], 
                    out_channels=decoder_channels[i+1]
                )
            )
        self.decoders.append(
            DecoderBlock(
                in_channels=decoder_channels[i+1],
                skip_channels=encoder_channels[0], 
                out_channels=decoder_channels[i+2], 
                skipbool = False
            )
        )
        self.decoders.append(
            DecoderBlock(
                in_channels=decoder_channels[i+2],
                skip_channels=encoder_channels[0],  
                out_channels=decoder_channels[i+3], 
                skipbool = False
            )
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        skips = []
        encoder_mid_output = []
        decoder_mid_out = []

        for encoder in self.encoders:
            x, skip = encoder(x)
            encoder_mid_output.append(skip.contiguous().mean(dim=[0,1], keepdim=False))
            skips.append(skip)
        x = skip
        

        for i, decoder in enumerate(self.decoders):
            if i < 3:
                skip = skips[-(i+2)] 
                x = decoder(x, skip)
                decoder_mid_out.append(x.contiguous().mean(dim=[0,1], keepdim=False))
            else:
                skip = skips[0] 
                x = decoder(x, skip)
                decoder_mid_out.append(x.contiguous().mean(dim=[0,1], keepdim=False))
        
        # 获取最终输出
        final_output = self.final_conv(x)
        
        # 动态裁剪到原始尺寸（去掉padding）
        # 假设输入是经过padding的，需要裁剪回原始尺寸
        original_height, original_width = 100, 200
        final_output = final_output[:, :, :original_height, :original_width]
        
        return encoder_mid_output, decoder_mid_out, final_output.squeeze()  # 返回 [150, 460]