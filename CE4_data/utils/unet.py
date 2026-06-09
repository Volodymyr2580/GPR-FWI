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
        

        # 不使用前两层的跳跃连接，只在深层使用
        self.decoders = nn.ModuleList()
        
        # 前两层decoder：不使用skip connection
        self.decoders.append(
            DecoderBlock(
                in_channels=decoder_channels[0],
                skip_channels=encoder_channels[0],  # 不会实际使用
                out_channels=decoder_channels[1], 
                skipbool=False
            )
        )
        self.decoders.append(
            DecoderBlock(
                in_channels=decoder_channels[1],
                skip_channels=encoder_channels[0],  # 不会实际使用
                out_channels=decoder_channels[2], 
                skipbool=False
            )
        )
        
        # 后面的decoder：使用skip connection
        # 由于前两层不使用skip，索引需要调整
        # decoder总数 = 前2层 + range(2, len-1) = 2 + (len-3) = len-1 层
        n_decoders_total = len(decoder_channels) - 1  # 4层decoder (0,1,2,3)
        for i in range(2, len(decoder_channels)-1):
            # i=2: 经过2次upsample，空间x4，连接encoder 1
            # i=3: 经过3次upsample，空间x8，连接encoder 0
            encoder_idx = n_decoders_total - 1 - i
            self.decoders.append(
                DecoderBlock(
                    in_channels=decoder_channels[i],
                    skip_channels=encoder_channels[encoder_idx],
                    out_channels=decoder_channels[i+1],
                    skipbool=True
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
            if i < 2:
                # 前两层decoder：不使用skip connection
                x = decoder(x, skip=None)
                decoder_mid_out.append(x.contiguous().mean(dim=[0,1], keepdim=False))
            else:
                # 后面的decoder：使用skip connection
                # 因为前两层不使用skip，所以索引需要调整
                # i=2: decoder经过2次upsample，空间尺寸x4，应连接skip1
                # i=3: decoder经过3次upsample，空间尺寸x8，应连接skip0
                # 正确索引: skip_idx = len(decoders) - 1 - i
                skip_idx = len(self.decoders) - 1 - i
                skip = skips[skip_idx]
                x = decoder(x, skip)
                decoder_mid_out.append(x.contiguous().mean(dim=[0,1], keepdim=False))
        
        # 获取最终输出
        final_output = self.final_conv(x)
        
        # 动态裁剪到原始尺寸（去掉padding）
        # 假设输入是经过padding的，需要裁剪回原始尺寸
        original_height, original_width = 410, 5000  # 原始目标尺寸
        final_output = final_output[:, :, :original_height, :original_width]
        
        return encoder_mid_output, decoder_mid_out, final_output.squeeze()  # 返回 [410, 5000]