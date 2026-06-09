import torch
import torch.nn as nn

from .unet import ConvBlock, EncoderBlock, DecoderBlock


class UNetTest(nn.Module):
    """
    A lightweight wrapper around the existing UNet blocks that allows cropping
    the network output to arbitrary target sizes without touching the original
    implementation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_channels=None,
        decoder_channels=None,
        target_height: int = 100,
        target_width: int = 500,
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [8, 16, 32, 64, 128]
        if decoder_channels is None:
            decoder_channels = [128, 64, 32, 16, 8]

        assert len(encoder_channels) == len(decoder_channels), (
            "Encoder and decoder stage counts must match."
        )
        assert (
            decoder_channels[0] == encoder_channels[-1]
        ), "First decoder channel should equal last encoder channel."

        self.target_height = target_height
        self.target_width = target_width

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in encoder_channels:
            self.encoders.append(EncoderBlock(prev_ch, ch))
            prev_ch = ch

        self.decoders = nn.ModuleList()

        # First two decoders: no skip connections
        self.decoders.append(
            DecoderBlock(
                in_channels=decoder_channels[0],
                skip_channels=encoder_channels[0],  # placeholder
                out_channels=decoder_channels[1],
                skipbool=False,
            )
        )
        self.decoders.append(
            DecoderBlock(
                in_channels=decoder_channels[1],
                skip_channels=encoder_channels[0],  # placeholder
                out_channels=decoder_channels[2],
                skipbool=False,
            )
        )

        # Remaining decoders use skip connections
        n_decoders_total = len(decoder_channels) - 1
        for i in range(2, len(decoder_channels) - 1):
            encoder_idx = n_decoders_total - 1 - i
            self.decoders.append(
                DecoderBlock(
                    in_channels=decoder_channels[i],
                    skip_channels=encoder_channels[encoder_idx],
                    out_channels=decoder_channels[i + 1],
                    skipbool=True,
                )
            )

        self.final_conv = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        encoder_mid_output = []
        decoder_mid_out = []

        for encoder in self.encoders:
            x, skip = encoder(x)
            encoder_mid_output.append(skip.contiguous().mean(dim=[0, 1], keepdim=False))
            skips.append(skip)
        x = skip

        for i, decoder in enumerate(self.decoders):
            if i < 2:
                x = decoder(x, skip=None)
                decoder_mid_out.append(x.contiguous().mean(dim=[0, 1], keepdim=False))
            else:
                skip_idx = len(self.decoders) - 1 - i
                skip_tensor = skips[skip_idx]
                x = decoder(x, skip_tensor)
                decoder_mid_out.append(x.contiguous().mean(dim=[0, 1], keepdim=False))

        final_output = self.final_conv(x)
        final_output = final_output[:, :, : self.target_height, : self.target_width]

        return encoder_mid_output, decoder_mid_out, final_output.squeeze()

