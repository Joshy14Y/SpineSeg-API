import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DoubleConv(nn.Module):
    """Standard UNet block: Conv -> BN -> ReLU (x2)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply double convolution block."""
        return self.block(x)


class Down(nn.Module):
    """Encoder block: MaxPool followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample and apply convolution."""
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Decoder block: Upsample, skip connection, then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample, align dimensions, concatenate skip and apply convolution."""
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Modified UNet for dense multiclass segmentation with z, fg, and refine heads."""

    def __init__(
        self, in_channels: int = 1, out_channels: int = 17, base_channels: int = 64
    ):
        super().__init__()
        self.in_conv = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
        self.up3 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up1 = Up(base_channels * 4, base_channels * 2)
        self.up0 = Up(base_channels * 2, base_channels)
        self.z_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.fg_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.refine_head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict:
        """Run encoder, bottleneck, decoder and return z, fg_logits, refine_logits."""
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = checkpoint(self.down3, x3, use_reentrant=False)
        x5 = checkpoint(self.bottleneck, x4, use_reentrant=False)
        x = self.up3(x5, x4)
        x = self.up2(x, x3)
        x = self.up1(x, x2)
        x = self.up0(x, x1)
        return {
            "z": torch.sigmoid(self.z_head(x)),
            "fg_logits": self.fg_head(x),
            "refine_logits": self.refine_head(x),
        }
