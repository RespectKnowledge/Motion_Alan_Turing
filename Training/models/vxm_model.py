import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# Simple Conv Block
# -------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# -------------------------------
# Simplified 3D U-Net
# -------------------------------
class SimpleUnet3D(nn.Module):
    def __init__(self, in_channels=2, base_ch=16):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch)         # 2 → 16
        self.enc2 = ConvBlock(base_ch, base_ch * 2)         # 16 → 32
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)     # 32 → 64

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)  # 64 → 128

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        # Final feature map
        self.final_nf = base_ch

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool3d(e1, 2))
        e3 = self.enc3(F.max_pool3d(e2, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool3d(e3, 2))

        # Decoder
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return d1

class VxmDenseSimple(nn.Module):
    def __init__(self, inshape, base_ch=16):
        super().__init__()
        self.unet = SimpleUnet3D(in_channels=2, base_ch=base_ch)
        self.flow = nn.Conv3d(self.unet.final_nf, len(inshape), kernel_size=3, padding=1)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, x):
        source = x[:, 0:1, ...]
        x = self.unet(x)
        flow_field = self.flow(x)
        y_source = self.transformer(source, flow_field)
        return y_source, flow_field

import torch

# -------------------------------
# Dummy Spatial Transformer
# (replace with your version later)
# -------------------------------
import torch.nn.functional as F

class SpatialTransformer(torch.nn.Module):
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode

        # Create grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(*vectors, indexing="ij")
        grid = torch.stack(grids)  # (3, D, H, W)
        grid = grid.unsqueeze(0).float()  # (1, 3, D, H, W)
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Normalize to [-1, 1]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # Reorder dims for grid_sample
        new_locs = new_locs.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3)
        new_locs = new_locs[..., [2, 1, 0]]         # (z, y, x)

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


# -------------------------------
# Your VxmDenseSimple Model
# -------------------------------
model = VxmDenseSimple(inshape=(224, 224, 32), base_ch=16)