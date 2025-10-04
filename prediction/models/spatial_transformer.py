import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Spatial Transformer --------------------
class SpatialTransformer(nn.Module):
    """N-D Spatial Transformer from VoxelMorph."""
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode

        # Create a sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids).unsqueeze(0).float()
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Normalize grid to [-1, 1]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # Move channels to last dim and reorder for grid_sample
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=False, mode=self.mode)


# -------------------- Registration Model Wrapper --------------------
class RegisterModel(nn.Module):
    """Wrapper to apply Spatial Transformer."""
    def __init__(self, img_size=(128,128,32), mode='bilinear'):
        super().__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, x):
        img = x[0].cuda()
        flow = x[1].cuda()
        return self.spatial_trans(img, flow)