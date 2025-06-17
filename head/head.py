import torch
import torch.nn as nn

from utils.head_func import scatter_to_grid

class OccupancyHead(nn.Module):
    def __init__(self, in_dim=64, grid_size=(100, 100, 8), num_classes=18):
        super().__init__()
        self.grid_size = grid_size  # (X, Y, Z)
        self.num_classes = num_classes

        # in_dim → num_classes를 위한 1x1 conv
        self.cls_layer = nn.Conv3d(in_dim, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, in_dim, X, Y, Z) — voxel feature map
        Returns:
            voxel_logits: (B, num_classes, X, Y, Z)
        """
        return self.cls_layer(x)  # 1x1x1 conv → (B, num_classes, X, Y, Z)


