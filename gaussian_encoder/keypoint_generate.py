import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class SparseGaussian3DKeyPointsGenerator(nn.Module):
    def __init__(self, num_keypoints=8, std=0.5):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.std = std
        # Learnable normalized offsets around origin
        self.offset_template = nn.Parameter(torch.randn(num_keypoints, 3))  # (K, 3)

    def forward(self, anchor_position, anchor_scale):
        """
        Args:
            anchor_position: (B, N, 3)
            anchor_scale:    (B, N, 3)
        Returns:
            keypoints:       (B, N, K, 3)
        """
        B, N, _ = anchor_position.shape
        K = self.num_keypoints

        # (1, 1, K, 3) — learnable normalized offsets
        offset_template = self.offset_template.unsqueeze(0).unsqueeze(0)  # (1, 1, K, 3)

        # (B, N, 1, 3) × (1, 1, K, 3) → (B, N, K, 3)
        scaled_offset = offset_template * anchor_scale.unsqueeze(2) * self.std

        # keypoints = anchor 중심 + offset
        keypoints = anchor_position.unsqueeze(2) + scaled_offset  # (B, N, K, 3)

        return keypoints

