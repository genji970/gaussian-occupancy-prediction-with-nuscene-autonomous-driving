import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config

class SparseGaussian3DRefinement(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        output_dim = 3 + 3 + 4 + 1  # delta for position + scale + rotation + opacity

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 예측할 delta
        )

    def forward(self, fused_feature, gaussians):
        """
        Args:
            fused_feature: (B, N, D)
            gaussians: dict {
                'position': (B, N, 3),
                'scale':    (B, N, 3),
                'rotation': (B, N, 4),
                'opacity':  (B, N, 1)
            }
        Returns:
            dict: refined gaussian params
        """
        delta = self.mlp(fused_feature)  # (B, N, 11)
        d_pos, d_scale, d_rot, d_opa = torch.split(delta, [3, 3, 4, 1], dim=-1)

        pos = gaussians['position'].to(config['device'])
        scale = gaussians['scale'].to(config['device'])
        rot = gaussians['rotation'].to(config['device'])
        opa = gaussians['opacity'].to(config['device'])

        refined = {
            'position': pos + d_pos,
            'scale': scale + d_scale,
            'rotation': F.normalize(rot + d_rot, dim=-1),
            'opacity': opa + d_opa
        }
        return refined

