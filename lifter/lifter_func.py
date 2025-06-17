import os
import numpy as np

from mmengine.model import BaseModule
from mmengine.registry import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F


from config import config

@MODELS.register_module()
class AnchorMultiViewLifter(BaseModule):
    def __init__(self,
                 num_anchor=6400,
                 init_anchor_range=((-50, 50), (-50, 50), (-3, 3)),
                 num_cams=6,
                 feat_dim=config['lifter_feat_dim'],
                 hidden_dim=config['lifter_hidden_dim'],
                 init_cfg=None):
        super().__init__(init_cfg)

        self.num_anchor = num_anchor
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # ✅ Learnable anchor initialization: (N, 3)
        anchor_xyz = torch.rand(num_anchor, 3)
        for i in range(3):
            min_val, max_val = init_anchor_range[i]
            anchor_xyz[:, i] = anchor_xyz[:, i] * (max_val - min_val) + min_val
        self.anchor = nn.Parameter(anchor_xyz)  # (N, 3), learnable

        # MLP for Gaussian parameter prediction
        self.param_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 11)  # offset(3), scale(3), rot(4), opacity(1)
        )

    def forward(self, cam_feats, cam_Ks, cam_Ts):
        """
        cam_feats: List of length num_cams, each [B, C, H, W]
        cam_Ks:    [B, num_cams, 3, 3]
        cam_Ts:    [B, num_cams, 4, 4]  # world→cam
        """
        B = cam_feats[0].shape[0]   # batch size
        N = self.anchor.shape[0]    # number of anchors
        device = cam_feats[0].device
        num_cams = len(cam_feats)

        anchor_homo = torch.cat([self.anchor, torch.ones(N, 1, device=device)], dim=-1)  # (N, 4)

        all_anchor_feats = []

        for b in range(B):
            per_view_feats = []
            for cam_idx in range(num_cams):
                K = cam_Ks[b, cam_idx]  # (3, 3)
                T = cam_Ts[b, cam_idx]  # (4, 4)

                xyz_cam = (T @ anchor_homo.T).T[:, :3]  # (N, 3)

                proj = (K @ xyz_cam.T).T  # (N, 3)
                uv = proj[:, :2] / (proj[:, 2:3] + 1e-6)  # (N, 2)

                _, C, H, W = cam_feats[cam_idx].shape

                uv_norm = uv.clone()
                uv_norm[:, 0] = (uv[:, 0] / W) * 2 - 1
                uv_norm[:, 1] = (uv[:, 1] / H) * 2 - 1
                grid = uv_norm.view(1, N, 1, 2)  # (1, N, 1, 2)

                sampled = F.grid_sample(
                    cam_feats[cam_idx][b].unsqueeze(0),  # (1, C, H, W)
                    grid,
                    align_corners=True,
                    mode='bilinear',
                    padding_mode='zeros'
                ).squeeze(-1).squeeze(0).T  # (N, C)

                per_view_feats.append(sampled)

            fused = torch.stack(per_view_feats, dim=0).mean(dim=0)  # (N, C)
            all_anchor_feats.append(fused)

        all_anchor_feats = torch.stack(all_anchor_feats, dim=0)  # (B, N, C)
        out = self.param_head(all_anchor_feats)  # (B, N, 11)

        offset   = out[..., :3]
        scale    = torch.exp(out[..., 3:6])
        rotation = F.normalize(out[..., 6:10], dim=-1)
        opacity  = torch.sigmoid(out[..., 10:11])
        position = self.anchor[None, :, :] + offset

        output = {
            'position': position,
            'scale': scale,
            'rotation': rotation,
            'opacity': opacity,
        }
        return output


