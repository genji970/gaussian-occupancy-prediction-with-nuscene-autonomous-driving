import torch
import torch.nn as nn
import torch.nn.functional as F

from down_conference.config import config
from down_conference.gaussian_encoder.deformable_attention import DeformableAttentionWithSpconv
from down_conference.gaussian_encoder.keypoint_generate import SparseGaussian3DKeyPointsGenerator
from down_conference.gaussian_encoder.refinement import SparseGaussian3DRefinement

class GaussianEncoder(nn.Module):  # (B, N, 11) → (B, N, D)
    def __init__(self, input_dim=11, embed_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )

        self.keypoint_gen = SparseGaussian3DKeyPointsGenerator(num_keypoints=8, std=0.5)
        self.attn = DeformableAttentionWithSpconv(embed_dim=64, num_points=8,
                 grid_size=(128, 128, 16), voxel_size=(0.5, 0.5, 0.5), pc_range=(0, 0, 0))
        self.refiner = SparseGaussian3DRefinement(hidden_dim=64)

    def forward(self, gaussians, voxel_features, voxel_coords):
        pos = gaussians['position'].to(config['device'])
        scale = gaussians['scale'].to(config['device'])
        rot = gaussians['rotation'].to(config['device'])
        opac = gaussians['opacity'].to(config['device'])

        x = torch.cat([pos, scale, rot, opac], dim=-1)  # (B, N, 11)
        emb = self.encoder(x)  # (B, N, D)

        keypoints = self.keypoint_gen.forward(pos, scale)  # (B, N, 3)

        fused_feature , voxel_coords = self.attn.forward(
            keypoints=keypoints,
            query_feature=emb,
            voxel_feature=voxel_features,
            voxel_coords=voxel_coords
        )

        refined = self.refiner.forward(fused_feature , gaussians)  # (B, N, D)
        return refined , voxel_coords

