import torch
import torch.nn as nn
import torch.nn.functional as F

from down_conference.utils.ops import *

class OccupancyDecoder(nn.Module):
    def __init__(self, embed_dim=11, hidden_dim=128, num_classes=18):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes  # 사용은 안 하지만 유지

        # Gaussian 임베딩 처리 MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, gaussian_embed, voxel_coords, batch_size=32):
        """
        gaussian_embed: dict of tensors (B, N, 11)
        voxel_coords: (Nv, 4) — sparse voxel: [batch_idx, x, y, z]
        """
        # 1. Gaussian concat → (B, N, 11)
        x = torch.cat([
            gaussian_embed['position'],
            gaussian_embed['scale'],
            gaussian_embed['rotation'],
            gaussian_embed['opacity']
        ], dim=-1)  # (B, N, 11)

        B, N, _ = x.shape
        x = self.mlp(x)  # (B, N, hidden_dim)

        # 2. anchor grid 생성
        anchor_pos = torch.linspace(-50, 50, int(N ** (1 / 3)), device=x.device)
        grid = torch.stack(torch.meshgrid(anchor_pos, anchor_pos, anchor_pos, indexing="ij"), dim=-1)
        anchor_grid = grid.reshape(-1, 3)  # (N, 3)

        # 3. sparse voxel 분리
        batch_ids = voxel_coords[:, 0].long()  # (Nv,)
        xyz = voxel_coords[:, 1:].float()  # (Nv, 3)

        xyz = xyz.half()
        anchor_grid = anchor_grid.half()

        # xyz: (Nv, 3), anchor_grid: (N, 3)
        dist = chunked_cdist(xyz, anchor_grid, chunk_size=512)  # (Nv, N)
        weight = chunked_softmax(-dist, dim=-1, chunk_size=512)

        # 5. Gaussian weighted sum
        # x: (B, N, hidden_dim), weight: (Nv, N)
        fused = torch.einsum('vn,bnd->bvd', weight, x)  # (B, Nv, hidden_dim)

        # 6. voxel feature grid 초기화
        x_max = int(voxel_coords[:, 1].max().item()) + 1
        y_max = int(voxel_coords[:, 2].max().item()) + 1
        z_max = int(voxel_coords[:, 3].max().item()) + 1
        occ_feat = torch.zeros((B, self.hidden_dim, x_max, y_max, z_max), device=x.device)

        # 7. 배치별 scatter
        for b in range(B):
            mask = (batch_ids == b)
            coords_b = voxel_coords[mask][:, 1:].long()  # (Vb, 3)
            fused_b = fused[b][mask]  # (Vb, C)
            x_, y_, z_ = coords_b[:, 0], coords_b[:, 1], coords_b[:, 2]
            occ_feat[b, :, x_, y_, z_] = fused_b.T  # (C, Vb)

        return occ_feat  # (B, C, X, Y, Z)



