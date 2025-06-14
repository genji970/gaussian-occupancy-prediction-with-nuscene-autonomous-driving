import torch
import torch.nn as nn
from spconv.pytorch import SparseConvTensor, SubMConv3d, SparseSequential


from spconv.pytorch import SparseConvTensor

class DeformableAttentionWithSpconv(nn.Module):
    def __init__(self, embed_dim=64, num_points=8,
                 grid_size=(128, 128, 16), voxel_size=(0.5, 0.5, 0.5), pc_range=(0, 0, 0)):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_points = num_points
        self.grid_size = grid_size
        self.voxel_size = torch.tensor(voxel_size)
        self.pc_range = torch.tensor(pc_range)

        self.sparse_net = SparseSequential(
            SubMConv3d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            SubMConv3d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def quantize_keypoints(self, keypoints):
        B, N, K, _ = keypoints.shape
        coords = (keypoints - self.pc_range.to(keypoints.device)) / self.voxel_size.to(keypoints.device)
        min_val = torch.zeros_like(coords)  # (B, N, 3)
        max_val = torch.tensor(self.grid_size, device=coords.device) - 1  # (3,)
        max_val = max_val.view(1, 1, 3).expand_as(coords)  # (B, N, 3)로 브로드캐스트

        coords = coords.long().clamp(min=min_val, max=max_val)

        batch_idx = torch.arange(B, device=coords.device).view(B, 1, 1, 1).expand(B, N, K, 1)
        full_coords = torch.cat([batch_idx, coords], dim=-1)  # (B, N, K, 4)
        return full_coords

    def forward(self, keypoints, query_feature, voxel_feature, voxel_coords):
        """
        Args:
            keypoints: (B, N, K, 3)
            query_feature: (B, N, C)
            voxel_feature: (B, V, C)
            voxel_coords: (B, V, 3)
        Returns:
            fused: (B, N, C)
        """
        B, N, C = query_feature.shape
        K = keypoints.shape[2]
        V = voxel_feature.shape[1]

        # 1. Make SparseConvTensor from voxel_feature & voxel_coords
        batch_idx = torch.arange(B, device=query_feature.device).view(B, 1).expand(B, V)  # (B, V)
        batch_idx = batch_idx.reshape(-1, 1)  # (B*V, 1)
        coords = voxel_coords.reshape(-1, 3).long()  # (B*V, 3)
        indices = torch.cat([batch_idx, coords], dim=1)  # (B*V, 4)
        features = voxel_feature.reshape(-1, C)  # (B*V, C)

        sparse_tensor = SparseConvTensor(
            features=features,
            indices=indices.int(),
            spatial_shape=self.grid_size,
            batch_size=B
        )

        # 2. SparseConv
        sparse_out = self.sparse_net(sparse_tensor)
        feature_bank = sparse_out.features  # (N_voxel, C)
        voxel_hash = sparse_out.indices     # (N_voxel, 4)

        # 3. Quantize keypoints to voxel indices
        voxel_indices = self.quantize_keypoints(keypoints).view(-1, 4)  # (B*N*K, 4)

        # 4. Hash table
        hash_map = {tuple(idx.tolist()): i for i, idx in enumerate(voxel_hash)}
        matched_idx = [hash_map.get(tuple(v.tolist()), 0) for v in voxel_indices]
        matched_idx = torch.tensor(matched_idx, dtype=torch.long, device=query_feature.device)
        sampled = feature_bank[matched_idx]  # (B*N*K, C)
        sampled = sampled.view(B, N, K, C)

        # 5. Aggregate
        sampled_mean = sampled.mean(dim=2)  # (B, N, C)

        # 6. Fuse
        fused = self.output_proj(sampled_mean + query_feature)  # (B, N, C)
        return fused , voxel_indices

