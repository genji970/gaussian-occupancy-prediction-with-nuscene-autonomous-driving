import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from neck.cfg import neck_cfg

from config import config


def neck_process(
    features: List[Tuple[int, List[List[torch.Tensor]]]],  # List[(index, [6][4][Tensor(C,H,W)])]
    neck: nn.Module,
    camera_data: List[Dict],
    batch_size: int = 1,
) -> List[Dict]:
    neck = neck.to(config['device'])
    neck.eval()
    results = [None] * len(camera_data)

    # 정렬 보장 위해 index 기준으로 정렬 (선택적)
    features = sorted(features, key=lambda x: x[0])

    for i in range(0, len(features), batch_size):
        batch_feats_with_idx = features[i:i + batch_size]  # List[(index, [6][4])]
        fused_batch = []

        with torch.no_grad():
            for index, multiview_feats in batch_feats_with_idx:
                fused_views = []

                for view_feats in multiview_feats:
                    raw_feats = view_feats
                    if isinstance(raw_feats[0], list):
                        raw_feats = [x for sublist in raw_feats for x in sublist]

                    per_view_feats = [f.to(config['device']) for f in raw_feats]

                    max_h = max(f.shape[1] for f in per_view_feats)
                    max_w = max(f.shape[2] for f in per_view_feats)

                    per_view_feats = [F.interpolate(f.unsqueeze(0), size=(max_h, max_w), mode='nearest').squeeze(0)
                                      if (f.shape[1] != max_h or f.shape[2] != max_w) else f
                                      for f in per_view_feats]

                    per_view_feats = list(reversed(per_view_feats))

                    out = neck(per_view_feats)
                    while isinstance(out, (list, tuple)):
                        out = out[-1]

                    fused_views.append(out)

                fused_views_tensor = torch.stack(fused_views, dim=0)  # (6, C, H, W)
                fused_batch.append((index, fused_views_tensor))

        for index, fused in fused_batch:
            meta = camera_data[index]
            results[index] = {
                'features': fused,
                'cam_Ks': meta['cam_Ks'],
                'cam_Ts': meta['cam_Ts'],
                'sample_token': meta['sample_token']
            }

    return results





"""

1) torch_scatter 등을 써서 voxel aggregation을 더 효율적으로 할 수도 있고,
2) depth map이 있다면 더 정밀하게 투영 가능.

✅ 1. 전체 공간의 범위 (pc_range)
즉, "어디부터 어디까지 3D 공간을 커버할 것인가?" 예: (-50, -50, -3) ~ (50, 50, 1)

✅ 2. 각 voxel의 실제 크기 (voxel_size)
예: voxel_size = (0.5m, 0.5m, 0.25m)
→ voxel 하나가 실제로 커버하는 물리적 공간

X = int((x_max - x_min) / voxel_size[0])
Y = int((y_max - y_min) / voxel_size[1])
Z = int((z_max - z_min) / voxel_size[2])

"""

def backproject_to_voxel(sample_token, cam_features, cam_Ks, cam_Ts, grid_size, voxel_size, pc_range):
    """
    Args:
        sample_token: list of str (길이 B)
        cam_features: Tensor (B, 6, C, H, W)
        cam_Ks:       Tensor (B, 6, 3, 3)
        cam_Ts:       Tensor (B, 6, 4, 4)
        grid_size:    (X, Y, Z)
        voxel_size:   (vx, vy, vz)
        pc_range:     (x0, y0, z0)

    Returns:
        voxel_features: Tensor (B, V_valid, C)
        voxel_coords:   Tensor (B, V_valid, 3)
        sample_token_list:   List[str] (길이 B), 각 sample의 token ID
    """
    import torch

    B, num_views, C, H, W = cam_features.shape
    X, Y, Z = grid_size
    device = cam_features.device

    x = torch.linspace(0, X - 1, X, device=device) * voxel_size[0] + pc_range[0]
    y = torch.linspace(0, Y - 1, Y, device=device) * voxel_size[1] + pc_range[1]
    z = torch.linspace(0, Z - 1, Z, device=device) * voxel_size[2] + pc_range[2]
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)  # (V, 3)
    V = xyz.shape[0]
    xyz_h = torch.cat([xyz, torch.ones((V, 1), device=device)], dim=1).T  # (4, V)

    voxel_features_list = []
    voxel_coords_list = []
    sample_token_list = []

    for b in range(B):
        feat_sum = torch.zeros((V, C), device=device)
        valid_count = torch.zeros(V, device=device)

        for view_idx in range(num_views):
            K = cam_Ks[b, view_idx]
            T = cam_Ts[b, view_idx]
            P = K @ T[:3, :]

            proj = P @ xyz_h
            uv = proj[:2] / (proj[2:] + 1e-6)

            u = uv[0].round().long()
            v_ = uv[1].round().long()

            mask = (u >= 0) & (u < W) & (v_ >= 0) & (v_ < H)
            valid_idx = mask.nonzero(as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                continue

            sampled_feat = cam_features[b, view_idx, :, v_[valid_idx], u[valid_idx]].T
            feat_sum[valid_idx] += sampled_feat
            valid_count[valid_idx] += 1

        valid_mask = valid_count > 0
        feat_sum[valid_mask] /= valid_count[valid_mask].unsqueeze(1)

        coords = xyz[valid_mask]
        voxel_features_list.append(feat_sum[valid_mask])
        voxel_coords_list.append(coords)

        # ✅ sample_token 그대로 추가
        sample_token_list.append(sample_token[b])

    voxel_features = torch.stack(voxel_features_list, dim=0)
    voxel_coords = torch.stack(voxel_coords_list, dim=0)

    return voxel_features, voxel_coords, sample_token_list











