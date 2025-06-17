import torch

from config import config

def scatter_to_grid(features, coords, grid_size, mode='mean'):
    """
    Args:
        features: (B, N, C)  — anchor-wise features
        coords:   (B, N, 3)  — 각 anchor의 voxel 좌표 (x, y, z), 정수값
        grid_size: (X, Y, Z) — 최종 grid 크기
        mode: 'mean' or 'sum'

    Returns:
        voxel_grid: (B, C, X, Y, Z)
    """
    B, N, C = features.shape
    X, Y, Z = grid_size
    device = features.device

    voxel_grid = torch.zeros((B, C, X, Y, Z), device=config['device'])
    count_grid = torch.zeros((B, 1, X, Y, Z), device=config['device'])

    for b in range(B):
        for i in range(N):
            x, y, z = coords[b, i]
            if 0 <= x < X and 0 <= y < Y and 0 <= z < Z:
                voxel_grid[b, :, x, y, z] += features[b, i]
                count_grid[b, :, x, y, z] += 1

    if mode == 'mean':
        count_grid = count_grid.clamp(min=1.0)  # prevent div 0
        voxel_grid = voxel_grid / count_grid

    return voxel_grid  # (B, C, X, Y, Z)
