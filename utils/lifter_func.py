import os
import numpy as np
from typing import List, Dict
import torch
import torch.nn as nn

from down_conference.config import config

def lifter_preprocess(
    neck_features: torch.Tensor,  # (B, num_cams, C, H, W)
    lifter: nn.Module,
    cam_Ks: torch.Tensor,          # (B, num_cams, 3, 3)
    cam_Ts: torch.Tensor,          # (B, num_cams, 4, 4)
    batch_size: int = 8,
    save_root: str = config['save_root']  # 저장 경로 인자로 받기
) -> List[Dict]:
    device = neck_features.device
    lifter = lifter.to(device)
    cam_Ks = cam_Ks.to(device)
    cam_Ts = cam_Ts.to(device)

    lifter.eval()
    B, num_cams, C, H, W = neck_features.shape

    os.makedirs(save_root, exist_ok=True)
    all_outputs = [None] * B  # index 기반으로 저장

    for i in range(0, B, batch_size):
        batch_feats = neck_features[i:i+batch_size]    # (B', num_cams, C, H, W)
        batch_Ks = cam_Ks[i:i+batch_size]              # (B', num_cams, 3, 3)
        batch_Ts = cam_Ts[i:i+batch_size]              # (B', num_cams, 4, 4)

        cam_feats = [batch_feats[:, cam_idx] for cam_idx in range(num_cams)]  # List[num_cams] of (B', C, H, W)

        with torch.no_grad():
            out = lifter(cam_feats, batch_Ks, batch_Ts)  # output: Dict[str, Tensor (B', N, ?)]

        B_ = batch_feats.shape[0]
        for j in range(B_):
            global_idx = i + j
            sample_data = {
                'position': out['position'][j].cpu().numpy(),  # (N, 3)
                'scale': out['scale'][j].cpu().numpy(),        # (N, 3)
                'rotation': out['rotation'][j].cpu().numpy(),  # (N, 4)
                'opacity': out['opacity'][j].cpu().numpy(),    # (N, 1)
            }

            # 파일로 저장
            save_path = os.path.join(save_root, "gaussian", f"gaussian_output_{global_idx}.npz")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez_compressed(save_path, **sample_data)

            """
            # 메모리 보관 
            all_outputs[global_idx] = {
                'index': global_idx,
                **sample_data  # position, scale, ...
            }
            """

    #return all_outputs  # List[B] of Dicts


