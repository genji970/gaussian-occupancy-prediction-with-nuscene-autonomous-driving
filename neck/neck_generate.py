import os
import numpy as np

from mmdet.models.necks import FPN
from mmdet.registry import MODELS
import torch

from neck.cfg import neck_cfg

from config import config
from utils.neck_func import backproject_to_voxel
from data.data_process import camera_data
from backbone.backbone_generate import features
from utils.neck_func import neck_process
"""

args : features: (List of List of Tensor)

"""
# 1) neck 생성
neck = MODELS.build(neck_cfg)

# 2) neck에 features 넣어 처리
neck_features = neck_process(features, neck, camera_data, config['neck_batch_num'])  # List[Dict], 각 Dict 안에 'features' key

# 3) cam_features 리스트와 텐서로 변환
cam_features_list = [sample['features'] for sample in neck_features]  # List[B] of tensor (6, C, H, W)
cam_features_tensor = torch.stack(cam_features_list, dim=0)          # (B, 6, C, H, W)
device = cam_features_tensor.device

# 4) camera_data에서 cam_Ks, cam_Ts 텐서 만들기 + device 일치시킴
cam_Ks_batch = torch.stack([sample['cam_Ks'] for sample in camera_data], dim=0).to(device)  # (B, 6, 3, 3)
cam_Ts_batch = torch.stack([sample['cam_Ts'] for sample in camera_data], dim=0).to(device)  # (B, 6, 4, 4)

"""
print(cam_features_tensor.shape)
print(cam_Ks_batch.shape)
print(cam_Ts_batch.shape)
"""
sample_tokens = [n['sample_token'] for n in neck_features]

voxel_features, voxel_coords, sample_token_list = backproject_to_voxel(
    sample_token=sample_tokens, # (B,N)
    cam_features=cam_features_tensor, # (B, 6, C_out, H, W)
    cam_Ks=cam_Ks_batch, # (B, 6, 3, 3)
    cam_Ts=cam_Ts_batch, # (B, 6, 4, 4)
    grid_size=(200, 200, 16),
    voxel_size=(0.5, 0.5, 0.5),
    pc_range=(-50, -50, -3)
)

# 파일로 저장
B_ = voxel_features.shape[0]  # 배치 크기

for i in range(0, B_):
    global_idx = i

    # voxel_features 저장 경로
    save_feat_path = os.path.join(config['save_root'], "voxel_features", f"voxel_features_{global_idx}.npz")
    if i == 0:
        os.makedirs(os.path.dirname(save_feat_path), exist_ok=True)
    np.savez_compressed(save_feat_path, voxel_features=voxel_features[i].cpu().numpy())

    # voxel_coords 저장 경로
    save_coord_path = os.path.join(config['save_root'], "voxel_coords", f"voxel_coords_{global_idx}.npz")
    if i == 0:
        os.makedirs(os.path.dirname(save_coord_path), exist_ok=True)
    np.savez_compressed(save_coord_path, voxel_coords=voxel_coords[i].cpu().numpy())





# pc_range = (-50, -50, -3)
# pc_range_max = (50, 50, 1)
# X = int((x_max - x_min) / voxel_size[0])
# Y = int((y_max - y_min) / voxel_size[1])
# Z = int((z_max - z_min) / voxel_size[2])
