"""
data_process -> backbone_generate -> neck_generate -> lifter_generate

[Gaussian Input]
    ↓
[GaussianEncoder]                 ← 인코더
    ↓
[Sparse Keypoint Generator]
    ↓
[DeformableFeatureAggregation]   ← 멀티뷰 피처 결합
    ↓
[Refinement Module (optional)]
    ↓
[OccupancyDecoder]               ← 디코더
    ↓
[Occupancy Logits / Semantic class]


######################

# ============================== NuScenes 데이터 구조 ==============================
#
# D:/nuscene_data/
# ├── maps/                        # 지도 데이터
# ├── samples/                    # 현재 시점 센서 데이터
# │   ├── CAM_BACK
# │   ├── CAM_BACK_LEFT
# │   ├── CAM_BACK_RIGHT
# │   ├── CAM_FRONT
# │   ├── CAM_FRONT_LEFT
# │   ├── CAM_FRONT_RIGHT
# │   ├── LIDAR_TOP
# │   ├── RADAR_BACK_LEFT
# │   ├── RADAR_BACK_RIGHT
# │   ├── RADAR_FRONT
# │   ├── RADAR_FRONT_LEFT
# │   └── RADAR_FRONT_RIGHT
# ├── sweeps/                     # 과거 시점 센서 시퀀스
# │   ├── (CAM_*, LIDAR_TOP, RADAR_*)
# └── v1.0-trainval/              # 메타데이터 (JSON 파일)
#     ├── attribute.json
#     ├── calibrated_sensor.json
#     ├── category.json
#     ├── ego_pose.json
#     ├── instance.json
#     ├── log.json
#     ├── map.json
#     ├── sample.json
#     ├── sample_annotation.json
#     ├── sample_data.json
#     ├── scene.json
#     ├── sensor.json
#     └── visibility.json
#
# ==================================================================================

# ========================== NuScenes Label 데이터 구조 ==========================
#
# D:/nuscene_label/
# ├── lidarseg/                          # LiDAR segmentation 라벨 (BIN 파일)
# │   ├── v1.0-mini/
# │   ├── v1.0-test/
# │   └── v1.0-trainval/
# │       └── <LIDAR_TOP_token>_lidarseg.bin  # 예: 0000fa1a7bfb46dc..._lidarseg.bin
#
# ├── v1.0-mini/                         # (선택적으로 있을 수 있음)
# ├── v1.0-test/
# ├── v1.0-trainval/                     # JSON 메타데이터가 포함될 수 있는 구조
# │   ├── category.json                  # 클래스 정보
# │   └── (기타 메타 JSON은 선택사항)
#
# ==============================================================================

"""

import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from config import config

from data.data_process import nusc, camera_data
from backbone.backbone_generate import features
from neck.neck_generate import sample_token_list
from lifter.lifter_generate import *

from utils.train_load import OccupancyDataset
from down_conference.gaussian_encoder.gaussian_encode import GaussianEncoder
from gaussian_encoder.gaussian_decoder.gaussian_decoder import OccupancyDecoder
from head.head import OccupancyHead


# ===== 1. 모델 구성 =====
gaussian_encoder = GaussianEncoder(input_dim=11, embed_dim=64).to(config['device'])
occupancy_decoder = OccupancyDecoder(embed_dim=11, num_classes=18, hidden_dim=64).to(config['device'])
occupancy_head = OccupancyHead(in_dim=64, grid_size=(100, 100, 8), num_classes=18).to(config['device'])

# ===== 2. Optimizer =====
params = list(gaussian_encoder.parameters()) + \
         list(occupancy_decoder.parameters()) + \
         list(occupancy_head.parameters())

optimizer = torch.optim.Adam(params, lr=1e-4)

# ===== 3. Dataset / DataLoader =====
dataset = OccupancyDataset(
    data_dir="D:/",
    token_ids=sample_token_list,
    nusc=nusc
)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False)

num_epochs = 30

scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in train_loader:
        # -------- 데이터 준비 --------
        voxel_feat = batch['voxel_feat'].to(config['device'])
        voxel_coords = batch['voxel_coords'].to(config['device'])
        label = batch['label'].to(config['device'])
        gaussians = batch['gaussians']  # dict 타입

        optimizer.zero_grad()

        # -------- Forward + Loss --------
        with autocast():
            gaussian_embed, voxel_coords = gaussian_encoder(gaussians, voxel_feat, voxel_coords)
            occ_feat = occupancy_decoder(gaussian_embed, voxel_coords)
            occ_logits = occupancy_head(occ_feat)

            loss = F.cross_entropy(
                occ_logits,
                label,
                ignore_index=255
            )

        # -------- Backward --------
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

        # -------- 메모리 해제 --------
        del occ_logits, occ_feat, gaussian_embed
        del voxel_feat, voxel_coords, label, gaussians, loss

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
