# camera_data backbone에 입력
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import List, Dict , Tuple

from down_conference.config import config
from down_conference.backbone.cfg import backbone_cfg

def extract_multiview_features(backbone: nn.Module, images: torch.Tensor) -> List[torch.Tensor]:
    multiview_features = []
    for i in range(images.shape[0]):
        img = images[i].unsqueeze(0)  # (1, 3, H, W)
        feat = backbone(img)
        if isinstance(feat, (tuple, list)):
            # 유지: [stage1, stage2, stage3, stage4]
            feat = [f.squeeze(0) for f in feat]  # remove batch dim
        else:
            feat = [feat.squeeze(0)]
        multiview_features.append(feat)  # 각 카메라에 대해 [stage1, stage2, stage3, stage4]
    return multiview_features  # shape: List[6][4][C, H, W]

def process_camera_data_batch(
    camera_data: List[Dict],
    backbone: nn.Module,
    batch_size: int = 1,
) -> List[Tuple[int, List[List[torch.Tensor]]]]:  # → List[(index, feature)]

    backbone = backbone.to(config['device'])
    backbone.eval()

    stacked_batch: List[Tuple[int, List[List[torch.Tensor]]]] = []

    with torch.no_grad():
        for i in range(0, len(camera_data), batch_size):
            batch = camera_data[i:i + batch_size]

            for j, sample in enumerate(batch):
                global_idx = i + j
                images = sample['images'].to(config['device'])  # (6, 3, H, W)

                view_features = extract_multiview_features(backbone, images)  # List[6][4][C, H, W]

                sample_tensor = [
                    [f.to(config['device']) for f in stage_feats]
                    for stage_feats in view_features
                ]

                # global_idx와 함께 feature 묶기
                stacked_batch.append((global_idx, sample_tensor))  # (index, (6 views × 4 stages))

    return stacked_batch  # List[(index, [6][4][Tensor(C, H, W)])]


"""
B: 샘플 수

6: 카메라(view) 수

4: SwinTransformer의 stage 수

C, H, W: 각 stage별 채널/크기
"""