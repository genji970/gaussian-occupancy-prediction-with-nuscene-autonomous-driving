import torch
from config import config
from mmseg.registry import MODELS
from mmseg.models.backbones import SwinTransformer
from backbone.cfg import backbone_cfg
# 2. backbone 생성
backbone = MODELS.build(backbone_cfg) # backbone ouput : feature
backbone = backbone.to(torch.device('cuda'))
images = torch.randn(1, 3, 224, 224).to(config['device'])  # 예시 입력
backbone.eval()
with torch.no_grad():
    out_feats = backbone(images)

if isinstance(out_feats, (list, tuple)):
    for i, feat in enumerate(out_feats):
        print(f"Stage {i}: {feat.shape}")
