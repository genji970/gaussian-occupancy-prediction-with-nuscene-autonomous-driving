from mmseg.registry import MODELS
from mmseg.models.backbones import SwinTransformer
import torch

from backbone.cfg import backbone_cfg

from config import config
from data.data_process import camera_data
from utils.backbone_func import process_camera_data_batch

# 2. backbone 생성
backbone = MODELS.build(backbone_cfg) # backbone ouput : feature

# 3. 멀티뷰 feature 추출
features = process_camera_data_batch(camera_data, backbone, batch_size=config['backbone_batch_num'])
# features : List[(index, [6][4][Tensor(C, H, W)])]
