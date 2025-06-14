import torch

from down_conference.config import config
from down_conference.lifter.lifter_func import AnchorMultiViewLifter
from down_conference.neck.neck_generate import neck_features
from down_conference.utils.lifter_func import lifter_preprocess
from down_conference.neck.neck_generate import cam_Ks_batch , cam_Ts_batch

# lifter 클래스 인스턴스 생성
lifter = AnchorMultiViewLifter(
    num_anchor=config['lifter_num_anchor'],
    init_anchor_range=((-50, 50), (-50, 50), (-3, 3)),
    num_cams=6,
    feat_dim=config['lifter_feat_dim'],
    hidden_dim=config['lifter_hidden_dim']
)

"""
print(neck_features)
print(cam_Ks_batch.shape) # torch.Size([3, 6, 3, 3])
print(cam_Ts_batch.shape) # torch.Size([3, 6, 4, 4])
"""

neck_features_tensor = torch.stack([sample['features'] for sample in neck_features], dim=0)

#output['position'][b, n]  # b번째 샘플의 n번째 anchor 위치 → [x, y, z]
lifter_preprocess(neck_features_tensor, lifter, cam_Ks_batch, cam_Ts_batch) # gaussian output save to directory

# 4. 결과: dict { 'position': ..., 'scale': ..., ... }
#(gaussians['position'].shape)  # → (B, N, 3) , torch.Size([3, 25600, 3])

"""
Return -> 
{   'index'   : int,
    'position': (B, N, 3),
    'scale':    (B, N, 3),
    'rotation': (B, N, 4),
    'opacity':  (B, N, 1)
}

"""