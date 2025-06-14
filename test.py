import os
from down_conference.data.data_process import nusc, camera_data
from down_conference.backbone.backbone_generate import features
from down_conference.neck.neck_generate import sample_token_list

missing = []
for token in sample_token_list:
    sample = nusc.get('sample', token)
    lidar_token = sample['data']['LIDAR_TOP']
    path = os.path.join("D:/lidar", f"{lidar_token}.bin")
    if not os.path.exists(path):
        missing.append(token)

print(f"Missing LIDAR files for {len(missing)} sample_tokens")