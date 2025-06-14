"""
semantic label 설명
semantic label은 uint8로 저장되어 있음 → nuScenes에서는 32개 클래스 (index 0~31)

비어 있는 voxel은 -1로 처리

원한다면 one-hot 처리하여 semantic_class_map = (N_class, D, H, W)로 확장 가능

모델 입력으로 넣기 위해선 (B, C, D, H, W) 형태로 바꿔야 할 수도 있음

다채널 입력을 만들고 싶다면 intensity, return num 등을 활용해 C 채널 추가 가능

output은 (B, num_classes, D, H, W) 형태로 예측

"""
import os
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from typing import List, Dict
from tqdm import tqdm

from down_conference.config import config
from nuscenes.nuscenes import NuScenes

from down_conference.data.data_func import load_all_lidar_samples , load_multiview_samples , camera_visualization

nusc = NuScenes(version='v1.0-trainval', dataroot='D:/nuscene_data', verbose=True)

nusc_num = len(nusc.sample)

if config['flag'] == 'LIDAR':
    # LIDAR
    lidar_data = load_all_lidar_samples(nusc , nusc_num)
else:
    # Camera
    camera_data = load_multiview_samples(nusc, 3)
    #camera_visualization(camera_data[0]['sample_token'], camera_data[0]['images']) # : sampling
    #camera_visualization(camera_data[1]['sample_token'], camera_data[1]['images']) : sampling
