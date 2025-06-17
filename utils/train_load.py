import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import Dict, List, Tuple
from nuscenes.nuscenes import NuScenes
from data.data_process import NuScenes

from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
import numpy as np
import torch
import os
from typing import List
from collections import Counter

class OccupancyDataset(Dataset):
    def __init__(self, data_dir: str, token_ids: List[str], nusc: NuScenes):
        self.data_dir = data_dir
        self.token_ids = token_ids  # List of sample_token
        self.nusc = nusc

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        sample_token = self.token_ids[idx]
        index_id = str(idx)

        voxel_features = self._load_voxel_feature(index_id)  # (V, C)
        voxel_coords = self._load_voxel_coords(index_id)  # (V, 3)
        gaussians = self._load_gaussian(index_id)  # dict
        label = self._load_voxel_label(sample_token)  # (200, 200, 16)

        return {
            'voxel_feat': voxel_features,  # (V, C)
            'voxel_coords': voxel_coords,  # (V, 3)
            'gaussians': gaussians,  # 각 key: (V, ...)
            'label': label  # (200, 200, 16)
        }

    def _load_voxel_feature(self, index_id):
        path = os.path.join(self.data_dir, "voxel_features", f"voxel_features_{index_id}.npz")
        data = np.load(path)
        return torch.tensor(data['voxel_features'], dtype=torch.float32)

    def _load_voxel_coords(self, index_id):
        path = os.path.join(self.data_dir, "voxel_coords", f"voxel_coords_{index_id}.npz")
        data = np.load(path)
        return torch.tensor(data['voxel_coords'], dtype=torch.long)

    def _load_gaussian(self, index_id):
        path = os.path.join(self.data_dir, "gaussian", f"gaussian_output_{index_id}.npz")
        data = np.load(path)
        return {
            'position': torch.tensor(data['position'], dtype=torch.float32),
            'scale': torch.tensor(data['scale'], dtype=torch.float32),
            'rotation': torch.tensor(data['rotation'], dtype=torch.float32),
            'opacity': torch.tensor(data['opacity'], dtype=torch.float32),
        }

    def _load_voxel_label(self, sample_token: str):
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']

        # --- LiDAR 포인트 클라우드 파일 경로 수정 ---
        # nusc.get('sample_data', lidar_token)으로 해당 LiDAR 데이터 엔트리를 가져옵니다.
        lidar_data_entry = self.nusc.get('sample_data', lidar_token)
        # lidar_data_entry['filename']은 nusc.dataroot에 대한 상대 경로를 제공합니다.
        # 따라서 nusc.dataroot와 os.path.join하여 절대 경로를 만듭니다.
        points_file_path = os.path.join(self.nusc.dataroot, lidar_data_entry['filename'])
        # --- 수정 끝 ---

        # LiDAR Segmentation 라벨 파일 경로 (이것은 사용자가 별도로 관리하는 경로일 수 있음)
        # 'D:/nuscene_label/lidarseg/v1.0-trainval' 이 경로가 정확하다는 가정 하에 사용.
        # 실제 파일이 '0000fa1a7bfb46dc872045096181303e_lidarseg.bin' 이런 식이라면 lidar_token 뒤에 _lidarseg.bin 붙여야 합니다.
        lidarseg_root = os.path.join("D:/nuscene_label/lidarseg", "v1.0-trainval")
        label_file_path = os.path.join(lidarseg_root, f"{lidar_token}_lidarseg.bin") # 예시 토큰을 사용하지 않고 실제 lidar_token 사용

        #print(f"[DEBUG] LiDAR label path: {label_file_path}")
        #print(f"[DEBUG] LiDAR points path: {points_file_path}")

        if not os.path.exists(label_file_path):
            raise FileNotFoundError(f"No lidarseg label found at {label_file_path}")

        if not os.path.exists(points_file_path):
            raise FileNotFoundError(f"No lidar file found at {points_file_path}") # 이제 이 에러가 발생하면 경로가 진짜 틀린겁니다.

        points = np.fromfile(points_file_path, dtype=np.float32).reshape(-1, 5)[:, :3]
        labels = np.fromfile(label_file_path, dtype=np.uint8)
        assert len(points) == len(labels), f"Points and labels count mismatch for {lidar_token}"

        voxel_size = np.array([1.0, 1.0, 1.0])
        pc_range = np.array([-50, -50, -3])
        grid_size = np.array([100, 100, 6])

        voxel_coords = ((points - pc_range) / voxel_size).astype(np.int32)
        valid_mask = (
            (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < grid_size[0]) &
            (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < grid_size[1]) &
            (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < grid_size[2])
        )

        voxel_coords = voxel_coords[valid_mask]
        voxel_labels = labels[valid_mask]

        voxel_dict = {}
        for coord, label in zip(voxel_coords, voxel_labels):
            key = tuple(coord)
            voxel_dict.setdefault(key, []).append(label)

        label_grid = np.full(grid_size, 255, dtype=np.uint8)
        for (x, y, z), lbls in voxel_dict.items():
            label_grid[x, y, z] = Counter(lbls).most_common(1)[0][0]

        return torch.tensor(label_grid, dtype=torch.long)










