'''

nuscenes/
├── samples/                # 이미지 등 원본 데이터
├── sweeps/                 # LiDAR sweep
├── maps/                   # 맵 데이터
├── v1.0-trainval/          # 메타정보 (JSON 파일들)
│   ├── sample.json
│   ├── sample_data.json
│   └── ...

'''

"""

Input: LIDAR 포인트 → Voxel grid

Label: Occupancy (해당 voxel에 점이 있는지 여부)


"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

###  LIDAR ###

# LIDAR_TOP input : x, y, z, intensity, ring_index

def load_lidar_bin(bin_path):
    """Load a .pcd.bin LIDAR file from NuScenes"""
    print(f"Loading lidar bin: {bin_path}")
    raw = np.fromfile(bin_path, dtype=np.float32)
    print(f"Loaded {raw.shape[0]} values")
    lidar = raw.reshape(-1, 5)
    print(f"Lidar shape: {lidar.shape}")
    return lidar[:, :4]  # (x, y, z, intensity)만 사용

def get_lidarseg_labels(nusc, sample_data_token):
    # NuScenes 라벨 경로 포맷: {token}_lidarseg.bin
    filename = f"{sample_data_token}_lidarseg.bin"
    lidarseg_path = os.path.join('D:/nuscene_label/lidarseg/v1.0-trainval', filename)

    if not os.path.exists(lidarseg_path):
        print(f"[라벨 없음] {lidarseg_path}")
        return None

    return np.fromfile(lidarseg_path, dtype=np.uint8)

def visualize_lidar_with_labels(points, labels=None):
    """
    points: (N, 4) numpy array with x, y, z, intensity
    labels: (N,) numpy array with class indices (optional)
    """
    x = points[:, 0]
    y = points[:, 1]

    plt.figure(figsize=(10, 10))
    if labels is not None:
        plt.scatter(x, y, c=labels, s=1, cmap='tab20', alpha=0.7)
        plt.title("LIDAR XY View (colored by labels)")
        plt.colorbar(label="Class Index")
    else:
        plt.scatter(x, y, s=1, c='gray', alpha=0.5)
        plt.title("LIDAR XY View (no labels)")

    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def load_all_lidar_samples(nusc, sample_num, label_root='D:/nuscene_label/lidarseg/v1.0-trainval'):
    lidar_data = []

    cnt = 0
    for sample in tqdm(nusc.sample):
        cnt += 1
        if cnt > sample_num:
            break
        lidar_top_token = sample['data']['LIDAR_TOP']
        sd = nusc.get('sample_data', lidar_top_token)
        pcd_path = os.path.join(nusc.dataroot, sd['filename'])

        # Load lidar points
        points = load_lidar_bin(pcd_path)

        # Load labels
        filename = f"{lidar_top_token}_lidarseg.bin"
        label_path = os.path.join(label_root, filename)
        if os.path.exists(label_path):
            labels = np.fromfile(label_path, dtype=np.uint8)
        else:
            labels = None

        lidar_data.append({
            'sample_token': sample['token'],
            'lidar_token': lidar_top_token,
            'lidar_points': points,   # (N, 4)
            'lidar_labels': labels    # (N,) or None
        })

    return lidar_data



###  camera ###

def load_multiview_inputs(nusc, sample_token, image_root, image_size=(1600, 900)):
    import torchvision.transforms as T

    cam_channels = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    sample = nusc.get('sample', sample_token)
    images, cam_Ks, cam_Ts = [], [], []

    transform = T.Compose([
        T.Resize((224, 224)),  # PIL: (H, W)
        T.ToTensor()
    ])

    for cam in cam_channels:
        cam_token = sample['data'][cam]
        cam_data = nusc.get('sample_data', cam_token)
        calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

        # Load image
        img_path = os.path.join(image_root, cam_data['filename'])
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)  # [3, H, W]
        images.append(img_tensor)

        # Intrinsic matrix
        K = np.array(calib['camera_intrinsic'])
        cam_Ks.append(torch.tensor(K, dtype=torch.float32))

        # Extrinsic (world → camera)
        rotation = np.array(calib['rotation'])
        translation = np.array(calib['translation'])

        R = quat2rotmat(rotation)  # (3, 3)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = translation
        cam_Ts.append(torch.tensor(T, dtype=torch.float32))

    images = torch.stack(images, dim=0)       # [6, 3, H, W]
    cam_Ks = torch.stack(cam_Ks, dim=0)       # [6, 3, 3]
    cam_Ts = torch.stack(cam_Ts, dim=0)       # [6, 4, 4]
    return images, cam_Ks, cam_Ts

def quat2rotmat(q):
    """Quaternion [w, x, y, z] → 회전행렬"""
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x**2+z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x**2+y**2)]
    ])

def load_multiview_samples(nusc, sample_num, image_root='D:/nuscene_data', image_size=(800, 450)):
    import torchvision.transforms as T

    camera_data = []

    cnt = 0
    for sample in tqdm(nusc.sample):
        cnt += 1
        if cnt > sample_num:
            break
        sample_token = sample['token']
        try:
            images, cam_Ks, cam_Ts = load_multiview_inputs(
                nusc, sample_token=sample_token,
                image_root=image_root, image_size=image_size
            )
        except Exception as e:
            print(f"[{sample_token}] 이미지 로딩 실패: {e}")
            continue

        camera_data.append({
            'sample_token': sample_token,
            'images': images,     # List of [3, H, W] tensors , image pixel value ~ (0,255), rgb인지 bgr인지 확인 필요
            'cam_Ks': cam_Ks,     # (6, 3, 3) camera's intrinsic parameter(3d -> 2d projection)
            'cam_Ts': cam_Ts      # (6, 4, 4) camera's extrinsic parameter
        })

    print(f"[DEBUG] Loaded {len(camera_data)} multiview samples")
    return camera_data

"""

cam_K = [
    [fx,  0, cx],     fx, fy: 초점 거리 (Focal length)
    [ 0, fy, cy],     cx, cy: 이미지 중심 (Principal point)
    [ 0,  0,  1]
],

cam_T = [               :  월드 좌표계 ↔ 카메라 좌표계 변환
    [R | t]             :  전체는 4×4 homogeneous transformation matrix   
    [0  0  0  1]
]

"""

# camera sampling해서 시각화
def camera_visualization(sample_token, images):
    """
    images: List of [3, H, W] float tensors (0~1)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Sample Token: {sample_token}")

    for i in range(6):
        ax = axes[i // 3][i % 3]
        img = images[i].permute(1, 2, 0).cpu().numpy()  # [H, W, 3], float
        img = (img * 255).astype('uint8')  # for visualization
        ax.imshow(img)
        ax.set_title(f"Camera {i}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

