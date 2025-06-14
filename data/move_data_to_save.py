import os
import shutil

BASE_DIR = 'D:/nuscene_data'
BLOB_DIRS = ['v1.0-trainval04_blobs']
SUBFOLDERS = ['samples', 'sweeps']

for blob in BLOB_DIRS:
    for subfolder in SUBFOLDERS:
        src_root = os.path.join(BASE_DIR, blob, subfolder)
        dst_root = os.path.join(BASE_DIR, subfolder)

        if not os.path.exists(src_root):
            continue

        for sensor_folder in os.listdir(src_root):
            src_sensor_path = os.path.join(src_root, sensor_folder)
            dst_sensor_path = os.path.join(dst_root, sensor_folder)

            if not os.path.isdir(src_sensor_path):
                continue

            os.makedirs(dst_sensor_path, exist_ok=True)

            for filename in os.listdir(src_sensor_path):
                src_file = os.path.join(src_sensor_path, filename)
                dst_file = os.path.join(dst_sensor_path, filename)

                if not os.path.exists(dst_file):  # 중복 방지
                    shutil.move(src_file, dst_file)
