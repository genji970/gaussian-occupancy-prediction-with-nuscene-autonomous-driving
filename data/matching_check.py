import os
from nuscenes.nuscenes import NuScenes
from data_func import load_lidar_bin, get_lidarseg_labels, visualize_lidar_with_labels

# 초기화
nusc = NuScenes(version='v1.0-trainval', dataroot='D:/nuscene_data', verbose=True)

# 통계
checked = 0
matched = 0
skipped = 0

for i, sample in enumerate(nusc.sample):
    lidar_top_token = sample['data']['LIDAR_TOP']
    sd = nusc.get('sample_data', lidar_top_token)

    pcd_path = os.path.join(nusc.dataroot, sd['filename'])
    label_path = f'D:/nuscene_label/lidarseg/v1.0-trainval/{lidar_top_token}_lidarseg.bin'

    if not os.path.exists(pcd_path):
        print(f"[❌ 누락] LIDAR 파일 없음: {pcd_path}")
        skipped += 1
        continue

    if not os.path.exists(label_path):
        print(f"[❌ 누락] 라벨 파일 없음: {label_path}")
        skipped += 1
        continue

    try:
        points = load_lidar_bin(pcd_path)
        labels = get_lidarseg_labels(nusc, lidar_top_token)

        if labels is None or len(points) != len(labels):
            print(f"[⚠️ 불일치] 포인트/라벨 수 다름: {len(points)} vs {len(labels)}")
            skipped += 1
            continue

        # 클래스 ID 확인 (예: 0~31 범위)
        unique_classes = set(labels.tolist())
        if max(unique_classes) > 31:
            print(f"[⚠️ 경고] 라벨 ID 이상치 포함: {sorted(unique_classes)}")

        print(f"[✅ 매칭] 샘플 {i}: {len(points)} pts, {len(unique_classes)} 클래스")

        matched += 1
        # 필요하면 1개만 시각화
        if matched == 1:
            visualize_lidar_with_labels(points, labels)

    except Exception as e:
        print(f"[❌ 오류] 샘플 {i} 처리 중 에러 발생: {e}")
        skipped += 1

    checked += 1

print(f"\n총 {checked}개 중 {matched}개 성공, {skipped}개 실패 또는 누락")
