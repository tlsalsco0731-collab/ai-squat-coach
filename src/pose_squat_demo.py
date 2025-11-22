from ultralytics import YOLO
from squat_score import score_squat

import os
import cv2
import numpy as np


def main():
    # 0. 하드코딩된 경로
    image_path = "/home/work/Minchaeggeo/Assignment_2/examples/pose.jpg"
    out_dir = "/home/work/Minchaeggeo/Assignment_2/outputs"

    # 1. 모델 로드
    model = YOLO("yolov8n-pose.pt")

    # 2. 출력 폴더 생성
    os.makedirs(out_dir, exist_ok=True)

    # 3. 포즈 추론
    results = model(image_path, save=False)  # save=False: 여기서 직접 저장
    if len(results) == 0:
        print("❌ No result from model.")
        return

    r = results[0]

    if r.keypoints is None or r.keypoints.xy is None:
        print("❌ No keypoints detected.")
        return

    kpts_all = r.keypoints.xy
    if kpts_all is None or len(kpts_all) == 0:
        print("❌ No person detected.")
        return

    # 첫 번째 사람만 사용
    kpts = kpts_all[0].cpu().numpy()  # (num_kpts, 2)

    # 4. 스쿼트 점수 계산
    score, details = score_squat(kpts)
    print(f"Score: {score:.1f}")
    print("Details:", details)

    # 5. 포즈가 그려진 이미지에 점수 텍스트 추가해서 저장
    annotated = r.plot()  # numpy array (BGR)

    text = f"Score: {score:.1f}"
    cv2.putText(
        annotated,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        lineType=cv2.LINE_AA,
    )

    out_path = os.path.join(out_dir, "squat_scored.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"✅ Saved result image to: {out_path}")


if __name__ == "__main__":
    main()