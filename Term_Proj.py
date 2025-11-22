
from ultralytics import YOLO

def main():
    # 1. 포즈 모델 불러오기 (자동으로 yolov8n-pose.pt 다운로드)
    model = YOLO("yolov8n-pose.pt")

    # 2. 추론 돌릴 이미지 경로
    image_path = "/home/work/Minchaeggeo/Assignment_2/pose.jpg"  # 여기에 네 이미지 파일 이름 맞춰주면 됨


    # 3. 추론 실행 + 결과 이미지 저장
    results = model(
        source=image_path,
        save=True,               # 결과 이미지 저장
        project="outputs",       # 결과 폴더 상위
        name="pose_run",         # 그 아래 하위 폴더 이름
        exist_ok=True,           # 폴더 이미 있어도 덮어쓰기 허용
    )

    print("✅ 추론 완료! 결과 이미지를 outputs/pose_run/ 폴더에서 확인해봐.")

if __name__ == "__main__":
    main()
