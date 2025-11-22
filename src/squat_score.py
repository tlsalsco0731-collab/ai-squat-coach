# src/squat_score.py
import numpy as np
from typing import Dict, Tuple


# A-B-C 3점이 있을 때, B에서의 각도(도, degree)를 계산
def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    a, b, c: shape (2,) or (3,) 좌표 (x, y [, z])
    반환값: 각도 (degree)
    """
    ba = a - b
    bc = c - b

    # 2D만 사용 (x, y)
    ba = ba[:2]
    bc = bc[:2]

    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm == 0 or bc_norm == 0:
        return 0.0

    cos_theta = np.dot(ba, bc) / (ba_norm * bc_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    return float(angle)


def score_squat(keypoints: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    keypoints: shape (num_kpts, 2) 의 (x, y) 좌표 (YOLOv8 pose 결과)
    COCO 기준 인덱스:
      5: left shoulder
      11: left hip
      13: left knee
      15: left ankle

    반환:
      점수(0~100), 상세 정보 dict
    """
    # 필요한 관절 인덱스
    LEFT_SHOULDER = 5
    LEFT_HIP = 11
    LEFT_KNEE = 13
    LEFT_ANKLE = 15

    num_kpts = keypoints.shape[0]
    needed = [LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]
    if any(i >= num_kpts for i in needed):
        # 키포인트가 없으면 0점 처리
        return 0.0, {"error": 1.0}

    shoulder = keypoints[LEFT_SHOULDER]
    hip = keypoints[LEFT_HIP]
    knee = keypoints[LEFT_KNEE]
    ankle = keypoints[LEFT_ANKLE]

    # 무릎 각도: hip - knee - ankle
    knee_angle = compute_angle(hip, knee, ankle)

    # 고관절(몸통) 각도: shoulder - hip - knee
    hip_angle = compute_angle(shoulder, hip, knee)

    # -------------------------------
    # 점수 로직 (간단 버전, 과제용)
    # -------------------------------
    score = 100.0

    # 1) 스쿼트 깊이: 무릎 각도 ≈ 90도가 이상적 (70~110도 사이면 OK)
    if knee_angle < 60 or knee_angle > 130:
        # 너무 얕거나, 너무 구부러진 경우
        score -= 40
    else:
        score -= abs(knee_angle - 90) * 0.5  # 최대 10점 정도 감점

    # 2) 상체 기울기: hip_angle이 클수록 상체가 곧게 서있는 편
    #   140~180도 정도를 이상적으로 보고, 너무 접히면 감점
    if hip_angle < 100:
        score -= 30  # 상체가 너무 앞으로 굽음
    elif hip_angle < 140:
        score -= (140 - hip_angle) * 0.5  # 살짝 굽은 정도는 약간만 감점

    # 3) 클리핑
    score = max(0.0, min(100.0, score))

    details = {
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
    }
    return score, details
