# src/squat_score.py
import numpy as np # Numfi for numerical calculations (vectors, gambling, internal, etc.)
from typing import Dict, Tuple # Type Hints to specify the type of function return


# Calculate the angle (degree, degree, degree) at B when there are 3 points of A-B-C
def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    a, b, c: shape (2,) or (3,) coordinate (x, y [, z])
    Score of Return: degree
    """
    ba = a - b # Vector BA = A - B
    bc = c - b # Vector BC = C - B

    # Only use the 2D (x, y)
    ba = ba[:2]
    bc = bc[:2]

    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm == 0 or bc_norm == 0: # Even one vector cannot define the angle if the length is 0 → 0 treatment
        return 0.0

    cos_theta = np.dot(ba, bc) / (ba_norm * bc_norm) # Cosine law: cosθ = (BA·BC) / (|BA||BC|)
    cos_theta = np.clip(cos_theta, -1.0, 1.0) # Prevent numerical errors from going beyond range
    angle = np.degrees(np.arccos(cos_theta)) # Prevent numerical error from going beyond range# transform radian angle to arcos → degrees in degrees
    return float(angle)


def score_squat(keypoints: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    keypoints: shape (num_kpts, 2)  (x, y) coordinates (YOLOv8 pose results)
    COCO index:
      5: left shoulder
      11: left hip
      13: left knee
      15: left ankle

    Return:
      Score(0~100), Information dict
    """
    # Required joint index
    LEFT_SHOULDER = 5
    LEFT_HIP = 11
    LEFT_KNEE = 13
    LEFT_ANKLE = 15

    num_kpts = keypoints.shape[0] # Total number of keypoints
    needed = [LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE] # List of required indexes
    if any(i >= num_kpts for i in needed): 
        # If do not have a key point, get 0 points
        return 0.0, {"error": 1.0}

# Extract each joint coordinate (x, y)
    shoulder = keypoints[LEFT_SHOULDER] 
    hip = keypoints[LEFT_HIP] 
    knee = keypoints[LEFT_KNEE]
    ankle = keypoints[LEFT_ANKLE]

    # Angle of knee: hip - knee - ankle
    knee_angle = compute_angle(hip, knee, ankle)

    # Hip (body) angle: shoulder - hip - knee
    hip_angle = compute_angle(shoulder, hip, knee)

    score = 100.0 # How to deduct points from the starting score of 100

    # 1) Squat depth: 90 degrees knee angle ≈ is ideal (between 70 and 110 degrees)
    if knee_angle < 60 or knee_angle > 130:
        # Too shallow or too bent
        score -= 40
    else:
        score -= abs(knee_angle - 90) * 0.5  # 최대 10점 정도 감점

    # 2) Upper body tilt: The larger the hip_angle, the more upright the upper body is
    #   Ideally, look at 140-180 degrees. If it folds too much, it is deducted
    if hip_angle < 100:
        score -= 30  # the upper body is bent forward too much
    elif hip_angle < 140:
        score -= (140 - hip_angle) * 0.5  # If bend it a little bit, deduct points

    # 3) Scoring range clipping (limited to between 0 and 100)
    score = max(0.0, min(100.0, score))

# Details for debugging and explanation (record angle values)
    details = {
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
    }
    return score, details # Return final score and detailed angle information together
