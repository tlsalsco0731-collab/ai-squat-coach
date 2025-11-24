# AI model to calculate the squat pose

This project uses a pretrained YOLOv8 Pose*model from Ultralytics to
estimate human pose and score squat posture based on joint angles.

---

1. Project Idea

Squat is a fundamental lower-body exercise, but many beginners struggle with
incorrect posture (knee position, torso leaning, depth, etc.).

This project aims to:
 Detect human body keypoints from a single squat image.
 Compute joint angles (knee and hip).
 Assign a squat score (0–100) based on simple, interpretable rules.
 Visualize the result by overlaying skeleton and score text on the image.

---

 2. Model and Data

 Model: YOLOv8n-Pose (pretrained, from Ultralytics)
 Input: A single RGB image of a person performing a squat.
 Output: 2D keypoints of human joints (COCO-style, e.g., shoulder, hip, knee, ankle).

We do not train or fine-tune the model in this project.
Instead, we reuse pretrained weights and focus on:
 Understanding the model output (keypoints).
 Designing a rule-based scoring algorithm on top of the model.

---

 3. Environment

 Python 3.12
 PyTorch 2.9.1
 torchvision 0.24.1
 ultralytics 8.3.229
 opencv-python 4.12.0.88
 numpy 2.2.6

Install dependencies:

```bash
pip install -r requirements.txt

