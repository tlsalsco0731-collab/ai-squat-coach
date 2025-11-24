from ultralytics import YOLO #Importing YOLO classes from the Ultralytics library
from squat_score import score_squat #Import the score_squat function within the squat_score.py file that I created separately

import os #Use os.madeirs to automatically create output folders
import cv2 #Draw score text over image with cv2.putText.& Save final image to file with cv2.imwrite
import numpy as np #Receive joint coordinates in a lumpy array inside score_squat and use them for angle calculation


def main():
    
    image_path = "/home/work/Minchaeggeo/Assignment_2/examples/squat.jpg" #Path with squat photo & use this image as input for YOLO model
    out_dir = "/home/work/Minchaeggeo/Assignment_2/outputs" #Folder path to store result image & later squat_scored.jpg is created within this folder

    # 1. Model load
    model = YOLO("yolov8n-pose.pt") #Role of a variable that performs reasoning in a model (image_path, save=False)

    # 2. Generate the output folder
    os.makedirs(out_dir, exist_ok=True) #even if the folder already exists, you can skip without error

    # 3. Pose inference
    results = model(image_path, save=False)  # save=False: Save directly
    if len(results) == 0:
        print("No result from model.")
        return

    r = results[0] #One Results object per image -> the first of which is pulled out

    if r.keypoints is None or r.keypoints.xy is None: #Objects with everyone's pokeypoints information detected in this image
        print("No keypoints detected.")
        return

    kpts_all = r.keypoints.xy #Each person has joint points (x, y) in 2D coordinates
    if kpts_all is None or len(kpts_all) == 0: #If the model does not fill the keypoints information, output "No keypoints detected" and exit
        print("No person detected.")
        return

    # Only use the first person
    kpts = kpts_all[0].cpu().numpy()  # (num_kpts, 2) 
    #The Ultralytics internal sensor may be on the GPU, so transfer it to CPU memory at .cpu(), and convert it to NumPy array at .numpy(). This array is used for angle calculation in the score_squat function

    # 4. Calculate score of the squat pose
    score, details = score_squat(kpts)
    print(f"Score: {score:.1f}")
    print("Details:", details)

    # 5. Add score text to the image with the pose and save it
    annotated = r.plot()  # numpy array (BGR)

    text = f"Score: {score:.1f}" #Insert score text into image
    cv2.putText( #Draw score text over image
        annotated,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        lineType=cv2.LINE_AA,
    )

    out_path = os.path.join(out_dir, "squat_scored.jpg")#outputs/squat_scored.jpg creates paths such as
    cv2.imwrite(out_path, annotated) #Save the annotated image as a real file
    print(f"Saved result image to: {out_path}") 


if __name__ == "__main__":
    main()
#An idiom that calls main() only when this file is executed directly, and prevents it from running if you import pose_squat_demo from another file.