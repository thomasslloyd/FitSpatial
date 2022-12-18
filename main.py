import cv2
import numpy as np
import openpose as op

# Load the OpenPose model
model = op.get_model("pose", "BODY_25", "float", "COCO")
params = dict()
params["model_folder"] = "models/"

# Read the video file
video = cv2.VideoCapture("cycling.mp4")

# Loop through each frame of the video
while video.isOpened():
    # Read the frame
    success, frame = video.read()
    if not success:
        break

    # Run the frame through the OpenPose model
    keypoints, output_image = op.forward(frame, model, params)

    # Extract the keypoints for the cyclist
    cyclist_keypoints = keypoints[0]

    # Loop through each keypoint and draw a circle on the frame
    for keypoint in cyclist_keypoints:
        cv2.circle(output_image, (int(keypoint[0]), int(keypoint[1])), 5, (255, 0, 0), -1)

    # Display the frame with keypoints
    cv2.imshow("Cycling Video", output_image)

    # Wait for a key press
    key = cv2.waitKey(1)
    if key == 27:  # Press 'ESC' to exit
        break

# Release the video file and destroy the window
video.release()
cv2.destroyAllWindows()
