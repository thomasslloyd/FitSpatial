import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.signal import butter, filtfilt


# source FSenv/bin/activate

# 1. Setup Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_heavy.task"),
    running_mode=VisionRunningMode.VIDEO,
)


def apply_butterworth_filter(data, fps, cutoff=5.0, order=4):
    nyquist = 0.5 * fps
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered_data[:, i] = filtfilt(b, a, data[:, i])
    return filtered_data


def process_video_and_plot(video_path):
    print(f"Loading video: {video_path}...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        sys.exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0

    # Dynamic trace arrays
    left_knee_path = []
    right_knee_path = []

    # Static context arrays (to be averaged)
    context_nodes = {
        "nose": [],
        "l_shoulder": [],
        "r_shoulder": [],
        "l_hip": [],
        "r_hip": [],
        "l_ankle": [],
        "r_ankle": [],
    }

    frame_index = 0
    print("Extracting spatial kinematics and postural context...")

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            timestamp_ms = int((frame_index / fps) * 1000)

            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.pose_world_landmarks:
                lm = results.pose_world_landmarks[0]

                # Extract dynamic knees
                left_knee_path.append([lm[25].x, lm[25].y, lm[25].z])
                right_knee_path.append([lm[26].x, lm[26].y, lm[26].z])

                # Extract context nodes
                context_nodes["nose"].append([lm[0].x, lm[0].y, lm[0].z])
                context_nodes["l_shoulder"].append([lm[11].x, lm[11].y, lm[11].z])
                context_nodes["r_shoulder"].append([lm[12].x, lm[12].y, lm[12].z])
                context_nodes["l_hip"].append([lm[23].x, lm[23].y, lm[23].z])
                context_nodes["r_hip"].append([lm[24].x, lm[24].y, lm[24].z])
                context_nodes["l_ankle"].append([lm[27].x, lm[27].y, lm[27].z])
                context_nodes["r_ankle"].append([lm[28].x, lm[28].y, lm[28].z])

            frame_index += 1

    cap.release()
    print(f"Processing complete. {frame_index} frames analyzed.")

    lk_data = np.array(left_knee_path)
    rk_data = np.array(right_knee_path)

    if len(lk_data) == 0:
        print("No pose data detected.")
        sys.exit()

    # Filter the dynamic knee data
    lk_filt = apply_butterworth_filter(lk_data, fps, cutoff=5.0)
    rk_filt = apply_butterworth_filter(rk_data, fps, cutoff=5.0)

    # Calculate the mean position for the context nodes
    mean_pose = {
        name: np.mean(np.array(data), axis=0) for name, data in context_nodes.items()
    }

    # 3. Build the Professional 3D Plot
    print("Generating Telemetry Plot...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Helper function to draw the ghost bones (Note: Y-axis is inverted with -p[1])
    def draw_bone(p1, p2):
        ax.plot(
            [p1[0], p2[0]],
            [p1[2], p2[2]],
            [-p1[1], -p2[1]],
            color="gray",
            linestyle="--",
            alpha=0.4,
        )

    # Draw Ghost Skeleton
    draw_bone(mean_pose["l_shoulder"], mean_pose["r_shoulder"])  # Shoulders
    draw_bone(mean_pose["l_shoulder"], mean_pose["l_hip"])  # Left Torso
    draw_bone(mean_pose["r_shoulder"], mean_pose["r_hip"])  # Right Torso
    draw_bone(mean_pose["l_hip"], mean_pose["r_hip"])  # Pelvis
    draw_bone(mean_pose["l_hip"], mean_pose["l_ankle"])  # Left Leg axis
    draw_bone(mean_pose["r_hip"], mean_pose["r_ankle"])  # Right Leg axis

    # Plot Ghost Nodes
    for name, pt in mean_pose.items():
        ax.scatter(pt[0], pt[2], -pt[1], color="gray", s=30, alpha=0.5)

    # Plot Dynamic Filtered Knee Traces
    ax.plot(
        lk_filt[:, 0],
        lk_filt[:, 2],
        -lk_filt[:, 1],
        label="Left Knee Trace",
        color="blue",
        linewidth=2.5,
    )
    ax.plot(
        rk_filt[:, 0],
        rk_filt[:, 2],
        -rk_filt[:, 1],
        label="Right Knee Trace",
        color="red",
        linewidth=2.5,
    )

    ax.set_xlabel("Lateral Sway (X)")
    ax.set_ylabel("Depth (Z)")
    ax.set_zlabel("Vertical Stroke (Y)")
    ax.set_title("Spatial-Dynamic Knee Telemetry with Mean Postural Reference")
    ax.legend()

    plt.show()


# --- EXECUTION ---
VIDEO_FILE = "test_video.mov"
process_video_and_plot(VIDEO_FILE)
