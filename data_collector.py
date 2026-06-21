import argparse
import csv
import math
import os
import random
import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Paths
LANDMARKER_PATH = "face_landmarker.task"
DATA_DIR = "dataset"

# Eye / Iris landmark indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

def setup_landmarker():
    base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True, # To get head pose
        num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)

def extract_head_pose(transformation_matrix):
    """Extract pitch, yaw, roll from the 4x4 transformation matrix."""
    # The upper 3x3 is the rotation matrix
    r_mat = transformation_matrix[:3, :3]
    # Decompose to Euler angles (in degrees)
    euler_angles = cv2.RQDecomp3x3(r_mat)[0]
    pitch, yaw, roll = euler_angles
    return pitch, yaw, roll

def get_center(landmarks, indices, img_w, img_h):
    """Get the 2D average pixel coordinates of a set of landmarks."""
    x = sum([landmarks[i].x for i in indices]) / len(indices)
    y = sum([landmarks[i].y for i in indices]) / len(indices)
    return x * img_w, y * img_h

def get_normalized_center(landmarks, indices):
    """Get the 3D average normalized coordinates."""
    x = sum([landmarks[i].x for i in indices]) / len(indices)
    y = sum([landmarks[i].y for i in indices]) / len(indices)
    z = sum([landmarks[i].z for i in indices]) / len(indices)
    return x, y, z

def main(args):
    if not os.path.exists(LANDMARKER_PATH):
        print(f"Error: {LANDMARKER_PATH} not found in current directory.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    csv_file = os.path.join(DATA_DIR, f"gaze_data_{args.user}_{int(time.time())}.csv")

    landmarker = setup_landmarker()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    # Get webcam resolution
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam initialized at {cam_w}x{cam_h}")

    # Setup Fullscreen Window
    window_name = "Gaze Data Collector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # We need to know the screen resolution to accurately log target coordinates.
    # We will show a quick screen to prompt the user to hit space to start, and we 
    # can measure the window size.
    cv2.imshow(window_name, np.zeros((400, 400, 3), dtype=np.uint8))
    cv2.waitKey(100) # Give it a moment to enter fullscreen
    _, _, screen_w, screen_h = cv2.getWindowImageRect(window_name)
    print(f"Detected screen resolution: {screen_w}x{screen_h}")

    csv_header = [
        "timestamp", "user", "lighting", "screen_w", "screen_h", "cam_w", "cam_h",
        "target_x", "target_y", 
        "head_pitch", "head_yaw", "head_roll",
        "l_iris_x", "l_iris_y", "l_iris_z", 
        "r_iris_x", "r_iris_y", "r_iris_z",
        "inter_ocular_dist"
    ]

    print("Starting data collection...")
    print(f"User: {args.user}, Lighting: {args.lighting}")
    print(f"Targets: {args.targets}, Frames per target: {args.frames_per_target}")
    
    start_time = time.perf_counter()

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

        # Wait for user to be ready
        ready_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.putText(ready_frame, f"User: {args.user} | Press SPACE to begin.", 
                    (screen_w//2 - 200, screen_h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(window_name, ready_frame)
        
        while True:
            k = cv2.waitKey(1)
            if k == 32: # SPACE
                break
            if k == 27: # ESC
                cap.release()
                cv2.destroyAllWindows()
                return

        for target_idx in range(args.targets):
            # Generate random target point (padding to avoid edge clipping)
            pad = 40
            tx = random.randint(pad, screen_w - pad)
            ty = random.randint(pad, screen_h - pad)
            
            frames_captured = 0
            
            while frames_captured < args.frames_per_target:
                success, frame = cap.read()
                if not success:
                    print("Camera read failed!")
                    break
                
                frame = cv2.flip(frame, 1) # Mirror
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                timestamp_ms = int((time.perf_counter() - start_time) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                # Draw UI
                display_img = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                
                # Draw the target
                # Change color slightly to show it's recording
                color = (0, 0, 255) if frames_captured % 10 < 5 else (0, 100, 255) 
                cv2.circle(display_img, (tx, ty), 15, color, -1)
                cv2.circle(display_img, (tx, ty), 2, (255, 255, 255), -1)
                
                # Info text
                cv2.putText(display_img, f"Target {target_idx+1}/{args.targets}", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                if result.face_landmarks and result.facial_transformation_matrixes:
                    landmarks = result.face_landmarks[0]
                    matrix = result.facial_transformation_matrixes[0]
                    
                    pitch, yaw, roll = extract_head_pose(matrix)
                    
                    lx, ly, lz = get_normalized_center(landmarks, LEFT_IRIS)
                    rx, ry, rz = get_normalized_center(landmarks, RIGHT_IRIS)
                    
                    # Approximate inter-ocular distance in normalized 3D space
                    iod = math.sqrt((lx - rx)**2 + (ly - ry)**2 + (lz - rz)**2)
                    
                    # Log the data
                    writer.writerow([
                        timestamp_ms, args.user, args.lighting, screen_w, screen_h, cam_w, cam_h,
                        tx, ty,
                        pitch, yaw, roll,
                        lx, ly, lz,
                        rx, ry, rz,
                        iod
                    ])
                    frames_captured += 1
                    cv2.putText(display_img, "Tracking: OK", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                else:
                    cv2.putText(display_img, "Face Not Found!", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

                cv2.imshow(window_name, display_img)
                if cv2.waitKey(1) == 27: # ESC
                    print("Aborted by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()
    cv2.destroyAllWindows()
    print(f"Data collection complete! Saved to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect gaze data mapped to screen coordinates.")
    parser.add_argument("--user", type=str, required=True, help="Name of the participant (e.g., aadarsha, yudhin)")
    parser.add_argument("--lighting", type=str, default="normal", help="Lighting condition (e.g., normal, dim, bright, side-lit)")
    parser.add_argument("--targets", type=int, default=60, help="Number of random screen targets to display")
    parser.add_argument("--frames-per-target", type=int, default=20, help="Number of valid frames to record per target")
    
    args = parser.parse_args()
    main(args)
