import cv2
import time
import numpy as np
import tkinter as tk
import threading
import os
import tensorflow as tf
import mediapipe as mp

BASE_DIR = r"d:\Projects\Projects\eye tracks"

# ==============================================================================
# Model & Feature Extraction
# ==============================================================================
def load_tflite_model(model_path=os.path.join(BASE_DIR, "models", "eye_model.tflite")):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

def predict_tflite(interpreter, input_details, output_details, X):
    interpreter.set_tensor(input_details[0]['index'], X.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

import pickle
def load_scaler(scaler_path=os.path.join(BASE_DIR, "models", "feature_scaler.pkl")):
    with open(scaler_path, "rb") as f:
        return pickle.load(f)

# MediaPipe Initialization
options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=os.path.join(BASE_DIR, "models", "face_landmarker.task")),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    output_facial_transformation_matrixes=True
)
landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

# Landmark Indices
LEFT_IRIS, RIGHT_IRIS = [474, 475, 476, 477], [469, 470, 471, 472]
LEFT_EAR_IDX, RIGHT_EAR_IDX = (263, 387, 385, 362, 380, 373), (33, 160, 158, 133, 153, 144)
LEFT_EYE_CONTOUR = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
RIGHT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

def get_normalized_center(landmarks, indices):
    return (sum(landmarks[i].x for i in indices)/len(indices), sum(landmarks[i].y for i in indices)/len(indices), sum(landmarks[i].z for i in indices)/len(indices))

def compute_gaze_ratio(landmarks, iris_idx, eye_contour_idx):
    ix, iy, _ = get_normalized_center(landmarks, iris_idx)
    xs, ys = [landmarks[i].x for i in eye_contour_idx], [landmarks[i].y for i in eye_contour_idx]
    hx, hy = (ix - min(xs)) / max(max(xs) - min(xs), 1e-7), (iy - min(ys)) / max(max(ys) - min(ys), 1e-7)
    return float(np.clip(hx, 0, 1)), float(np.clip(hy, 0, 1))

def compute_ear(landmarks, indices):
    pts = [np.array([landmarks[i].x, landmarks[i].y]) for i in indices]
    return float((np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])) / (2.0 * np.linalg.norm(pts[0] - pts[3]) + 1e-7))

def compute_face_area(landmarks):
    pts = np.array([(landmarks[i].x, landmarks[i].y) for i in FACE_OVAL])
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

# ==============================================================================
# Premium Tkinter Visualizer HUD + Instant Cubic Edge Stretch (ZERO Calibration)
# ==============================================================================
class GazeVisualizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gaze AI - HUD Visualizer")
        self.root.attributes("-fullscreen", True)
        self.root.configure(background='#0B0F19') # Deep Navy background
        
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        
        self.canvas = tk.Canvas(self.root, width=self.screen_w, height=self.screen_h, bg='#0B0F19', highlightthickness=0)
        self.canvas.pack()
        
        # 1. Background Grid
        for i in range(0, self.screen_w, 60):
            self.canvas.create_line(i, 0, i, self.screen_h, fill="#1E293B", dash=(4, 4))
        for i in range(0, self.screen_h, 60):
            self.canvas.create_line(0, i, self.screen_w, i, fill="#1E293B", dash=(4, 4))
            
        # 3. HUD Info Panel
        self.hud_bg = self.canvas.create_rectangle(20, 20, 320, 220, outline="#38BDF8", width=2, fill="#0F172A", stipple="gray50")
        self.hud_title = self.canvas.create_text(170, 45, text="GAZE AI ONLINE", fill="#38BDF8", font=("Consolas", 18, "bold"))
        self.hud_line = self.canvas.create_line(40, 65, 300, 65, fill="#38BDF8")
        
        self.txt_status = self.canvas.create_text(40, 90, anchor="w", text="Status: SEARCHING...", fill="#94A3B8", font=("Consolas", 14))
        self.txt_coords = self.canvas.create_text(40, 120, anchor="w", text="Target: X: 0 | Y: 0", fill="#F8FAFC", font=("Consolas", 14))
        self.txt_pose = self.canvas.create_text(40, 150, anchor="w", text="Pose  : P: 0° | Y: 0°", fill="#F8FAFC", font=("Consolas", 14))
        self.txt_ear = self.canvas.create_text(40, 180, anchor="w", text="EAR   : 0.00", fill="#F8FAFC", font=("Consolas", 14))
        
        self.txt_exit = self.canvas.create_text(self.screen_w/2, self.screen_h - 40, text="PRESS [ ESC ] TO INITIATE SHUTDOWN", fill="#64748B", font=("Consolas", 14, "bold"))
        
        # 4. Crosshair Elements
        self.trail_positions = []
        self.trail_dots = []
        for _ in range(8):
            self.trail_dots.append(self.canvas.create_oval(-10,-10,-10,-10, fill="#0284C7", outline=""))
            
        self.crosshair_outer = self.canvas.create_oval(-10,-10,-10,-10, outline="#38BDF8", width=2)
        self.crosshair_inner = self.canvas.create_oval(-10,-10,-10,-10, fill="#38BDF8", outline="")
        self.crosshair_h = self.canvas.create_line(-10,-10,-10,-10, fill="#38BDF8", width=2)
        self.crosshair_v = self.canvas.create_line(-10,-10,-10,-10, fill="#38BDF8", width=2)
        
        self.root.bind("<Escape>", self.quit)
        
        self.running = True
        self.target_x, self.target_y = self.screen_w / 2, self.screen_h / 2
        self.current_x, self.current_y = self.screen_w / 2, self.screen_h / 2
        
        # Real-time state
        self.face_detected = False
        self.pitch, self.yaw, self.ear = 0, 0, 0
        self.raw_pred_x, self.raw_pred_y = 0.5, 0.5

        print("Loading Model...")
        self.interpreter, self.input_det, self.output_det = load_tflite_model()
        self.scaler = load_scaler()
        
        # Start Threads
        self.cam_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.cam_thread.start()
        
        self.update_ui()
        self.root.mainloop()

    def update_ui(self):
        # Heavy smoothing to eliminate dancing and jitter
        smoothing = 0.08
        self.current_x += (self.target_x - self.current_x) * smoothing
        self.current_y += (self.target_y - self.current_y) * smoothing
        
        self.trail_positions.append((self.current_x, self.current_y))
        if len(self.trail_positions) > 8:
            self.trail_positions.pop(0)
            
        for i, (tx, ty) in enumerate(self.trail_positions):
            r = (i + 1) * 1.2
            self.canvas.coords(self.trail_dots[i], tx - r, ty - r, tx + r, ty + r)
            
        cx, cy = self.current_x, self.current_y
        r_out, r_in = 25, 4
        self.canvas.coords(self.crosshair_outer, cx - r_out, cy - r_out, cx + r_out, cy + r_out)
        self.canvas.coords(self.crosshair_inner, cx - r_in, cy - r_in, cx + r_in, cy + r_in)
        self.canvas.coords(self.crosshair_h, cx - 35, cy, cx + 35, cy)
        self.canvas.coords(self.crosshair_v, cx, cy - 35, cx, cy + 35)
        
        if self.face_detected:
            self.canvas.itemconfig(self.txt_status, text="Status: TRACKING ACTIVE", fill="#34D399")
            self.canvas.itemconfig(self.txt_coords, text=f"Target: X: {int(cx):04d} | Y: {int(cy):04d}")
            self.canvas.itemconfig(self.txt_pose, text=f"Pose  : P: {self.pitch:02.0f}° | Y: {self.yaw:02.0f}°")
            self.canvas.itemconfig(self.txt_ear, text=f"EAR   : {self.ear:.2f}")
        else:
            self.canvas.itemconfig(self.txt_status, text="Status: FACE NOT FOUND", fill="#F87171")
            
        if self.running:
            self.root.after(16, self.update_ui)

    def camera_loop(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            success, frame = cap.read()
            if not success: continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            res = landmarker.detect_for_video(mp_image, int(time.time() * 1000))
            
            if not res.face_landmarks:
                self.face_detected = False
                continue
            
            self.face_detected = True
            lm, mat = res.face_landmarks[0], res.facial_transformation_matrixes[0]
            
            pitch, yaw, roll = 0, 0, 0
            if mat is not None:
                pose_mat = mat[:3, :3]
                sy = np.sqrt(pose_mat[0,0]**2 + pose_mat[1,0]**2)
                if sy > 1e-6:
                    x, y, z = np.arctan2(pose_mat[2,1], pose_mat[2,2]), np.arctan2(-pose_mat[2,0], sy), np.arctan2(pose_mat[1,0], pose_mat[0,0])
                else:
                    x, y, z = np.arctan2(-pose_mat[1,2], pose_mat[1,1]), np.arctan2(-pose_mat[2,0], sy), 0
                pitch, yaw, roll = np.degrees(x), np.degrees(y), np.degrees(z)

            lx, ly, lz = get_normalized_center(lm, LEFT_IRIS)
            rx, ry, rz = get_normalized_center(lm, RIGHT_IRIS)
            iod = np.sqrt((lx - rx)**2 + (ly - ry)**2 + (lz - rz)**2)
            lgh, lgv = compute_gaze_ratio(lm, LEFT_IRIS, LEFT_EYE_CONTOUR)
            rgh, rgv = compute_gaze_ratio(lm, RIGHT_IRIS, RIGHT_EYE_CONTOUR)
            l_ear, r_ear = compute_ear(lm, LEFT_EAR_IDX), compute_ear(lm, RIGHT_EAR_IDX)
            face_area = compute_face_area(lm)

            self.pitch, self.yaw, self.ear = pitch, yaw, (l_ear + r_ear)/2

            avg_ix, avg_iy, avg_iz = (lx+rx)/2, (ly+ry)/2, (lz+rz)/2
            avg_gh, avg_gv = (lgh+rgh)/2, (lgv+rgv)/2
            
            features = [
                pitch, yaw, roll, lx, ly, lz, rx, ry, rz, iod, lgh, lgv, rgh, rgv,
                l_ear, r_ear, face_area, avg_ix, avg_iy, avg_iz, lx-rx, ly-ry,
                avg_gh, avg_gv, lgh-rgh, lgv-rgv, (l_ear+r_ear)/2, l_ear-r_ear,
                yaw*avg_ix, pitch*avg_iy, yaw*avg_gh, pitch*avg_gv,
                avg_ix/max(iod, 1e-7), avg_iy/max(iod, 1e-7),
                avg_ix**2, avg_iy**2, yaw**2, pitch**2
            ]

            X_scaled = np.nan_to_num(self.scaler.transform(np.array(features).reshape(1, -1)), nan=0.0, posinf=0.0, neginf=0.0)
            pred = predict_tflite(self.interpreter, self.input_det, self.output_det, X_scaled)
            
            # ---------------------------------------------------------
            # PURE LINEAR MAPPING (No Distortion)
            # ---------------------------------------------------------
            raw_x, raw_y = pred[0], pred[1]
            
            # Apply a very gentle 1.15x linear stretch from the center.
            # This preserves the exact proportions of the neural network 
            # across the entire screen, but gives it just enough reach 
            # to hit the corners.
            scale = 1.15
            norm_x = (raw_x - 0.5) * scale + 0.5
            norm_y = (raw_y - 0.5) * scale + 0.5
            
            # Map to screen
            final_x = norm_x * self.screen_w
            final_y = norm_y * self.screen_h
            
            # Constraint
            self.target_x = max(0, min(self.screen_w, final_x))
            self.target_y = max(0, min(self.screen_h, final_y))
            
        cap.release()

    def quit(self, event=None):
        self.running = False
        self.root.destroy()
        os._exit(0)

if __name__ == "__main__":
    app = GazeVisualizer()
