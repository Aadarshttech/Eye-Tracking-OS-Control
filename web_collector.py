import argparse
import base64
import csv
import math
import os
import time
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

LANDMARKER_PATH = "face_landmarker.task"
DATA_DIR = "dataset"

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

landmarker = None
session_data = []
session_user = ""
session_name = ""
is_recording = False

def setup_landmarker():
    base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE, 
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)

def extract_head_pose(transformation_matrix):
    r_mat = transformation_matrix[:3, :3]
    euler_angles = cv2.RQDecomp3x3(r_mat)[0]
    return euler_angles

def get_normalized_center(landmarks, indices):
    x = sum([landmarks[i].x for i in indices]) / len(indices)
    y = sum([landmarks[i].y for i in indices]) / len(indices)
    z = sum([landmarks[i].z for i in indices]) / len(indices)
    return x, y, z

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_session')
def handle_start_session(data):
    global session_data, session_user, session_name, is_recording
    session_user = data.get('user', 'unknown')
    session_name = data.get('session_name', 'session')
    session_data = []
    is_recording = True
    print(f"Started memory buffering for session: {session_name} by {session_user}")
    emit('session_started', {'status': 'success'})

@socketio.on('discard_session')
def handle_discard_session():
    global session_data, is_recording
    session_data = []
    is_recording = False
    print("Session discarded. Memory cleared.")
    emit('session_discarded', {'status': 'success'})

@socketio.on('save_session')
def handle_save_session():
    global session_data, session_user, session_name, is_recording
    is_recording = False
    if not session_data:
        print("No data to save.")
        emit('session_saved', {'status': 'empty'})
        return
        
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"gaze_data_{session_user}_{session_name}_{int(time.time())}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    
    csv_header = [
        "timestamp", "user", "lighting", "screen_w", "screen_h", "cam_w", "cam_h",
        "target_x", "target_y", 
        "head_pitch", "head_yaw", "head_roll",
        "l_iris_x", "l_iris_y", "l_iris_z", 
        "r_iris_x", "r_iris_y", "r_iris_z",
        "inter_ocular_dist"
    ]
    
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(session_data)
        
    session_data = [] # clear memory
    print(f"Session saved successfully to {filepath}")
    emit('session_saved', {'status': 'success', 'file': filepath})

@socketio.on('process_frame')
def handle_process_frame(data):
    global landmarker, session_data, is_recording
    
    if not is_recording:
        return 
        
    image_data = data['image'] 
    target_x = data['target_x']
    target_y = data['target_y']
    screen_w = data['screen_w']
    screen_h = data['screen_h']
    cam_w = data['cam_w']
    cam_h = data['cam_h']
    user = data.get('user', 'unknown')
    lighting = data.get('lighting', 'normal')
    
    # Decode base64 JPEG from web
    try:
        header, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Error decoding image:", e)
        return
    
    # MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    result = landmarker.detect(mp_image)
    
    success = False
    if result.face_landmarks and result.facial_transformation_matrixes:
        landmarks = result.face_landmarks[0]
        matrix = result.facial_transformation_matrixes[0]
        pitch, yaw, roll = extract_head_pose(matrix)
        lx, ly, lz = get_normalized_center(landmarks, LEFT_IRIS)
        rx, ry, rz = get_normalized_center(landmarks, RIGHT_IRIS)
        iod = math.sqrt((lx - rx)**2 + (ly - ry)**2 + (lz - rz)**2)
        
        timestamp_ms = int(time.time() * 1000)
        session_data.append([
            timestamp_ms, user, lighting, screen_w, screen_h, cam_w, cam_h,
            target_x, target_y,
            pitch, yaw, roll,
            lx, ly, lz,
            rx, ry, rz,
            iod
        ])
        success = True
        
    emit('frame_result', {'success': success})

if __name__ == '__main__':
    if not os.path.exists(LANDMARKER_PATH):
        print(f"Error: {LANDMARKER_PATH} not found!")
    else:
        landmarker = setup_landmarker()
        print("Starting Flask Web UI Server on http://localhost:5000")
        socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
