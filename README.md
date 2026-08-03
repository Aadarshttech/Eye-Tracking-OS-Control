
<div align="center">

# ??? AI-Powered Eye Tracking for OS Control

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-FF6F00?logo=tensorflow)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-00A67E?logo=google)](https://google.github.io/mediapipe/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-5C3EE8?logo=opencv)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Control your Windows operating system purely using your eyes.**<br>
*A complete end-to-end deep learning pipeline built for motor accessibility.*

</div>

<hr>

## ?? Overview

Historically, eye-tracking has required expensive, specialized infrared hardware. This project democratizes accessibility by providing a robust, hands-free mouse controller utilizing only a standard consumer webcam (720p/1080p).

By fusing **Google's real-time MediaPipe Face Mesh**, a heavily optimized **TensorFlow Multi-Layer Perceptron (MLP)**, and advanced mathematical filtering (the **One Euro Filter**), this software maps 3D facial geometry directly to 2D screen coordinates with an average Euclidean error of just **27.4 pixels**.

### ? Core Features
- **Look to Move:** Seamlessly drives the OS cursor by translating head pose and iris bounds.
- **Blink to Click:** Solves the "Midas Touch" problem using Eye Aspect Ratio (EAR) temporal blink detection for reliable left-clicking.
- **Anti-Jitter:** Incorporates adaptive low-pass filtering to maintain cursor stability during reading (fixation) without introducing input lag during quick glances.
- **Hardware Agnostic:** Fully operates on standard CPU architectures achieving $\sim inference latency (50+ theoretical FPS).

---

## ?? Getting Started: The Three-Phase Pipeline

We have designed this repository to be strictly linear. Follow the three phases to collect your own data, train your personalized AI, and take control of your machine.

### Phase 1: Data Collection
Before the neural network can map your gaze, it needs to learn your specific eye geometry. We provide a Flask-based web collector for rapid, safe dataset generation.

1. Launch the web collector:
   \\\ash
   python src/web_collector.py
   \\\
2. Navigate to http://localhost:5000 in your browser.
3. Follow the on-screen red dots. The software automatically captures your 478-point MediaPipe mesh and logs your screen coordinates into a CSV inside the dataset/ folder. *(Aim for 3-4 sessions for high accuracy).*

### Phase 2: Model Training
Once the training data is collected, it's time to train the MLP Large v2 model. Our architecture employs **Cosine-Decaying Learning Rates**, **Huber Loss**, and **Synthetic Gaussian Noise** data augmentation to prevent overfitting.

1. Run the training script:
   \\\ash
   python src/train_gaze_model_v3.py
   \\\
2. The pipeline loads all CSVs, triples the dataset size via augmentation, and exports a lightweight eye_model.tflite model.

> ?? *For a deep dive into hyperparameter tuning and model architectures, please see our [Comprehensive Training Guide](docs/guide_for_training.md).*

### Phase 3: AI Mouse Control
With your trained model exported, you are ready to control your operating system.

1. Launch the actuation software:
   \\\ash
   python src/gaze_mouse.py
   \\\
2. Complete the rapid **9-point per-session calibration**. This executes a Polynomial Ridge Regression that dynamically maps the AI's raw output to your specific monitor boundaries, completely eliminating edge-bias.
3. You now have full control of your mouse!

---

## ??? Setup & Installation

### Requirements
- **OS:** Windows 10/11 (Validated)
- **Environment:** Python 3.11+
- **Hardware:** Standard Web Camera (720p minimum)

### Installation
Clone the repository and install the strict dependencies:
\\\ash
git clone https://github.com/Aadarshttech/Eye-Tracking-OS-Control.git
cd Eye-Tracking-OS-Control
pip install -r requirements.txt
\\\
*(Optional: Run 
pm install if you wish to use localtunnel for remote data collection as defined in package.json.)*

---

## ?? System Architecture

The pipeline executes entirely within 19 milliseconds per frame:
1. **Frontend/Capture:** OpenCV & MediaPipe isolate the face and draw a 478-point 3D mesh.
2. **Feature Engineering:** NumPy calculates Head Pitch, Yaw, Roll, and localized Iris offsets.
3. **Backend/AI:** A custom TensorFlow Keras MLP executes matrix multiplication to predict (X, Y) coordinates.
4. **Filtering:** The One Euro Filter adapts to velocity, destroying high-frequency webcam jitter.
5. **Actuation:** PyAutoGUI executes physical cursor movement and triggers clicks upon detecting EAR threshold drops.

<div align="center">
  <img src="docs/images/target_dist.png" width="45%" />
  <img src="docs/images/ear_dist.png" width="45%" />
  <br>
  <em>(Left: Gaze distribution during training. Right: EAR Blink Separation Thresholds)</em>
</div>

---

## ????? Project Team & Academic Origin
This repository represents the final semester defense project for the B.Tech AI (Batch 2024) program at Kathmandu University.
- **Aadarsha Pandit** (21747)
- **Yudhin Khanal** (21740)
- **Ishan Pandey** (21746)
- **Kushal Kunwar** (21742)

*For our full technical thesis and defense documentation, see the [Final Defense Report PDF](docs/Final_Defense_Report.pdf).*

