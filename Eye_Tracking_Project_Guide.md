# Comprehensive Guide & Roadmap: AI-Powered Eye Tracking for OS Control

## 1. Project Overview
**Objective:** Build an accurate and robust eye-tracking system that replaces a standard computer mouse. The system will map the user's gaze to the (X, Y) coordinates of the screen and use intentional blinks as mouse clicks.
**Target Status:** Semester Project (High complexity, AI-driven, Computer Vision + HCI).

---

## 2. The "Significant Contribution" (Novelty)
To make this stand out from basic tutorials, the project will implement:
1. **Hybrid Landmark-AI Pipeline:** Instead of heavy, slow Convolutional Neural Networks (CNNs) on raw images, we use Google MediaPipe for rapid 3D landmark extraction, feeding those numeric coordinates into a lightweight, custom-trained Neural Network.
2. **Intentional Blink Detection:** Using a time-series AI algorithm (like an SVM or lightweight RNN) to differentiate between a 150ms involuntary blink and a 400ms intentional "click" blink.
3. **Advanced Jitter Reduction:** Implementing an Exponential Moving Average (EMA) or One Euro Filter to humanize and smooth the cursor movement, preventing erratic screen jumping.

---

## 3. Step-by-Step Implementation Roadmap

### Phase 1: Core Detection Foundation
* **Goal:** Successfully track the face and eyes in real-time.
* **Tools:** Python, OpenCV, MediaPipe (`face_landmarker.task`).
* **Action:** Extract the 468 3D facial landmarks. Isolate the specific landmark indices for the left eye, right eye, and irises. 

### Phase 2: Custom Data Collection
* **Goal:** Gather personalized data to train the AI.
* **Gaze Data (5,000 - 15,000 samples):** Write a script that displays a moving red dot on the screen. As the user follows the dot, record:
  * Input features: Iris 3D coordinates, Head Pose (Pitch, Yaw, Roll).
  * Labels (Ground Truth): Screen X, Y coordinates of the dot.
* **Blink Data (500 - 1,000 samples):** Record sequences of Eye Aspect Ratio (EAR) calculations.
  * Labels: `0` (Open/Natural Blink), `1` (Intentional Click).

### Phase 3: AI Model Training
* **Goal:** Train lightweight AI models for prediction.
* **Gaze Model:** Train a Multi-Layer Perceptron (MLP) or Random Forest to predict Screen (X,Y) from the eye/head landmarks. Save as `.keras` or `.tflite` (You already have initial versions of these!).
* **Click Model:** Train a time-series classifier to detect the intentional blink threshold.

### Phase 4: Real-Time Calibration System
* **Goal:** Adapt to the user's current sitting position and screen size.
* **Action:** Build a 5-second startup calibration UI. The user looks at 5 dots (4 corners + center). Use Polynomial Regression to map the AI's predictions precisely to the user's specific monitor dimensions.

### Phase 5: OS Integration & Smoothing
* **Goal:** Actually control the computer.
* **Action:** Use `pyautogui` or `pynput` to move the mouse. 
* **Action:** Apply mathematical filters to the raw predicted (X,Y) arrays so the cursor moves smoothly rather than vibrating.

---

## 4. Major Hurdles & Paths to Overcome Them

### Hurdle 1: The "Midas Touch" Problem (False Clicks)
* **Problem:** If a normal blink triggers a click, the system is unusable. Humans blink 15-20 times a minute.
* **Solution:** Do not rely on a static "is eye closed?" rule. Measure the *duration* of the closure. Require a blink of exactly 400ms-600ms to trigger a click. Alternatively, implement "Dwell Clicking" (staring at an icon for 1 second clicks it).

### Hurdle 2: Head Movement Ruining Calibration
* **Problem:** If the user moves their head 2 inches to the left, but keeps their eyes fixed on the target, standard eye trackers will move the mouse. 
* **Solution:** Calculate Head Pose (yaw, pitch, roll) using MediaPipe face meshes. Feed this as an input into your Neural Network so the AI learns to subtract head movement from eye movement.

### Hurdle 3: Cursor Jitter
* **Problem:** Webcam pixel noise causes the detected iris to vibrate by 1-2 pixels every frame. Multiplied across a 1080p screen, the mouse cursor will shake violently.
* **Solution:** Never use raw predictions. Buffer the last 5-10 frame predictions and apply a **One Euro Filter** (specifically designed for HCI to reduce high-frequency noise while minimizing lag).

### Hurdle 4: Lighting Conditions
* **Problem:** Shadows on one side of the face can shift the detected pupil center.
* **Solution:** When collecting your dataset in Phase 2, ensure you record sessions in bright light, dim light, and side-lit environments to make the neural network robust.

---

## 5. Current Workspace Integration
You already have:
* `face_landmarker.task`: Ready for Phase 1.
* `best_eye_model.keras` & `eye_model.tflite`: Indicates Phase 3 has been partially completed or prototyped.
* `eye_tracker.py`: The main execution script.

**Next Immediate Step:** Review `eye_tracker.py` to evaluate how `best_eye_model.keras` is currently structured and identifying where the data collection & smoothing algorithms need to be injected.