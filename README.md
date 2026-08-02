# AI-Powered Eye Tracking for OS Control

Welcome to the **Eye Tracking OS Control** project! This repository contains a complete, end-to-end deep learning pipeline that allows you to control your Windows computer mouse purely using your eyes.

By fusing Google's real-time **MediaPipe Face Mesh**, a custom-trained **TensorFlow MLP**, and advanced signal filtering, this software achieves high-accuracy gaze tracking with a standard webcam.

---

## 🚀 The Three-Step Flow

To get this project running on your local machine, follow the three phases below in order. We have specifically designed this repository so you don't need to dig through unnecessary code—just follow the flow.

### Phase 1: Data Collection
Before you can control the mouse, the neural network needs to learn your specific eye geometry. 
We provide a local Flask-based web collector that captures your gaze data safely and quickly.

1. Run the web collector:
   ```bash
   python src/web_collector.py
   ```
2. Open your browser to `http://localhost:5000`.
3. Follow the on-screen red dots. The software will automatically capture your MediaPipe landmarks and map them to the screen coordinates, saving everything to a CSV file in the `dataset/` folder.

### Phase 2: Model Training
Once you have collected your training data (we recommend at least 3-4 sessions for high accuracy), it's time to train your personal AI.

We use a highly optimized Multi-Layer Perceptron (MLP) with a Cosine-Decaying Learning Rate Scheduler and synthetic Gaussian Noise data augmentation to achieve absolute minimal error.

1. Run the training pipeline:
   ```bash
   python src/train_gaze_model_v3.py
   ```
2. The script will automatically load all CSVs in your `dataset/` folder, train the model, and export a lightweight `eye_model.tflite` file.

*For a deeper dive into hyperparameter tuning and model architectures, please see our [Training Guide](docs/guide_for_training.md).*

### Phase 3: AI Mouse Control
With your `.tflite` model generated, you are ready to control your computer.

1. Launch the AI mouse controller:
   ```bash
   python src/gaze_mouse.py
   ```
2. The system will prompt you to do a quick 9-point session calibration. Look at the edges and center of your screen as instructed.
3. You now have full control of your mouse! 
   - **Look** to move the cursor.
   - **Blink** to click. (Blink detection is handled automatically via Eye Aspect Ratio geometry).

---

## 🛠️ Setup & Installation

### Requirements
- Python 3.12+
- A standard 720p/1080p Webcam

### Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/Aadarshttech/Eye-Tracking-OS-Control.git
cd Eye-Tracking-OS-Control
pip install -r requirements.txt
```
*(Note: If you are setting up the web collector for remote access, you may also run `npm install` to utilize `localtunnel` as defined in `package.json`.)*

---

## 🧠 Architecture Overview
- **Frontend/Capture:** OpenCV & MediaPipe (Extracts a 468-point face mesh, refined to 478 with iris tracking).
- **Backend/AI:** TensorFlow & Keras (Residual MLP with dropout and batch normalization).
- **Filtering:** 1 Euro Filter (Eliminates high-frequency webcam jitter).
- **Actuation:** PyAutoGUI (Translates mathematical coordinates to physical OS cursor movements).

*This project was developed for the final semester defense and successfully lowered the gaze tracking error to 27.4 pixels on a standard 1080p display.*
