# GazeTrack: AI-Powered Eye Tracking for OS Control

GazeTrack is an advanced, AI-driven eye-tracking system designed to replace a standard computer mouse. By leveraging Google MediaPipe for rapid 3D facial landmark extraction and a lightweight custom neural network, GazeTrack translates your gaze into precise screen coordinates and uses intentional blinks as mouse clicks.

## Key Features

- **Hybrid Landmark-AI Pipeline**: Combines the speed of MediaPipe's 468 3D facial landmarks with a custom-trained Neural Network for accurate, low-latency gaze prediction.
- **Intentional Blink Detection**: Differentiates between natural, involuntary blinks (~150ms) and intentional "click" blinks (400ms+) using a time-series classifier.
- **Advanced Jitter Reduction**: Implements a One Euro Filter to smooth cursor movements and prevent erratic screen jumping, providing a natural interaction.
- **Real-Time Calibration**: A fast calibration UI maps predictions precisely to any monitor's dimensions, accounting for user position and screen size.
- **Head Movement Compensation**: Incorporates head pose (pitch, yaw, roll) to ensure gaze remains accurate even when the user's head moves.

## Project Structure

- `data_collector.py`: Script to gather personalized gaze and blink data for training.
- `train_gaze_model.py`: Script to train lightweight AI models using the collected data.
- `eye_tracker.py`: Main execution script that runs the real-time tracking, applies smoothing filters, and interfaces with the OS.
- `Eye_Tracking_Project_Guide.md`: Comprehensive roadmap and technical details of the project phases.

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/GazeTrack.git
   cd GazeTrack
   ```

2. **Set up a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install opencv-python mediapipe numpy tensorflow pyautogui
   ```

4. **Ensure Models are Present**:
   Place `face_landmarker.task`, `best_eye_model.keras` or `eye_model.tflite` in the project root directory.

## Usage

1. **Run Calibration and Tracking**:
   ```bash
   python eye_tracker.py
   ```
2. **Follow on-screen instructions** to complete the quick calibration.
3. Use your eyes to move the cursor and perform an intentional blink (approx. 400-600ms) to click.
