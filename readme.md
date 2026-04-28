# Gesture Machine Learning Tools

This repository is designed to be a collaborative database of MediaPipe gesture task files and a toolset for creating new ones. The goal is for contributors to share their trained `.task` models while providing an easy-to-use pipeline for anyone to record and train custom hand gestures.

## Repository Goals
1.  **Shared Task Database**: A collection of pre-trained `.task` files that can be dropped into MediaPipe-powered applications. (Contributions are welcome!)
2.  **Custom Training Pipeline**: Tools to go from raw camera feed to a working machine learning model in minutes.

## How the Program Works
The project consists of three main phases implemented in the Python scripts:

### 1. Data Collection ([data_collector.py](data_collector.py))
The [`collect_images`](data_collector.py) function opens your webcam and allows you to capture frames for two categories:
-   **Target Gesture**: The specific hand sign you want to recognize.
-   **None/Neutral**: Background noise or resting hand positions to prevent false positives.

### 2. Model Training ([gesture_trainer.py](gesture_trainer.py))
Once enough samples are collected (min. 100), the [`run_training_pipeline`](gesture_trainer.py) uses the `mediapipe-model-maker`. It:
-   Loads the images from the `training_data/` folder.
-   Performs an 80/20 train/validation split.
-   Exports a quantized `.task` file into a specific `model_output_` directory.

### 3. Testing & Inference ([test.py](test.py))
The [`run_live_inference`](test.py) script allows you to verify your results. It loads the generated `.task` file and runs a live webcam feed, overlaying the predicted gesture name on the screen in real-time.

## Getting Started
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the main entry point to start collecting and training:
   ```sh
   python main.py
   ```
3. Test your model:
   ```sh
   python test.py
   ```
   *(Note: Update the `MODEL_PATH` in [test.py](test.py) to point to your new folder).*

## Contributing
If you train a unique gesture, please consider contributing your `.task` file to the repository so others can use it in their projects!