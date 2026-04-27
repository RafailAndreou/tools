import os
from mediapipe_model_maker import gesture_recognizer

def run_training_pipeline(data_root, gesture_label, output_dir):
    """Encapsulates the MediaPipe training logic."""
    print(f"\n--- Training Model for: {gesture_label} ---")
    
    # Load dataset from the structured folders
    data = gesture_recognizer.Dataset.from_folder(
        dirname=data_root,
        hparams=gesture_recognizer.HandDataPreprocessingParams()
    )
    
    # Split data
    train_data, validation_data = data.split(0.8)
    
    # Configure and Train
    hparams = gesture_recognizer.HParams(export_dir=output_dir)
    options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
    
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options
    )
    
    # Export the final .task file
    model.export_model()
    print(f"\n[DONE] Model exported to {output_dir}")