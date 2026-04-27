import os
import shutil
import tensorflow as tf
from mediapipe_model_maker import gesture_recognizer

def train_custom_gesture(image_folder, gesture_label, non_gesture_folder=None, output_path="custom_model"):
    """
    Trains a new MediaPipe gesture and exports a .task file.
    
    Args:
        image_folder (str): Path to the folder containing your gesture images.
        gesture_label (str): The name/label for the new gesture.
        non_gesture_folder (str, optional): Path to neutral/none images. 
        output_path (str): Directory where the final .task file will be saved.
    """
    
    # 1. Setup Temporary Training Directory
    # MediaPipe requires: root/label_name/images
    train_root = "temp_training_data"
    gesture_dir = os.path.join(train_root, gesture_label)
    none_dir = os.path.join(train_root, "none")
    
    os.makedirs(gesture_dir, exist_ok=True)
    os.makedirs(none_dir, exist_ok=True)
    
    # 2. Populate folders (copying files to keep original data safe)
    def copy_images(src, dest):
        for img in os.listdir(src):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy(os.path.join(src, img), os.path.join(dest, img))

    copy_images(image_folder, gesture_dir)
    
    if non_gesture_folder and os.path.exists(non_gesture_folder):
        copy_images(non_gesture_folder, none_dir)
    else:
        print("Warning: No 'none' images provided. Model stability may be low.")
        # Create a dummy image or prompt user if 'none' is strictly required by the specific version
    
    # 3. Load Dataset
    # This automatically detects landmarks; images without hands are discarded
    data = gesture_recognizer.Dataset.from_folder(
        dirname=train_root,
        hparams=gesture_recognizer.HandDataPreprocessingParams()
    )
    
    # Split: 80% train, 20% validation/test
    train_data, validation_data = data.split(0.8)
    
    # 4. Train the Model
    # Using simple defaults to avoid unnecessary complexity
    hparams = gesture_recognizer.HParams(export_dir=output_path)
    options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
    
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options
    )
    
    # 5. Export the .task file
    model.export_model()
    print(f"Success! Model exported to: {output_path}/gesture_recognizer.task")

    # Cleanup temp folder (optional)
    # shutil.rmtree(train_root)

# Example Usage:
# train_custom_gesture("my_thumbs_up_imgs", "thumbs_up", "neutral_hand_imgs")