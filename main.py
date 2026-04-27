import os
from data_collector import collect_images
from gesture_trainer import run_training_pipeline

def main():
    # 1. Get user input
    gesture_name = input("Enter the name of your new gesture: ").strip().lower()
    if not gesture_name: gesture_name = "new_gesture"

    # 2. Setup folder structure
    base_data_path = "training_data"
    gesture_path = os.path.join(base_data_path, gesture_name)
    none_path = os.path.join(base_data_path, "none")
    
    os.makedirs(gesture_path, exist_ok=True)
    os.makedirs(none_path, exist_ok=True)

    # 3. Phase 1: Collect Data
    print(f"\nStarting collection. Press '1' for {gesture_name}, '2' for Neutral.")
    should_train = collect_images(gesture_path, none_path, gesture_name)

    # 4. Phase 2: Train Model
    if should_train:
        output_folder = f"model_output_{gesture_name}"
        run_training_pipeline(base_data_path, gesture_name, output_folder)
    else:
        print("Training cancelled by user.")

if __name__ == "__main__":
    main()