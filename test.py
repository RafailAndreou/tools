import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Paths to your .task files
MODEL_PATH = 'model_output_middle_finger/gesture_recognizer.task' # Update this to your file path

def run_live_inference(model_path):
    # Initialize the detector/recognizer
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(0)

    print("Starting live loop. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Recognize gestures
        recognition_result = recognizer.recognize(mp_image)

        # Draw results on the frame
        if recognition_result.gestures:
            gesture_name = recognition_result.gestures[0][0].category_name
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('MediaPipe Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_inference(MODEL_PATH)