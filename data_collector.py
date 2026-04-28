import cv2
import os
import time

def collect_images(gesture_dir, none_dir, gesture_name, min_samples=100):
    """Opens camera and returns True if user is ready to train."""
    cap = cv2.VideoCapture(0)
    count_g = len(os.listdir(gesture_dir))
    count_n = len(os.listdir(none_dir))

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        # UI Info
        cv2.putText(frame, f"Gesture '{gesture_name}': {count_g}/{min_samples}", (10, 30), 2, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"None: {count_n}/{min_samples}", (10, 60), 2, 0.7, (255,255,255), 2)
        
        ready = count_g >= min_samples and count_n >= min_samples
        if ready:
            cv2.putText(frame, "READY: Press 'T' to Train", (10, 100), 2, 0.8, (0, 255, 0), 2)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        fname = f"img_{int(time.time() * 1000)}.jpg"
        if key == ord('1'):
            cv2.imwrite(os.path.join(gesture_dir, fname), frame)
            count_g += 1
        elif key == ord('2'):
            cv2.imwrite(os.path.join(none_dir, fname), frame)
            count_n += 1
        elif key == ord('t') and ready:
            cap.release()
            cv2.destroyAllWindows()
            return True
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return False
    
    cap.release()
    cv2.destroyAllWindows()
    return False