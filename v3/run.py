import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path to your gesture recognizer model
model_path = 'exported_model/gesture_classifier.task'

# Aliases for convenience
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Callback function to handle results
def print_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if result.gestures:
        top_gesture = result.gestures[0][0]
        print(f"[{timestamp_ms} ms] Gesture={top_gesture.category_name}, "
              f"Score={top_gesture.score:.2f}")

# Create options for live stream mode
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

with GestureRecognizer.create_from_options(options) as recognizer:
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap into MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Current timestamp in ms
        timestamp_ms = int((time.time() - start_time) * 1000)

        # Send frame to recognizer
        recognizer.recognize_async(mp_image, timestamp_ms)

        # Show webcam feed
        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
