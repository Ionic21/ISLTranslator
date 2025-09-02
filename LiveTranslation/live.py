import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle

# ===== Load TFLite model =====
interpreter = tf.lite.Interpreter(model_path="/home/ayush-daga/Documents/Sem3/ESW/SignLanguageDetector/Training/gesture_recognizer_2hands.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===== Load label encoder (must save it during training) =====
# If you didn’t save it, you need to re-fit on the same classes
# Example: pickle.dump(encoder, open("label_encoder.pkl", "wb")) in training script
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ===== MediaPipe Hands =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# ===== Webcam =====
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Default feature vector = two hands, each 63 vals (126 total)
    row_left = [-1] * 63
    row_right = [-1] * 63

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            coords = []
            for lm in hand_landmarks.landmark:
                coords += [lm.x, lm.y, lm.z]

            handedness = results.multi_handedness[idx].classification[0].label
            if handedness == "Left":
                row_left = coords
            else:
                row_right = coords

            # Draw landmarks on frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Combine into feature vector
    input_data = np.array(row_left + row_right, dtype=np.float32).reshape(1, -1)

    # Run inference only if at least one hand detected
    if results.multi_hand_landmarks:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        pred_class = np.argmax(output)
        pred_label = encoder.inverse_transform([pred_class])[0]
        confidence = output[pred_class]

        # Overlay prediction
        cv2.putText(frame, f"{pred_label} ({confidence:.2f})",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("ISL Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
