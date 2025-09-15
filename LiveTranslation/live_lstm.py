# LiveTranslation/live_lstm.py
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

mp_hands = mp.solutions.hands
model = tf.keras.models.load_model("../Training/gesture_lstm.h5")

gesture_classes = ["hello", "come", "go"]  # same order as training
seq_length = 30
buffer = deque(maxlen=seq_length)

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
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
    buffer.append(row_left + row_right)

    if len(buffer) == seq_length:
        input_data = np.expand_dims(buffer, axis=0)
        preds = model.predict(input_data)[0]
        pred_class = np.argmax(preds)
        gesture = gesture_classes[pred_class]
        conf = preds[pred_class]

        cv2.putText(frame, f"{gesture} ({conf:.2f})", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("ISL LSTM Translation", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
