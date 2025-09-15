import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands

def extract_sequence_from_video(video_path, seq_length=30):
    cap = cv2.VideoCapture(video_path)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
    
    frames = []
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
        frames.append(row_left + row_right)

    cap.release()
    hands.close()

    # Pad/trim sequence to fixed length
    if len(frames) < seq_length:
        # pad with last frame
        while len(frames) < seq_length:
            frames.append(frames[-1])
    else:
        frames = frames[:seq_length]

    return np.array(frames, dtype=np.float32)
