import cv2
import mediapipe as mp
import os
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,  # detect both hands
                       min_detection_confidence=0.5)

DATASET_DIR = "data"  # dataset structured as data/class_name/videos
OUTPUT_CSV = "landmarks_2hands.csv"

with open(OUTPUT_CSV, mode="w", newline="") as f:
    writer = csv.writer(f)
    # 2 hands × 21 landmarks × (x,y,z) = 126 features
    header = []
    for hand in ["left", "right"]:
        for i in range(21):
            header += [f"{hand}_x{i}", f"{hand}_y{i}", f"{hand}_z{i}"]
    header.append("label")
    writer.writerow(header)

    # Loop through gesture folders
    for gesture_label in os.listdir(DATASET_DIR):
        gesture_path = os.path.join(DATASET_DIR, gesture_label)
        if not os.path.isdir(gesture_path):
            continue

        # Loop through videos of each gesture
        for video_file in os.listdir(gesture_path):
            video_path = os.path.join(gesture_path, video_file)
            cap = cv2.VideoCapture(video_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                # Default = missing hand filled with -1s
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

                row = row_left + row_right + [gesture_label]
                writer.writerow(row)

            cap.release()

hands.close()
print("✅ Two-hand landmarks saved to", OUTPUT_CSV)
