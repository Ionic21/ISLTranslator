#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import cv2
import mediapipe as mp
import joblib
from collections import deque
import tensorflow as tf

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Configuration
# The buffer size must match the sequence length used for training the LSTM on real data
BUFFER_SIZE = 30
PREDICT_STRIDE = 5
CONFIDENCE_THRESHOLD = 0.2

# MediaPipe feature dimensions (MUST match training)
HAND_LANDMARKS = 21
POSE_LANDMARKS = 12
COORDS_PER_LANDMARK = 3
FEATURE_DIM = HAND_LANDMARKS * COORDS_PER_LANDMARK * 2 + POSE_LANDMARKS * COORDS_PER_LANDMARK  # 162

class FixedSignTranslator:
    def __init__(self):
        self.lstm_model = None
        self.class_labels = None
        self.feature_buffer = deque(maxlen=BUFFER_SIZE)
        
        self.last_word_pred = ""
        self.last_word_conf = 0.0
        self.predictions_made = 0
        
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        print("🔄 Loading models...")
        
        # Load LSTM model from .h5 file
        MODEL_PATH = "sentence_lstm_real_msasl.h5"
        LABELS_PATH = "processed_data_real_msasl/sentence_labels.npy"
        
        if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
            try:
                self.lstm_model = tf.keras.models.load_model(MODEL_PATH)
                self.class_labels = np.load(LABELS_PATH, allow_pickle=True)
                print(f"✅ LSTM model loaded: {len(self.class_labels)} classes")
                print(f"   Classes: {list(self.class_labels)}")
            except Exception as e:
                print(f"❌ Failed to load LSTM model: {e}")
        else:
            print(f"❌ {MODEL_PATH} or {LABELS_PATH} not found!")
            
        if not self.lstm_model:
            raise RuntimeError("No LSTM model found!")
    
    def extract_mediapipe_features(self, frame_bgr):
        """Extract both hands and pose features (matching training format)"""
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)
        pose_results = pose.process(rgb_frame)
        
        features = []
        hands_detected = 0
        
        # Hand landmarks (up to 2 hands)
        if hand_results.multi_hand_landmarks:
            hands_detected = len(hand_results.multi_hand_landmarks)
            for hand_landmarks in hand_results.multi_hand_landmarks[:2]:
                hand_features = []
                for landmark in hand_landmarks.landmark:
                    hand_features.extend([landmark.x, landmark.y, landmark.z])
                features.extend(hand_features)
        
        # Pad if less than 2 hands detected
        while len(features) < HAND_LANDMARKS * COORDS_PER_LANDMARK * 2:
            features.append(0.0)

        # Pose landmarks (shoulders to hands)
        if pose_results.pose_landmarks:
            upper_body_indices = list(range(11, 23)) # Shoulders to hands
            for idx in upper_body_indices:
                if idx < len(pose_results.pose_landmarks.landmark):
                    landmark = pose_results.pose_landmarks.landmark[idx]
                    features.extend([landmark.x, landmark.y, landmark.z])
                else:
                    features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0] * (POSE_LANDMARKS * COORDS_PER_LANDMARK))

        return np.array(features, dtype=np.float32), hand_results, hands_detected
    
    def predict_word(self):
        """Predict using the LSTM model"""
        if not self.lstm_model or len(self.feature_buffer) < BUFFER_SIZE:
            return
        
        try:
            input_data = np.expand_dims(list(self.feature_buffer), axis=0)
            
            print(f"🔍 Feature shape: {input_data.shape} (should be (1, {BUFFER_SIZE}, {FEATURE_DIM}))")
            
            probabilities = self.lstm_model.predict(input_data, verbose=0)[0]
            best_idx = np.argmax(probabilities)
            confidence = probabilities[best_idx]
            
            print(f"🔍 Prediction: {self.class_labels[best_idx]} (conf: {confidence:.3f})")
            
            if confidence >= CONFIDENCE_THRESHOLD:
                self.last_word_pred = self.class_labels[best_idx]
                self.last_word_conf = confidence
                self.predictions_made += 1
                print(f"✅ Prediction accepted: {self.last_word_pred}")
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")

    def draw_ui(self, frame, hands_results, hands_count):
        """Draw UI and landmarks"""
        h, w = frame.shape[:2]
        
        # Draw hand landmarks
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
        
        # Background for text
        cv2.rectangle(frame, (0, 0), (w, 130), (0, 0, 0), -1)
        
        y = 25
        
        # Current prediction
        if self.last_word_pred:
            color = (0, 255, 0) if self.last_word_conf > 0.5 else (0, 165, 255)
            cv2.putText(frame, f"Sign: {self.last_word_pred}", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y += 30
            cv2.putText(frame, f"Confidence: {self.last_word_conf:.2f}", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25
        
        # Status info
        status_color = (255, 255, 255)
        cv2.putText(frame, f"Hands: {hands_count} | Buffer: {len(self.feature_buffer)}/{BUFFER_SIZE}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        y += 20
        cv2.putText(frame, f"Predictions: {self.predictions_made}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Controls
        cv2.putText(frame, "Q=Quit | R=Reset | C=Clear", (10, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

def main():
    print("🤖 Live Sign Translation (Real MS-ASL Model)")
    print("=" * 40)
    
    try:
        translator = FixedSignTranslator()
    except RuntimeError as e:
        print(f"❌ {e}")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    frame_count = 0
    print("🟢 Translation started!")
    print("📝 Show sign language gestures")
    print("🔍 Check terminal for prediction info")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        features, hands_results, hands_count = translator.extract_mediapipe_features(frame)
        
        print(f"📊 Feature vector size: {len(features)} (should be 162)")
        
        if hands_count > 0:
            translator.feature_buffer.append(features)
        
        if frame_count % PREDICT_STRIDE == 0 and len(translator.feature_buffer) == BUFFER_SIZE:
            translator.predict_word()
        
        translator.draw_ui(frame, hands_results, hands_count)
        
        cv2.imshow("Live Sign Translation", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            translator.feature_buffer.clear()
            print("🔄 Buffer reset")
        elif key == ord('c'):
            translator.last_word_pred = ""
            translator.last_word_conf = 0.0
            translator.predictions_made = 0
            print("🧹 Predictions cleared")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Translation stopped")

if __name__ == "__main__":
    main()