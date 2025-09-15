#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import joblib
from collections import deque

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Configuration
BUFFER_SIZE = 40
PREDICT_STRIDE = 3
CONFIDENCE_THRESHOLD = 0.4  # Higher threshold for better predictions

# MediaPipe feature extraction
HAND_LANDMARKS = 21
POSE_LANDMARKS = 12  # Upper body only
COORDS_PER_LANDMARK = 3
FEATURE_DIM = HAND_LANDMARKS * COORDS_PER_LANDMARK * 2 + POSE_LANDMARKS * COORDS_PER_LANDMARK

class MediaPipeSignTranslator:
    def __init__(self):
        self.word_model_data = None
        self.sentence_model = None
        self.sentence_classes = None
        self.feature_buffer = deque(maxlen=BUFFER_SIZE)
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Prediction state
        self.last_word_pred = ""
        self.last_sentence_pred = ""
        self.last_word_conf = 0.0
        self.last_sentence_conf = 0.0
        
        self.load_models()
    
    def load_models(self):
        """Load MediaPipe-trained models"""
        print("🔄 Loading MediaPipe models...")
        
        # Load word classifier
        if os.path.exists("word_classifier_mediapipe.pkl"):
            try:
                self.word_model_data = joblib.load("word_classifier_mediapipe.pkl")
                accuracy = self.word_model_data.get('accuracy', 0)
                print(f"✅ Word model: {len(self.word_model_data['classes'])} classes")
                print(f"   Accuracy: {accuracy:.3f}")
            except Exception as e:
                print(f"❌ Failed to load word model: {e}")
        
        # Load sentence model
        if os.path.exists("sentence_lstm_mediapipe.h5"):
            try:
                self.sentence_model = load_model("sentence_lstm_mediapipe.h5")
                self.sentence_classes = np.load("sentence_labels_mediapipe.npy")
                print(f"✅ Sentence model: {len(self.sentence_classes)} classes")
            except Exception as e:
                print(f"❌ Failed to load sentence model: {e}")
        
        if not self.word_model_data and not self.sentence_model:
            raise RuntimeError("No MediaPipe models found! Run training first.")
    
    def extract_mediapipe_features(self, frame_bgr):
        """Extract MediaPipe features from frame"""
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        hand_results = hands.process(rgb_frame)
        pose_results = pose.process(rgb_frame)
        
        features = []
        
        # Hand landmarks (up to 2 hands)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks[:2]:
                hand_features = []
                for landmark in hand_landmarks.landmark:
                    hand_features.extend([landmark.x, landmark.y, landmark.z])
                features.extend(hand_features)
        
        # Pad if less than 2 hands
        while len(features) < HAND_LANDMARKS * COORDS_PER_LANDMARK * 2:
            features.append(0.0)
        
        # Pose landmarks (upper body)
        if pose_results.pose_landmarks:
            upper_body_indices = list(range(11, 23))  # Shoulders to hands
            for idx in upper_body_indices:
                if idx < len(pose_results.pose_landmarks.landmark):
                    landmark = pose_results.pose_landmarks.landmark[idx]
                    features.extend([landmark.x, landmark.y, landmark.z])
                else:
                    features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0] * (POSE_LANDMARKS * COORDS_PER_LANDMARK))
        
        return np.array(features, dtype=np.float32), hand_results, pose_results
    
    def predict_word(self):
        """Predict word using MediaPipe features"""
        if not self.word_model_data or len(self.feature_buffer) < 20:
            return
        
        try:
            # Create statistical features from sequence
            seq_array = np.array(list(self.feature_buffer))
            mean_feat = seq_array.mean(axis=0)
            std_feat = seq_array.std(axis=0)
            max_feat = seq_array.max(axis=0)
            min_feat = seq_array.min(axis=0)
            
            # Combine features
            combined = np.concatenate([mean_feat, std_feat, max_feat, min_feat]).reshape(1, -1)
            
            # Predict
            model = self.word_model_data['model']
            classes = self.word_model_data['classes']
            probs = model.predict_proba(combined)[0]
            
            best_idx = probs.argmax()
            confidence = probs[best_idx]
            
            if confidence >= CONFIDENCE_THRESHOLD:
                self.last_word_pred = classes[best_idx]
                self.last_word_conf = confidence
            
        except Exception as e:
            print(f"Word prediction error: {e}")
    
    def predict_sentence(self):
        """Predict sentence using LSTM"""
        if not self.sentence_model or len(self.feature_buffer) < BUFFER_SIZE:
            return
        
        try:
            # Prepare sequence
            seq_array = np.array(list(self.feature_buffer))
            X = seq_array.reshape(1, BUFFER_SIZE, -1)
            
            # Predict
            probs = self.sentence_model.predict(X, verbose=0)[0]
            best_idx = probs.argmax()
            confidence = probs[best_idx]
            
            if confidence >= CONFIDENCE_THRESHOLD:
                self.last_sentence_pred = self.sentence_classes[best_idx]
                self.last_sentence_conf = confidence
            
        except Exception as e:
            print(f"Sentence prediction error: {e}")
    
    def update_fps(self):
        """Calculate FPS"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            elapsed = time.time() - self.fps_start_time
            self.current_fps = 30 / elapsed
            self.fps_start_time = time.time()
    
    def draw_mediapipe_landmarks(self, frame, hand_results, pose_results):
        """Draw MediaPipe landmarks on frame"""
        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
        
        # Draw pose landmarks (upper body only)
        if pose_results.pose_landmarks:
            # Only draw upper body connections
            upper_body_connections = [
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            ]
            
            for connection in upper_body_connections:
                start_idx = connection[0].value
                end_idx = connection[1].value
                
                if (start_idx < len(pose_results.pose_landmarks.landmark) and 
                    end_idx < len(pose_results.pose_landmarks.landmark)):
                    
                    start_landmark = pose_results.pose_landmarks.landmark[start_idx]
                    end_landmark = pose_results.pose_landmarks.landmark[end_idx]
                    
                    h, w = frame.shape[:2]
                    start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
                    end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
                    
                    cv2.line(frame, start_point, end_point, (255, 255, 0), 2)
    
    def draw_predictions(self, frame):
        """Draw predictions and UI"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_pos = 25
        
        # Word prediction
        if self.last_word_pred:
            color = (0, 255, 0) if self.last_word_conf > 0.7 else (0, 165, 255)
            word_text = f"Word: {self.last_word_pred}"
            conf_text = f"({self.last_word_conf:.2f})"
            
            cv2.putText(frame, word_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, conf_text, (10 + len(word_text) * 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y_pos += 35
        
        # Sentence prediction
        if self.last_sentence_pred:
            color = (0, 255, 0) if self.last_sentence_conf > 0.7 else (0, 165, 255)
            sent_text = f"Sentence: {self.last_sentence_pred}"
            conf_text = f"({self.last_sentence_conf:.2f})"
            
            cv2.putText(frame, sent_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, conf_text, (10 + len(sent_text) * 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Status info
        status_y = h - 80
        status_color = (200, 200, 200)
        
        cv2.putText(frame, f"MediaPipe Features: {len(self.feature_buffer)}/{BUFFER_SIZE}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                   (10, status_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Controls
        controls = "Q=Quit | R=Reset | C=Clear | MediaPipe ON"
        cv2.putText(frame, controls, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

def main():
    print("🤖 MediaPipe Sign Language Live Translation")
    print("=" * 50)
    
    # Initialize translator
    try:
        translator = MediaPipeSignTranslator()
    except RuntimeError as e:
        print(f"❌ {e}")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # Camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    
    print("🟢 MediaPipe translation started!")
    print("📝 Show sign language gestures to the camera")
    print("🤖 MediaPipe will track hand and pose landmarks")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror frame
        frame = cv2.flip(frame, 1)
        
        # Extract MediaPipe features
        features, hand_results, pose_results = translator.extract_mediapipe_features(frame)
        
        # Add to buffer
        if features is not None and len(features) == FEATURE_DIM:
            translator.feature_buffer.append(features)
        
        # Draw MediaPipe landmarks
        translator.draw_mediapipe_landmarks(frame, hand_results, pose_results)
        
        # Predict periodically
        if frame_count % PREDICT_STRIDE == 0:
            translator.predict_word()
            translator.predict_sentence()
        
        # Update FPS
        translator.update_fps()
        
        # Draw UI
        translator.draw_predictions(frame)
        
        # Display
        cv2.imshow("MediaPipe Sign Language Translation", frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            translator.feature_buffer.clear()
            print("🔄 Feature buffer reset")
        elif key == ord('c'):
            translator.last_word_pred = ""
            translator.last_sentence_pred = ""
            print("🧹 Predictions cleared")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 MediaPipe translation stopped")

if __name__ == "__main__":
    main()
