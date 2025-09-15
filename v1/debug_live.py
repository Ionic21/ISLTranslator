#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import cv2
import mediapipe as mp
import joblib
from collections import deque

# MediaPipe setup with lower thresholds
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,  # Much lower
    min_tracking_confidence=0.3    # Much lower
)

# Configuration
BUFFER_SIZE = 20  # Smaller buffer
PREDICT_STRIDE = 10  # Predict less frequently
CONFIDENCE_THRESHOLD = 0.1  # Very low threshold for testing

class SimpleSignTranslator:
    def __init__(self):
        self.word_model_data = None
        self.feature_buffer = deque(maxlen=BUFFER_SIZE)
        
        # Prediction state
        self.last_word_pred = ""
        self.last_word_conf = 0.0
        self.predictions_made = 0
        
        self.load_models()
    
    def load_models(self):
        """Load word model only"""
        print("🔄 Loading word model...")
        
        if os.path.exists("word_classifier_mediapipe.pkl"):
            try:
                self.word_model_data = joblib.load("word_classifier_mediapipe.pkl")
                print(f"✅ Word model loaded: {len(self.word_model_data['classes'])} classes")
                print(f"   Classes: {list(self.word_model_data['classes'])}")
            except Exception as e:
                print(f"❌ Failed to load word model: {e}")
        else:
            print("❌ word_classifier_mediapipe.pkl not found!")
            
        if not self.word_model_data:
            raise RuntimeError("No word model found!")
    
    def extract_simple_features(self, frame_bgr):
        """Extract simple hand features"""
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Simple feature extraction - just use first 10 landmarks of first hand
        features = []
        hands_detected = 0
        
        if results.multi_hand_landmarks:
            hands_detected = len(results.multi_hand_landmarks)
            
            # Use first hand only for simplicity
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract first 10 landmarks (30 values: x,y,z for each)
            for i, landmark in enumerate(hand_landmarks.landmark[:10]):
                features.extend([landmark.x, landmark.y, landmark.z])
                if i >= 9:  # Only first 10 landmarks
                    break
            
            # Pad to exactly 30 features
            while len(features) < 30:
                features.append(0.0)
                
        else:
            # No hands - use zeros
            features = [0.0] * 30
            
        return np.array(features, dtype=np.float32), results, hands_detected
    
    def predict_word_simple(self):
        """Simple prediction with debugging"""
        if not self.word_model_data or len(self.feature_buffer) < 10:
            return
        
        try:
            # Create simple statistical features
            seq_array = np.array(list(self.feature_buffer))
            mean_feat = seq_array.mean(axis=0)
            
            # Use only mean features for simplicity
            features = mean_feat.reshape(1, -1)
            
            # Predict
            model = self.word_model_data['model']
            classes = self.word_model_data['classes']
            
            probs = model.predict_proba(features)[0]
            best_idx = probs.argmax()
            confidence = probs[best_idx]
            
            print(f"🔍 Prediction: {classes[best_idx]} (conf: {confidence:.3f})")
            
            if confidence >= CONFIDENCE_THRESHOLD:
                self.last_word_pred = classes[best_idx]
                self.last_word_conf = confidence
                self.predictions_made += 1
                print(f"✅ Prediction accepted: {self.last_word_pred}")
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
    
    def draw_debug_info(self, frame, hands_results, hands_count):
        """Draw debug information"""
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
        cv2.rectangle(frame, (0, 0), (w, 150), (0, 0, 0), -1)
        
        y = 25
        color = (0, 255, 0) if self.last_word_conf > 0.3 else (0, 165, 255)
        
        # Current prediction
        if self.last_word_pred:
            cv2.putText(frame, f"Prediction: {self.last_word_pred}", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 25
            cv2.putText(frame, f"Confidence: {self.last_word_conf:.3f}", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25
        
        # Debug info
        cv2.putText(frame, f"Hands detected: {hands_count}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 20
        cv2.putText(frame, f"Buffer: {len(self.feature_buffer)}/{BUFFER_SIZE}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 20
        cv2.putText(frame, f"Predictions made: {self.predictions_made}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Controls
        cv2.putText(frame, "Q=Quit | R=Reset | C=Clear", (10, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def main():
    print("🤖 Simple MediaPipe Sign Translation (Debug Mode)")
    print("=" * 50)
    
    try:
        translator = SimpleSignTranslator()
    except RuntimeError as e:
        print(f"❌ {e}")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    frame_count = 0
    print("🟢 Translation started!")
    print("📝 Show your hands and make gestures")
    print("🔍 Check terminal for prediction debug info")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Extract features
        features, hands_results, hands_count = translator.extract_simple_features(frame)
        
        # Add to buffer if hands detected
        if hands_count > 0:
            translator.feature_buffer.append(features)
        
        # Predict occasionally
        if frame_count % PREDICT_STRIDE == 0:
            translator.predict_word_simple()
        
        # Draw everything
        translator.draw_debug_info(frame, hands_results, hands_count)
        
        cv2.imshow("Debug Sign Translation", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            translator.feature_buffer.clear()
            print("🔄 Buffer reset")
        elif key == ord('c'):
            translator.last_word_pred = ""
            translator.last_word_conf = 0.0
            print("🧹 Predictions cleared")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Translation stopped")

if __name__ == "__main__":
    main()
