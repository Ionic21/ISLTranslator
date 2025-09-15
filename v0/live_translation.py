#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import joblib
from collections import deque
import threading
import queue

# Configuration  
OUTPUT_DIR = "processed_data"
WORD_MODEL_PATH = "word_classifier.pkl"
SENTENCE_MODEL_PATH = "sentence_lstm.h5"
SENTENCE_CLASSES_PATH = os.path.join(OUTPUT_DIR, "sentence_labels.npy")

# Match preprocessing parameters
TARGET_SIZE = (32, 32)
NORMALIZE = True
DTYPE = np.float32
BUFFER_SIZE = 40
PREDICT_STRIDE = 5
CONFIDENCE_THRESHOLD = 0.05

class SignLanguageTranslator:
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
        """Load available models"""
        print("🔄 Loading models...")
        
        # Load word classifier
        if os.path.exists(WORD_MODEL_PATH):
            try:
                self.word_model_data = joblib.load(WORD_MODEL_PATH)
                print(f"✅ Word model: {len(self.word_model_data['classes'])} classes")
                print(f"   Accuracy: {self.word_model_data.get('test_accuracy', 0):.3f}")
            except Exception as e:
                print(f"❌ Failed to load word model: {e}")
        
        # Load sentence model
        if os.path.exists(SENTENCE_MODEL_PATH):
            try:
                self.sentence_model = load_model(SENTENCE_MODEL_PATH)
                self.sentence_classes = np.load(SENTENCE_CLASSES_PATH)
                
                # Check training info
                if os.path.exists("training_info.npy"):
                    info = np.load("training_info.npy", allow_pickle=True).item()
                    accuracy = info.get('test_accuracy', 0)
                    print(f"✅ Sentence model: {len(self.sentence_classes)} classes")
                    print(f"   Accuracy: {accuracy:.3f}")
                    
                    if accuracy < 0.2:
                        print("⚠️ Low sentence accuracy, focusing on word predictions")
                else:
                    print(f"✅ Sentence model: {len(self.sentence_classes)} classes")
                    
            except Exception as e:
                print(f"❌ Failed to load sentence model: {e}")
                self.sentence_model = None
        
        if not self.word_model_data and not self.sentence_model:
            raise RuntimeError("No models available! Train models first.")
    
    def preprocess_frame(self, frame_bgr):
        """Convert frame to feature vector (matches training preprocessing)"""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        processed = resized.astype(DTYPE)
        if NORMALIZE:
            processed /= 255.0
        processed = np.expand_dims(processed, axis=-1)  # (H,W,1)
        return processed.flatten()  # (H*W*1,)
    
    def predict_word(self):
        """Predict word using time-pooled features"""
        if not self.word_model_data or len(self.feature_buffer) < 20:
            return
            
        try:
            # Create pooled features (same as training)
            seq_array = np.array(list(self.feature_buffer))
            mean_feat = seq_array.mean(axis=0)
            std_feat = seq_array.std(axis=0)
            pooled = np.concatenate([mean_feat, std_feat]).reshape(1, -1)
            
            # Predict
            model = self.word_model_data['model']
            classes = self.word_model_data['classes']
            probs = model.predict_proba(pooled)[0]
            
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
            # Prepare sequence (same as training)
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
    
    def draw_predictions(self, frame):
        """Draw predictions and UI on frame"""
        h, w = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_pos = 25
        
        # Word prediction
        if self.last_word_pred:
            color = (0, 255, 0) if self.last_word_conf > 0.6 else (0, 165, 255)
            word_text = f"Word: {self.last_word_pred}"
            conf_text = f"({self.last_word_conf:.2f})"
            
            cv2.putText(frame, word_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, conf_text, (10 + len(word_text) * 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y_pos += 35
        
        # Sentence prediction
        if self.last_sentence_pred:
            color = (0, 255, 0) if self.last_sentence_conf > 0.6 else (0, 165, 255)
            sent_text = f"Sentence: {self.last_sentence_pred}"
            conf_text = f"({self.last_sentence_conf:.2f})"
            
            cv2.putText(frame, sent_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, conf_text, (10 + len(sent_text) * 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y_pos += 35
        
        # Status information
        status_y = h - 60
        status_color = (200, 200, 200)
        
        # Buffer and FPS
        buffer_text = f"Buffer: {len(self.feature_buffer)}/{BUFFER_SIZE}"
        fps_text = f"FPS: {self.current_fps:.1f}"
        
        cv2.putText(frame, buffer_text, (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        cv2.putText(frame, fps_text, (10, status_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Controls
        controls_text = "Controls: Q=Quit, R=Reset, C=Clear"
        cv2.putText(frame, controls_text, (w - 350, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

def main():
    print("🎥 MS-ASL Live Translation")
    print("=" * 40)
    
    # Initialize translator
    try:
        translator = SignLanguageTranslator()
    except RuntimeError as e:
        print(f"❌ {e}")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Reduce CPU usage
    try:
        cv2.setNumThreads(2)
    except:
        pass
    
    frame_count = 0
    
    print("🟢 Translation started!")
    print("📝 Show signs to the camera for recognition")
    print("💡 Better lighting and clear gestures improve accuracy")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        # Mirror for better user experience
        frame = cv2.flip(frame, 1)
        
        # Process frame
        features = translator.preprocess_frame(frame)
        translator.feature_buffer.append(features)
        
        # Predict periodically
        if frame_count % PREDICT_STRIDE == 0:
            translator.predict_word()
            translator.predict_sentence()
        
        # Update performance metrics
        translator.update_fps()
        
        # Draw UI
        translator.draw_predictions(frame)
        
        # Display
        cv2.imshow("MS-ASL Live Translation", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            translator.feature_buffer.clear()
            print("🔄 Buffer reset")
        elif key == ord('c'):
            translator.last_word_pred = ""
            translator.last_sentence_pred = ""
            translator.last_word_conf = 0.0
            translator.last_sentence_conf = 0.0
            print("🧹 Predictions cleared")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Live translation stopped")

if __name__ == "__main__":
    main()
