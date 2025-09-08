#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf

# Config
OUTPUT_DIR = "processed_data"
SENT_MODEL_PATH = "sentence_lstm_improved.h5"  # Use the improved model
SENT_CLASSES_PATH = os.path.join(OUTPUT_DIR, "sentence_labels.npy")

# Match preprocessing
TARGET_SIZE = (32, 32)  # (W, H)
NORMALIZE = True
DTYPE = np.float32
MAX_SEQ_LEN = None  # infer from model
PREDICT_STRIDE = 3  # predict every N frames to save CPU
CONFIDENCE_THRESHOLD = 0.1  # Minimum confidence to display (adjusted for low accuracy model)

# GPU optimization
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🚀 GPU acceleration enabled: {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU setup failed: {e}")

def preprocess_frame(frame_bgr):
    # Grayscale, resize, normalize, flatten
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    g = g.astype(DTYPE)
    if NORMALIZE:
        g /= 255.0
    g = np.expand_dims(g, axis=-1)   # (H,W,1)
    return g.reshape(-1)             # (F,)

def main():
    print("🎥 Loading ISL Live Translation...")
    
    # Load sentence model and classes
    try:
        model = load_model(SENT_MODEL_PATH)
        classes = np.load(SENT_CLASSES_PATH)
        print(f"✅ Model loaded: {len(classes)} classes")
        
        # Load feature projection if it exists
        feature_projection = None
        if os.path.exists("feature_projection.pkl"):
            import joblib
            feature_projection = joblib.load("feature_projection.pkl")
            print("✅ Feature projection loaded")
            
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Infer (L,F) from model input
    _, L, F = model.input_shape
    print(f"📊 Model expects: {L} frames × {F} features")

    # Video capture with better settings
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: cannot open camera")
        return

    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Reduce OpenCV threads
    try:
        cv2.setNumThreads(2)  # Use 2 threads since you have GPU
    except Exception:
        pass

    buffer = []     # list of feature vectors (F,)
    last_pred = ""
    last_conf = 0.0
    frame_id = 0
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0

    print("🟢 Live translation started!")
    print("Controls: 'q' to quit, 'r' to reset buffer, 'c' to clear prediction")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            print("❌ Failed to read frame")
            break

        # Preprocess frame
        feat = preprocess_frame(frame)  # (F,)
        
        # Apply feature projection if available
        if feature_projection is not None:
            feat = feature_projection.transform(feat.reshape(1, -1)).flatten()
        
        buffer.append(feat)
        if len(buffer) > L:
            buffer = buffer[-L:]        # keep last L

        # Predict sparsely to save CPU/GPU
        if len(buffer) == L and (frame_id % PREDICT_STRIDE == 0):
            try:
                X = np.zeros((1, L, F), dtype=DTYPE)
                X[0, :, :] = np.stack(buffer, axis=0)
                
                # Predict with GPU acceleration
                probs = model.predict(X, verbose=0)[0]  # Get first (and only) sample
                best_idx = int(probs.argmax())
                confidence = float(probs[best_idx])
                
                # Only update if confidence is above threshold
                if confidence >= CONFIDENCE_THRESHOLD:
                    last_pred = classes[best_idx]
                    last_conf = confidence
                    
            except Exception as e:
                print(f"⚠️ Prediction error: {e}")

        # Calculate FPS
        fps_counter += 1
        if fps_counter % 30 == 0:
            current_fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()

        # Create display overlay
        overlay_color = (0, 255, 0) if last_conf >= 0.3 else (0, 165, 255)  # Green if confident, orange if uncertain
        
        # Main prediction
        if last_pred:
            label = f"Prediction: {last_pred}"
            confidence_text = f"Confidence: {last_conf:.3f}"
            cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, overlay_color, 2)
            cv2.putText(frame, confidence_text, (12, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, overlay_color, 2)
        else:
            cv2.putText(frame, "Analyzing...", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        # Status information
        status_color = (255, 255, 255)
        cv2.putText(frame, f"Buffer: {len(buffer)}/{L}", (12, frame.shape[0] - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (12, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        cv2.putText(frame, f"Frame: {frame_id}", (12, frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        cv2.putText(frame, "Q:Quit R:Reset C:Clear", (12, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

        # Show frame
        cv2.imshow("ISL Live Translation (GPU Accelerated)", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            buffer.clear()
            print("🔄 Buffer reset")
        elif key == ord('c'):
            last_pred = ""
            last_conf = 0.0
            print("🧹 Prediction cleared")

        frame_id += 1

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Live translation stopped")

if __name__ == "__main__":
    main()
