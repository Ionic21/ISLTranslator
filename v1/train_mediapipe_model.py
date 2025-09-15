#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# TensorFlow for LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

OUTPUT_DIR = "processed_data_mediapipe"

def train_word_classifier():
    """Train word-level classifier with MediaPipe features"""
    print("🤖 Training MediaPipe Word Classifier")
    print("=" * 40)
    
    # Load data
    X = np.load(os.path.join(OUTPUT_DIR, "X_words.npy"))
    y = np.load(os.path.join(OUTPUT_DIR, "y_words.npy"))
    classes = np.load(os.path.join(OUTPUT_DIR, "word_labels.npy"))
    
    print(f"Dataset: {X.shape[0]} samples, {len(classes)} classes")
    print(f"Feature dimension: {X.shape[1]} (MediaPipe landmarks)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train RandomForest (works well with MediaPipe features)
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    
    print("Training word classifier...")
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = (y_test == y_pred).mean()
    
    print(f"\n📊 Word Classifier Results:")
    print(f"Test Accuracy: {accuracy:.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))
    
    # Save model
    model_data = {
        "model": clf,
        "classes": classes,
        "accuracy": accuracy,
        "feature_type": "mediapipe"
    }
    
    joblib.dump(model_data, "word_classifier_mediapipe.pkl")
    print("✅ Word classifier saved as 'word_classifier_mediapipe.pkl'")
    
    return accuracy

def train_sentence_lstm():
    """Train sentence-level LSTM with MediaPipe features"""
    print("\n🤖 Training MediaPipe Sentence LSTM")
    print("=" * 40)
    
    # Setup GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    
    # Load data
    X = np.load(os.path.join(OUTPUT_DIR, "X_sentences.npy"))
    y = np.load(os.path.join(OUTPUT_DIR, "y_sentences.npy"))
    classes = np.load(os.path.join(OUTPUT_DIR, "sentence_labels.npy"))
    
    print(f"Dataset: {X.shape}")
    print(f"Classes: {len(classes)}")
    
    # Prepare data
    y_cat = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build LSTM model (optimized for MediaPipe features)
    model = Sequential([
        # Input layer with masking for variable-length sequences
        tf.keras.layers.Masking(mask_value=0.0, input_shape=(X.shape[1], X.shape[2])),
        
        # LSTM layers
        LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        
        LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        
        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(len(classes), activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Train
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
    ]
    
    print("\n🚀 Training LSTM...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n📊 LSTM Results:")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"Test Loss: {test_loss:.3f}")
    
    # Save model
    model.save("sentence_lstm_mediapipe.h5")
    np.save("sentence_labels_mediapipe.npy", classes)
    
    print("✅ LSTM model saved as 'sentence_lstm_mediapipe.h5'")
    
    return test_accuracy

def main():
    # Train both models
    word_acc = train_word_classifier()
    lstm_acc = train_sentence_lstm()
    
    print(f"\n🎯 Final Results:")
    print(f"Word Classifier Accuracy: {word_acc:.3f}")
    print(f"Sentence LSTM Accuracy: {lstm_acc:.3f}")
    
    if word_acc > 0.85 and lstm_acc > 0.85:
        print("🎉 Excellent results! Ready for live translation.")
    elif word_acc > 0.7 and lstm_acc > 0.7:
        print("👍 Good results! Should work well for live translation.")
    else:
        print("⚠️ Lower accuracy. Consider more training data or tuning.")

if __name__ == "__main__":
    main()
