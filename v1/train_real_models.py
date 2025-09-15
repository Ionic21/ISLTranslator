#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

OUTPUT_DIR = "processed_data_real_msasl"

def train_real_word_classifier():
    """Train classifier on real MS-ASL data"""
    print("🎬 Training on Real MS-ASL Data - Word Classifier")
    print("=" * 50)
    
    # Load real data
    X = np.load(os.path.join(OUTPUT_DIR, "X_words.npy"))
    y = np.load(os.path.join(OUTPUT_DIR, "y_words.npy"))
    classes = np.load(os.path.join(OUTPUT_DIR, "word_labels.npy"))
    
    print(f"Real dataset: {X.shape[0]} samples, {len(classes)} classes")
    print(f"Feature dimension: {X.shape[1]} (MediaPipe features)")
    print(f"Classes: {list(classes)}")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for i, count in zip(unique, counts):
        print(f"  {classes[i]}: {count} samples")
    
    # Split data
    if len(set(y)) > 1 and min(counts) >= 2:
        stratify = y
    else:
        stratify = None
        print("⚠️ Cannot stratify - some classes have too few samples")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )
    
    # Train Random Forest (good for real data)
    clf = RandomForestClassifier(
        n_estimators=500,  # More trees for real data
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced real data
    )
    
    print("\n🌳 Training Random Forest...")
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = (y_test == y_pred).mean()
    
    print(f"\n📊 Real Data Results:")
    print(f"Test Accuracy: {accuracy:.3f}")
    
    if len(X_test) > 0:
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))
    
    # Feature importance
    feature_importance = clf.feature_importances_
    print(f"Feature importance range: {feature_importance.min():.4f} - {feature_importance.max():.4f}")
    
    # Save model
    model_data = {
        "model": clf,
        "classes": classes,
        "accuracy": accuracy,
        "feature_type": "real_mediapipe",
        "training_samples": len(X_train)
    }
    
    joblib.dump(model_data, "word_classifier_real_msasl.pkl")
    print("✅ Real MS-ASL word classifier saved!")
    
    return accuracy

def train_real_sentence_lstm():
    """Train LSTM on real MS-ASL sequences"""
    print("\n🎬 Training on Real MS-ASL Data - Sentence LSTM")
    print("=" * 50)
    
    # Load real data
    X = np.load(os.path.join(OUTPUT_DIR, "X_sentences.npy"))
    y = np.load(os.path.join(OUTPUT_DIR, "y_sentences.npy"))
    classes = np.load(os.path.join(OUTPUT_DIR, "sentence_labels.npy"))
    
    print(f"Real sequence dataset: {X.shape}")
    print(f"Classes: {len(classes)}")
    
    # Prepare for training
    y_cat = to_categorical(y)
    
    # Split data
    unique, counts = np.unique(y, return_counts=True)
    if min(counts) >= 2:
        stratify = y
    else:
        stratify = None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.25, random_state=42, stratify=stratify
    )
    
    print(f"Training split: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    # Build LSTM optimized for real data
    model = Sequential([
        tf.keras.layers.Masking(mask_value=0.0, input_shape=(X.shape[1], X.shape[2])),
        
        # Bidirectional LSTM for better sequence understanding
        tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True, dropout=0.3)),
        BatchNormalization(),
        
        LSTM(32, return_sequences=False, dropout=0.3),
        BatchNormalization(),
        
        Dense(32, activation='relu'),
        Dropout(0.4),
        Dense(len(classes), activation='softmax')
    ])
    
    # Compile with conservative settings for real data
    model.compile(
        optimizer=Adam(learning_rate=0.0003),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nLSTM Architecture for Real Data:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-6)
    ]
    
    # Train
    print("\n🚀 Training LSTM on real sequences...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,  # More epochs for real data
        batch_size=8,   # Smaller batches for limited data
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n📊 Real LSTM Results:")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"Test Loss: {test_loss:.3f}")
    
    # Save model
    model.save("sentence_lstm_real_msasl.h5")
    np.save("sentence_labels_real_msasl.npy", classes)
    
    print("✅ Real MS-ASL LSTM saved!")
    
    return test_accuracy

def main():
    print("🎬 Training Models on Real MS-ASL Data")
    print("=" * 45)
    
    # Check if real data exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"❌ Real data directory not found: {OUTPUT_DIR}")
        print("Please run train_real_msasl_mediapipe.py first!")
        return
    
    try:
        # Train both models
        word_acc = train_real_word_classifier()
        lstm_acc = train_real_sentence_lstm()
        
        print(f"\n🎯 Final Results on Real MS-ASL Data:")
        print(f"Word Classifier: {word_acc:.3f}")
        print(f"Sentence LSTM: {lstm_acc:.3f}")
        
        if word_acc > 0.6:
            print("🎉 Good accuracy on real data! Ready for real-world testing.")
        elif word_acc > 0.4:
            print("👍 Moderate accuracy. Consider collecting more diverse data.")
        else:
            print("⚠️ Low accuracy. May need more training data or parameter tuning.")
        
        print(f"\n📋 Models saved:")
        print(f"- word_classifier_real_msasl.pkl")
        print(f"- sentence_lstm_real_msasl.h5")
        
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        print("Please run train_real_msasl_mediapipe.py first!")

if __name__ == "__main__":
    main()
