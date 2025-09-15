#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# Keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Suppress TF warnings
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

OUTPUT_DIR = "processed_data"
MODEL_PATH = "sentence_lstm.h5"
LABELS_OUT = "sentence_labels.npy"

def setup_gpu():
    """Configure GPU if available"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🚀 GPU acceleration enabled: {len(gpus)} GPU(s)")
            return True
        except RuntimeError as e:
            print(f"GPU setup failed: {e}")
    print("💻 Using CPU")
    return False

def main():
    print("MS-ASL Sentence LSTM Training")
    print("=" * 40)
    
    # Setup compute
    gpu_available = setup_gpu()
    
    # Load data
    X = np.load(os.path.join(OUTPUT_DIR, "X_sentences.npy"))
    y = np.load(os.path.join(OUTPUT_DIR, "y_sentences.npy"))
    classes = np.load(os.path.join(OUTPUT_DIR, "sentence_labels.npy"))

    if X.size == 0 or y.size == 0:
        raise RuntimeError("Empty sentence dataset; run preprocessing first.")

    print(f"Dataset: {X.shape[0]} samples, {len(classes)} classes")
    print(f"Input shape: (samples={X.shape[0]}, timesteps={X.shape[1]}, features={X.shape[2]})")

    # Check data quality
    class_counts = Counter(y)
    min_samples = min(class_counts.values())
    max_samples = max(class_counts.values())
    avg_samples = np.mean(list(class_counts.values()))
    
    print(f"Class distribution: min={min_samples}, max={max_samples}, avg={avg_samples:.1f}")

    # Filter classes with very few samples if needed
    if min_samples < 4:
        print("⚠️ Some classes have very few samples. Consider collecting more data.")

    # Convert to categorical
    y_cat = to_categorical(y)
    print(f"Categorical output shape: {y_cat.shape}")

    # Split data
    stratify = y if np.bincount(y).min() >= 2 else None
    test_size = 0.2 if X.shape[0] > 100 else 0.15  # Smaller test set if limited data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=test_size, random_state=42, stratify=stratify
    )

    print(f"Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

    # Compute class weights for imbalanced classes
    if len(np.unique(y)) > 1:
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"Class weights: {len(class_weight_dict)} classes balanced")
    else:
        class_weight_dict = None

    # Build model architecture
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(X.shape[1], X.shape[2])),
        
        # First LSTM layer
        LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        
        # Second LSTM layer  
        LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        
        # Dense layers
        Dense(32, activation="relu"),
        Dropout(0.4),
        Dense(y_cat.shape[1], activation="softmax")
    ])

    # Compile with appropriate learning rate
    initial_lr = 0.001 if gpu_available else 0.0005
    optimizer = Adam(learning_rate=initial_lr)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy", 
        metrics=["accuracy"]
    )

    print("\nModel Architecture:")
    model.summary()
    print(f"Total parameters: {model.count_params():,}")

    # Callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=7, 
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Training parameters
    epochs = 100 if X_train.shape[0] > 200 else 50
    batch_size = min(32, max(8, X_train.shape[0] // 20))  # Adaptive batch size

    print(f"\n🏋️ Training configuration:")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"Callbacks: EarlyStopping + ReduceLROnPlateau")

    # Train model
    print("\n🚀 Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Final evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n📊 Final Results:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Performance assessment
    if test_accuracy > 0.7:
        print("🎉 Excellent accuracy! Model ready for deployment.")
    elif test_accuracy > 0.5:
        print("👍 Good accuracy. Consider fine-tuning or more data.")
    elif test_accuracy > 0.3:
        print("⚠️ Moderate accuracy. More data recommended.")
    else:
        print("🔴 Low accuracy. Check data quality or try word-level model.")

    # Save model and metadata
    model.save(MODEL_PATH)
    np.save(LABELS_OUT, classes)

    # Save training info
    training_info = {
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "num_classes": len(classes),
        "num_samples": X.shape[0],
        "input_shape": X.shape,
        "epochs_trained": len(history.history['loss']),
        "final_lr": float(model.optimizer.learning_rate),
        "gpu_used": gpu_available
    }

    np.save("training_info.npy", training_info)

    print(f"\n✅ Training complete!")
    print(f"Model: {MODEL_PATH}")
    print(f"Classes: {LABELS_OUT}")
    print(f"Info: training_info.npy")
    
    # Prediction confidence analysis
    if test_accuracy > 0.2:  # Only if reasonable accuracy
        predictions = model.predict(X_test, verbose=0)
        max_probs = np.max(predictions, axis=1)
        avg_confidence = np.mean(max_probs)
        print(f"Average prediction confidence: {avg_confidence:.3f}")

if __name__ == "__main__":
    main()
