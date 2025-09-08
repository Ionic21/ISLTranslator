#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# Import keras from tensorflow for compatibility
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Suppress excessive logging
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

print("TensorFlow version:", tf.__version__)

# Configure GPU/CPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU setup failed: {e}")

OUTPUT_DIR = "processed_data"
MODEL_PATH = "sentence_lstm_improved.h5"
LABELS_OUT = "sentence_label_encoder_improved.npy"

# Load data
X = np.load(os.path.join(OUTPUT_DIR, "X_sentences.npy"))
y = np.load(os.path.join(OUTPUT_DIR, "y_sentences.npy"))
classes_original = np.load(os.path.join(OUTPUT_DIR, "sentence_labels.npy"))

if X.size == 0 or y.size == 0:
    raise RuntimeError("Empty sentence dataset; run preprocessing first.")

print(f"Original dataset: {X.shape[0]} samples, {len(classes_original)} classes")

# Filter out classes with very few samples (< 4 samples)
class_counts = Counter(y)
min_samples = 4
valid_classes = [class_id for class_id, count in class_counts.items() if count >= min_samples]

print(f"Removing {len(classes_original) - len(valid_classes)} classes with <{min_samples} samples")

# Filter dataset
mask = np.isin(y, valid_classes)
X_filtered = X[mask]
y_filtered = y[mask]

# Re-encode labels to be contiguous
valid_class_names = classes_original[valid_classes]
le = LabelEncoder()
y_filtered_encoded = le.fit_transform(y_filtered)
y_cat = to_categorical(y_filtered_encoded)

print(f"Filtered dataset: {X_filtered.shape[0]} samples, {len(valid_class_names)} classes")

# Optional: Reduce feature dimensionality using random projection (faster than PCA)
from sklearn.random_projection import GaussianRandomProjection

N, L, F = X_filtered.shape
if F > 256:  # If features are high-dimensional
    target_dim = min(256, F // 2)
    print(f"Reducing feature dimension from {F} to {target_dim}")
    
    # Reshape for dimensionality reduction
    X_reshaped = X_filtered.reshape(-1, F)
    rp = GaussianRandomProjection(n_components=target_dim, random_state=42)
    X_reduced = rp.fit_transform(X_reshaped)
    X_filtered = X_reduced.reshape(N, L, target_dim)
    F = target_dim
    
    # Save the projection for later use
    import joblib
    joblib.dump(rp, "feature_projection.pkl")
    print(f"Feature projection saved to feature_projection.pkl")

print(f"Final data shape: {X_filtered.shape}")

# Split data with stratification
stratify = y_filtered_encoded if np.bincount(y_filtered_encoded).min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_cat, test_size=0.2, random_state=42, stratify=stratify
)

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Classes: {y_cat.shape[1]}")

# Compute class weights to handle remaining imbalance
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_filtered_encoded),
    y=y_filtered_encoded
)
class_weight_dict = dict(enumerate(class_weights))

# Build improved model
model = Sequential([
    Masking(mask_value=0.0, input_shape=(L, F)),
    LSTM(32, return_sequences=True),  # Reduced size, keep sequences
    Dropout(0.4),
    LSTM(32, return_sequences=False),  # Second LSTM layer
    Dropout(0.4),
    BatchNormalization(),  # Add batch normalization
    Dense(32, activation="relu"),  # Smaller dense layer
    Dropout(0.5),
    Dense(y_cat.shape[1], activation="softmax"),
])

# Use lower learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

print("Model summary:")
model.summary()

# Better callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',  # Monitor accuracy instead of loss
        patience=10,  # More patience
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-7,
        verbose=1
    )
]

# Train model
print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,  # More epochs with early stopping
    batch_size=16,  # Larger batch size
    callbacks=callbacks,
    class_weight=class_weight_dict,  # Handle class imbalance
    verbose=1
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")

# Check if accuracy is reasonable
if test_accuracy < 0.1:  # Less than 10%
    print("⚠️ WARNING: Very low accuracy detected!")
    print("Possible issues:")
    print("- Dataset too small for number of classes")
    print("- Need better feature engineering")
    print("- Consider using word-level model instead")

# Save model and metadata
model.save(MODEL_PATH)
np.save(LABELS_OUT, valid_class_names)

# Save training info
training_info = {
    "test_accuracy": test_accuracy,
    "num_classes": len(valid_class_names),
    "num_samples": X_filtered.shape[0],
    "feature_dim": F,
    "min_samples_per_class": min_samples
}

np.save("training_info.npy", training_info)

print(f"✅ Improved model saved at {MODEL_PATH}")
print(f"✅ Labels saved at {LABELS_OUT}")
print(f"✅ Training info saved at training_info.npy")

# Show class distribution in final dataset
final_counts = Counter(y_filtered_encoded)
print(f"\nFinal class distribution: {min(final_counts.values())}-{max(final_counts.values())} samples per class")