#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

OUTPUT_DIR = "processed_data"
MODEL_PATH = "word_classifier.pkl"

# Load compact 2D features (N,F) and labels
X = np.load(os.path.join(OUTPUT_DIR, "X_words.npy"))
y = np.load(os.path.join(OUTPUT_DIR, "y_words.npy"))
classes = np.load(os.path.join(OUTPUT_DIR, "word_labels.npy"))

if X.size == 0 or y.size == 0:
    raise RuntimeError("Empty word dataset; run preprocessing first.")

print(f"Dataset info: {X.shape[0]} samples, {len(classes)} classes")

# Safer split: stratify only if all classes have >= 2 members
counts = np.bincount(y)
print(f"Class distribution: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

stratify = y if counts.min() >= 2 else None
if stratify is None:
    print("Warning: Some classes have only 1 sample. Stratification disabled.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# Lightweight RandomForest to keep memory small
clf = RandomForestClassifier(
    n_estimators=150, max_depth=None, n_jobs=-1, random_state=42
)

print("Training model...")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Get unique classes present in test set to avoid label mismatch
unique_test_classes = np.unique(np.concatenate([y_test, y_pred]))
test_class_names = classes[unique_test_classes]

print("Classification Report (Word-level):")
print(classification_report(
    y_test, y_pred, 
    labels=unique_test_classes,  # Specify which labels to include
    target_names=test_class_names
))

# Additional metrics
print(f"\nModel Performance Summary:")
print(f"Test Accuracy: {(y_test == y_pred).mean():.3f}")
print(f"Classes in test set: {len(unique_test_classes)}/{len(classes)}")

# Save model with metadata
model_data = {
    "model": clf,
    "classes": classes,
    "feature_shape": X.shape[1],
    "n_classes": len(classes),
    "test_accuracy": (y_test == y_pred).mean()
}

joblib.dump(model_data, MODEL_PATH)
print(f"✅ Word-level model saved at {MODEL_PATH}")

# Optional: Print classes not in test set for debugging
missing_classes = set(range(len(classes))) - set(unique_test_classes)
if missing_classes:
    print(f"\nNote: {len(missing_classes)} classes not present in test set:")
    for idx in sorted(list(missing_classes))[:10]:  # Show first 10
        print(f"  - {classes[idx]}")
    if len(missing_classes) > 10:
        print(f"  ... and {len(missing_classes) - 10} more")