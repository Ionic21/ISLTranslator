#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "processed_data"
MODEL_PATH = "word_classifier.pkl"

def main():
    print("MS-ASL Word Classifier Training")
    print("=" * 40)
    
    # Load compact 2D features (N,F) and labels
    X = np.load(os.path.join(OUTPUT_DIR, "X_words.npy"))
    y = np.load(os.path.join(OUTPUT_DIR, "y_words.npy"))
    classes = np.load(os.path.join(OUTPUT_DIR, "word_labels.npy"))

    if X.size == 0 or y.size == 0:
        raise RuntimeError("Empty word dataset; run preprocessing first.")

    print(f"Dataset info: {X.shape[0]} samples, {len(classes)} classes")
    print(f"Feature dimension: {X.shape[1]}")

    # Check class distribution
    counts = np.bincount(y)
    print(f"Class distribution: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

    # Safer split: stratify only if all classes have >= 2 members
    stratify = y if counts.min() >= 2 else None
    if stratify is None:
        print("⚠️ Warning: Some classes have only 1 sample. Stratification disabled.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # Optimized RandomForest
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced classes
    )

    print("Training model...")
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    # Get classes present in test set
    unique_test_classes = np.unique(np.concatenate([y_test, y_pred]))
    test_class_names = classes[unique_test_classes]

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, 
        labels=unique_test_classes,
        target_names=test_class_names,
        zero_division=0
    ))

    # Calculate metrics
    accuracy = (y_test == y_pred).mean()
    confidence_scores = y_prob[np.arange(len(y_test)), y_test]
    avg_confidence = confidence_scores.mean()

    print(f"\n📊 Model Performance:")
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"Average Confidence: {avg_confidence:.3f}")
    print(f"Classes in test set: {len(unique_test_classes)}/{len(classes)}")

    # Feature importance
    feature_importance = clf.feature_importances_
    print(f"Feature importance range: {feature_importance.min():.6f} - {feature_importance.max():.6f}")

    # Save model with comprehensive metadata
    model_data = {
        "model": clf,
        "classes": classes,
        "feature_shape": X.shape[1],
        "n_classes": len(classes),
        "test_accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "class_distribution": counts,
        "training_samples": X_train.shape[0]
    }

    joblib.dump(model_data, MODEL_PATH)
    print(f"✅ Word-level model saved at {MODEL_PATH}")

    # Show classes not in test set for debugging
    missing_classes = set(range(len(classes))) - set(unique_test_classes)
    if missing_classes:
        print(f"\n📝 Note: {len(missing_classes)} classes not in test set:")
        for idx in sorted(list(missing_classes))[:5]:  # Show first 5
            print(f"  - {classes[idx]}")
        if len(missing_classes) > 5:
            print(f"  ... and {len(missing_classes) - 5} more")

    print(f"\n🎯 Expected live performance: {accuracy*0.8:.1%} - {accuracy*0.9:.1%}")

if __name__ == "__main__":
    main()
