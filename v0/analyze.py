#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load data
OUTPUT_DIR = "processed_data"
X = np.load(os.path.join(OUTPUT_DIR, "X_sentences.npy"))
y = np.load(os.path.join(OUTPUT_DIR, "y_sentences.npy"))
classes = np.load(os.path.join(OUTPUT_DIR, "sentence_labels.npy"))

print("=== Dataset Analysis ===")
print(f"Total samples: {X.shape[0]}")
print(f"Sequence length: {X.shape[1]}")
print(f"Feature dimension: {X.shape[2]}")
print(f"Number of classes: {len(classes)}")
print(f"Samples per class (average): {X.shape[0] / len(classes):.1f}")

# Analyze class distribution
class_counts = Counter(y)
print(f"\nClass distribution:")
print(f"Min samples per class: {min(class_counts.values())}")
print(f"Max samples per class: {max(class_counts.values())}")
print(f"Median samples per class: {np.median(list(class_counts.values())):.1f}")

# Show classes with very few samples
few_samples = {classes[k]: v for k, v in class_counts.items() if v <= 3}
if few_samples:
    print(f"\nClasses with ≤3 samples ({len(few_samples)} classes):")
    for class_name, count in sorted(few_samples.items()):
        print(f"  {class_name}: {count}")

# Check for data quality issues
print(f"\n=== Data Quality ===")
print(f"Zero sequences: {np.sum(np.all(X == 0, axis=(1,2)))}")
print(f"Feature range: [{X.min():.3f}, {X.max():.3f}]")
print(f"Mean feature value: {X.mean():.3f}")
print(f"Feature std: {X.std():.3f}")

# Analyze sequence lengths (non-zero frames)
actual_lengths = []
for seq in X:
    # Count non-zero frames (assuming padding is all zeros)
    non_zero_frames = np.sum(np.any(seq != 0, axis=1))
    actual_lengths.append(non_zero_frames)

print(f"\nActual sequence lengths:")
print(f"Min: {min(actual_lengths)}")
print(f"Max: {max(actual_lengths)}")
print(f"Mean: {np.mean(actual_lengths):.1f}")
print(f"Sequences using full length (40): {np.sum(np.array(actual_lengths) == 40)}")

# Recommendations
print(f"\n=== Recommendations ===")
if len(classes) > X.shape[0] / 5:  # More than 20% classes
    print("⚠️  Too many classes for dataset size. Consider:")
    print("   - Grouping similar classes")
    print("   - Using top N most frequent classes only")
    print("   - Collecting more data")

if min(class_counts.values()) < 5:
    print("⚠️  Some classes have very few samples. Consider:")
    print("   - Data augmentation")
    print("   - Removing classes with <5 samples")
    print("   - Using stratified sampling carefully")

if X.shape[2] > 512:  # High dimensionality
    print("⚠️  High feature dimensionality. Consider:")
    print("   - PCA dimensionality reduction")
    print("   - Smaller image resolution")
    print("   - Feature selection")

# Suggest better model parameters
print(f"\n=== Suggested Model Changes ===")
print("1. Reduce model complexity:")
print("   - Smaller LSTM units (32 instead of 64)")
print("   - Higher dropout (0.5)")
print("   - Lower learning rate (0.0001)")
print("2. Better preprocessing:")
print("   - Remove classes with <5 samples")
print("   - Reduce feature dimensions")
print("3. Training improvements:")
print("   - Longer patience for early stopping")
print("   - Class weights to handle imbalance")