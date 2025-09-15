import os
import numpy as np

# Path where your sign frames dataset is stored
DATASET_PATH = "processed_data_mediapipe"  # adjust if different

# Get all sign class folders
classes = sorted(os.listdir(DATASET_PATH))

# Save as numpy file
os.makedirs("models", exist_ok=True)
np.save("models/classes.npy", classes)

print("✅ Classes saved:", classes)
