import os
import numpy as np
from landmark_extraction_sequences import extract_sequence_from_video

gestures = ["hello", "come", "go"]  # your gesture class names
data = []
labels = []

for idx, gesture in enumerate(gestures):
    folder = f"data/{gesture}"
    for filename in os.listdir(folder):
        if filename.endswith(".mp4"):
            seq = extract_sequence_from_video(os.path.join(folder, filename))
            data.append(seq)
            labels.append(idx)

X = np.array(data)       # shape: (samples, 30, 126)
y = np.array(labels)     # shape: (samples,)
np.save("X_sequences.npy", X)
np.save("y_sequences.npy", y)
