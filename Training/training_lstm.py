import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Load data
X = np.load("X_sequences.npy")
y = np.load("y_sequences.npy")

num_classes = len(np.unique(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Build model
model = models.Sequential([
    layers.Masking(mask_value=-1.0, input_shape=(30, 126)),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30, batch_size=16,
                    validation_data=(X_test, y_test))

model.save("gesture_lstm.h5")
