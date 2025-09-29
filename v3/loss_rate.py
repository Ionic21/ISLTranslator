import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the data using TensorFlow
# Define the image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load the training and validation datasets
train_data = image_dataset_from_directory(
    'archive/ASL_Gestures_36_Classes/train',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_data = image_dataset_from_directory(
    'archive/ASL_Gestures_36_Classes/train',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_data = image_dataset_from_directory(
    'archive/ASL_Gestures_36_Classes/test',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Get class names
class_names = train_data.class_names
num_classes = len(class_names)

# 2. Build the model
# Use a pre-trained MobileNetV2 as the base model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False # Freeze the base model

# Add new layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Train the model and get the history object
epochs = 20
history = model.fit(train_data,
                    validation_data=validation_data,
                    epochs=epochs)

# 4. Plot the loss curves from the history object
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 5. Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_data)
print(f"\nTest accuracy: {accuracy:.4f}")
print(f"Test loss: {loss:.4f}")

# 6. Generate confusion matrix, classification report, and F1 score
y_true = []
y_pred = []
for images, labels in test_data:
    y_true.extend(labels.numpy())
    predictions = model.predict(images)
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print(f"F1 Score (weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()