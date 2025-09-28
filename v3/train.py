from mediapipe_model_maker import gesture_recognizer
import tensorflow as tf # Often needed for backend operations

# Verify TensorFlow version (Model Maker requires a specific version)
# print(tf.__version__)

# 1. Load the data
data = gesture_recognizer.Dataset.from_folder("archive/ASL_Gestures_36_Classes/train")
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

# 2. Create and train the model
hparams = gesture_recognizer.HParams(
    export_dir='./exported_model',
    epochs=20,
    batch_size=4,
    learning_rate=0.001
)
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)

# 3. Evaluate the model
loss, accuracy = model.evaluate(test_data)
print(f"Test accuracy: {accuracy}")
print(f"Test loss: {loss}")

# 4. Export the model as .task file
model.export_model('gesture_classifier.task')
print("Model export complete.")