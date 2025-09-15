import tensorflow as tf
import os

# Set the path to your trained model
MODEL_PATH = "sentence_lstm_real_msasl.h5"

# Load the trained Keras model
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize the TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set the conversion flags to include both built-in and select TensorFlow ops
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # The standard built-in ops
    tf.lite.OpsSet.SELECT_TF_OPS    # The ops needed for your model, not included in built-ins
]

# Set experimental flag to false to prevent errors with tensor lists
converter._experimental_lower_tensor_list_ops = False

print("Converting model to TFLite for Android compatibility...")

try:
    # Perform the conversion
    tflite_model = converter.convert()

    # Save the TFLite model
    with open('lstm_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("✅ Model converted successfully. The file is now compatible with Android TFLite interpreter.")

except Exception as e:
    print(f"❌ Conversion failed with an error: {e}")