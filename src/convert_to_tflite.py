import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('trained_models/vww_96.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Apply optimizations
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset_gen
tflite_model = converter.convert()

# Save the model
with open('trained_models/vww_96.tflite', 'wb') as f:
    f.write(tflite_model)