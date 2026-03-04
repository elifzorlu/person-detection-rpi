import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random


BASE_DIR = 'vw_coco2014_96'
SPLITS_DIR = 'splits'
IMAGE_SIZE = 96


def representative_dataset_gen():
    with open(os.path.join(SPLITS_DIR, 'train.txt'), 'r') as f:
        paths = [l.strip() for l in f if l.strip()]
    random.shuffle(paths)
    paths = paths[:1000]  # Use 1000 random samples for better calibration
    for path in paths:
            img = Image.open(os.path.join(BASE_DIR, path)).resize((IMAGE_SIZE, IMAGE_SIZE))
            img = np.array(img, dtype=np.float32) / 255.0
            yield [img[np.newaxis, ...]] # shape (1, 96, 96, 3)

            


# Load the trained model
model = tf.keras.models.load_model('trained_models/vww_96_labelsmooth_distill.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Apply optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

tflite_model = converter.convert()

# Save the model
with open('models/vww_96_labelsmooth_distill_int8.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Size: {len(tflite_model) / 1024:.1f} KB")