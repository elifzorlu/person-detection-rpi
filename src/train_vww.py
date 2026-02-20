# Lint as: python3
"""Training for the visual wakewords person detection model.

The visual wakewords person detection model is a core model for the TinyMLPerf
benchmark suite. This script provides source for how the reference model was
created and trained, and can be used as a starting point for open submissions
using re-training.

Source: https://github.com/mlcommons/tiny/blob/master/benchmark/training/visual_wake_words/train_vww.py
"""

import os

from absl import app
from vww_model import mobilenet_v1

import tensorflow as tf
assert tf.__version__.startswith('2')

IMAGE_SIZE = 96
BATCH_SIZE = 64
EPOCHS = 20

BASE_DIR = os.path.join(os.getcwd(), 'vw_coco2014_96')
SPLITS_DIR = os.path.join(os.getcwd(), 'splits')


def load_manifest(manifest_path):
  """Load image paths from manifest file."""
  with open(manifest_path, 'r') as f:
    return [line.strip() for line in f if line.strip()]


def create_generator_from_manifest(manifest_path, augment=False):
  """Create high-performance tf.data pipeline from manifest file."""
  image_paths = load_manifest(manifest_path)
  
  # Create full paths and labels
  filepaths = [os.path.join(BASE_DIR, path) for path in image_paths]
  labels = [0 if path.startswith('non_person/') else 1 for path in image_paths]
  
  # 1. Create native tf.data dataset
  generator = tf.data.Dataset.from_tensor_slices((filepaths, labels))
  
  # 2. Define parsing function
  def parse_image(filename, label):
      image = tf.io.read_file(filename)
      image = tf.image.decode_jpeg(image, channels=3)
      # convert_image_dtype automatically scales values from [0, 255] to [0.0, 1.0]
      image = tf.image.convert_image_dtype(image, tf.float32)
      image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
      # One-hot encode label to match 'categorical_crossentropy' in train_epochs
      label = tf.one_hot(label, depth=2)
      return image, label

  # 3. Apply parsing (parallelized across CPU cores)
  generator = generator.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

  # 4. Shuffle before batching
  if augment:
      generator = generator.shuffle(buffer_size=10000, reshuffle_each_iteration=True)

  # 5. Batch the data
  generator = generator.batch(BATCH_SIZE)

  # 6. Apply augmentations on batched tensors (much faster)
  if augment:
      data_augmentation = tf.keras.Sequential([
          tf.keras.layers.RandomFlip("horizontal"),
          tf.keras.layers.RandomRotation(factor=10./360.), # 10 degrees rotation
          tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
          tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1)
      ])
      generator = generator.map(lambda x, y: (data_augmentation(x, training=True), y),
                                num_parallel_calls=tf.data.AUTOTUNE)

  # 7. Prefetch for GPU optimization (fixes the bottleneck)
  generator = generator.prefetch(buffer_size=tf.data.AUTOTUNE)
  
  # Hack to maintain compatibility with original print statement in main()
  generator.class_indices = {'0': 0, '1': 1}
  
  return generator


def main(argv):
  strategy = tf.distribute.MirroredStrategy()
  print(f'Number of devices running in sync: {strategy.num_replicas_in_sync}')

  # Load data generators (data pipeline creation stays outside the scope)
  train_generator = create_generator_from_manifest(
      os.path.join(SPLITS_DIR, 'train.txt'), augment=True)
  val_generator = create_generator_from_manifest(
      os.path.join(SPLITS_DIR, 'val.txt'), augment=False)
  
  print(train_generator.class_indices)

  # 2. Open the strategy scope. Everything related to model creation 
  # and compilation MUST happen inside this block.
  with strategy.scope():
    if len(argv) >= 2:
      model = tf.keras.models.load_model(argv[1])
    else:
      model = mobilenet_v1()
      
    model.summary()

    # We call train_epochs inside the scope so the internal model.compile() 
    # correctly attaches the optimizer variables to both GPUs.
    model = train_epochs(model, train_generator, val_generator, 20, 0.001)
    model = train_epochs(model, train_generator, val_generator, 10, 0.0005)
    model = train_epochs(model, train_generator, val_generator, 20, 0.00025)

  # 3. Save the final model (can be safely done outside the scope)
  if len(argv) >= 3:
    model.save(argv[2])
  else:
    model.save('trained_models/vww_96.h5')


def train_epochs(model, train_generator, val_generator, epoch_count,
                 learning_rate):
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate),
      loss='categorical_crossentropy',
      metrics=['accuracy'])
  history_fine = model.fit(
      train_generator,
      steps_per_epoch=len(train_generator),
      epochs=epoch_count,
      validation_data=val_generator,
      validation_steps=len(val_generator),
      batch_size=BATCH_SIZE)
  return model


if __name__ == '__main__':
  app.run(main)
