import os

from absl import app
from vww_model import mobilenet_v1

import tensorflow as tf
assert tf.__version__.startswith('2')
import tensorflow_model_optimization as tfmot

IMAGE_SIZE = 96
BATCH_SIZE = 64
EPOCHS = 10

BASE_DIR = os.path.join(os.getcwd(), 'vw_coco2014_96')
SPLITS_DIR = os.path.join(os.getcwd(), 'splits')


def load_manifest(manifest_path):
  """Load image paths from manifest file."""
  with open(manifest_path, 'r') as f:
    return [line.strip() for line in f if line.strip()]


def create_generator_from_manifest(manifest_path, augment=False):
  """Create high-performance tf.data pipeline from manifest file."""
  image_paths = load_manifest(manifest_path)

  filepaths = [os.path.join(BASE_DIR, path) for path in image_paths]
  labels = [0 if path.startswith('non_person/') else 1 for path in image_paths]

  generator = tf.data.Dataset.from_tensor_slices((filepaths, labels))

  def parse_image(filename, label):
      image = tf.io.read_file(filename)
      image = tf.image.decode_jpeg(image, channels=3)
      image = tf.image.convert_image_dtype(image, tf.float32)
      image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
      label = tf.one_hot(label, depth=2)
      return image, label

  generator = generator.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

  if augment:
      generator = generator.shuffle(buffer_size=10000, reshuffle_each_iteration=True)

  generator = generator.batch(BATCH_SIZE)

  if augment:
      data_augmentation = tf.keras.Sequential([
          tf.keras.layers.RandomFlip("horizontal"),
          tf.keras.layers.RandomRotation(factor=10./360.),
          tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
          tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1)
      ])
      generator = generator.map(lambda x, y: (data_augmentation(x, training=True), y),
                                num_parallel_calls=tf.data.AUTOTUNE)

  generator = generator.prefetch(buffer_size=tf.data.AUTOTUNE)
  generator.class_indices = {'0': 0, '1': 1}
  return generator


def main(argv):
  train_generator = create_generator_from_manifest(
      os.path.join(SPLITS_DIR, 'train.txt'), augment=True)
  val_generator = create_generator_from_manifest(
      os.path.join(SPLITS_DIR, 'val.txt'), augment=False)

  # Load pretrained baseline
  base_model = tf.keras.models.load_model('trained_models/vww_96.h5')

  # Calculate pruning end step
  end_step = len(train_generator) * EPOCHS

  # Define pruning schedule: ramp from 0% to 50% sparsity
  pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
      initial_sparsity=0.0,
      final_sparsity=0.50,
      begin_step=0,
      end_step=end_step,
      frequency=100
  )

  # Wrap model with pruning
  model = tfmot.sparsity.keras.prune_low_magnitude(
      base_model, pruning_schedule=pruning_schedule)

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
      loss='categorical_crossentropy',
      metrics=['accuracy'])

  model.fit(
      train_generator,
      epochs=EPOCHS,
      validation_data=val_generator,
      callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])

  # Strip pruning wrappers and save
  pruned_model = tfmot.sparsity.keras.strip_pruning(model)
  pruned_model.save('trained_models/vww_96_pruned.h5')
  print("Saved pruned model to trained_models/vww_96_pruned.h5")


if __name__ == '__main__':
  app.run(main)
