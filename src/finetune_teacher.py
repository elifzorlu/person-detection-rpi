import os

from absl import app

import tensorflow as tf
assert tf.__version__.startswith('2')

IMAGE_SIZE = 96
BATCH_SIZE = 64

BASE_DIR = os.path.join(os.getcwd(), 'vw_coco2014_96')
SPLITS_DIR = os.path.join(os.getcwd(), 'splits')


def load_manifest(manifest_path):
  with open(manifest_path, 'r') as f:
    return [line.strip() for line in f if line.strip()]


def create_generator_from_manifest(manifest_path, augment=False):
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
          tf.keras.layers.RandomRotation(factor=15./360.),
          tf.keras.layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
          tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
          tf.keras.layers.RandomBrightness(factor=0.1),
          tf.keras.layers.RandomContrast(factor=0.1),
      ])
      generator = generator.map(lambda x, y: (data_augmentation(x, training=True), y),
                                num_parallel_calls=tf.data.AUTOTUNE)

  generator = generator.prefetch(buffer_size=tf.data.AUTOTUNE)
  return generator


def main(argv):
  strategy = tf.distribute.MirroredStrategy()

  train_generator = create_generator_from_manifest(
      os.path.join(SPLITS_DIR, 'train.txt'), augment=True)
  val_generator = create_generator_from_manifest(
      os.path.join(SPLITS_DIR, 'val.txt'), augment=False)

  with strategy.scope():
    model = tf.keras.models.load_model('trained_models/vww_96.h5')
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'trained_models/vww_96_improved.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=30,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks)

  print("Fine-tuning complete. Best model saved to trained_models/vww_96_improved.h5")


if __name__ == '__main__':
  app.run(main)
