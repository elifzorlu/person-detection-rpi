import os

from absl import app
from vww_model import mobilenet_v1

import tensorflow as tf
assert tf.__version__.startswith('2')

IMAGE_SIZE = 96
BATCH_SIZE = 64
EPOCHS = 30
TEMPERATURE = 2.0
ALPHA = 0.7        # prioritize ground truth

# Fine-tune LR: start low, already near optimum at 83.33%
def get_lr(epoch):
    if epoch >= 20:
        return 1e-5   # settle
    elif epoch >= 10:
        return 3e-5   # refine
    return 5e-5       # gentle fine-tune

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
  train_generator = create_generator_from_manifest(
      os.path.join(SPLITS_DIR, 'train.txt'), augment=True)
  val_generator = create_generator_from_manifest(
      os.path.join(SPLITS_DIR, 'val.txt'), augment=False)

  # Warm-start from best 6-filter checkpoint, fine-tune with label smoothing
  teacher = tf.keras.models.load_model('trained_models/vww_96.h5')
  student = tf.keras.models.load_model('trained_models/vww_96_relu6_distill.h5')

  teacher.trainable = False  # freeze teacher

  optimizer = tf.keras.optimizers.Adam(get_lr(0))

  best_val_acc = 0.0

  for epoch in range(EPOCHS):
    # Apply stepped LR schedule
    current_lr = get_lr(epoch)
    optimizer.learning_rate.assign(current_lr)

    # Training
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_generator:
      with tf.GradientTape() as tape:
        student_logits = student(images, training=True)
        teacher_logits = teacher(images, training=False)

        # Convert probabilities back to logit-space: log(p) ≈ logit (constant cancels in softmax)
        teacher_log = tf.math.log(teacher_logits + 1e-10)
        student_log = tf.math.log(student_logits + 1e-10)

        # Soft targets from teacher (temperature scaling on logit-space)
        soft_teacher = tf.nn.softmax(teacher_log / TEMPERATURE)
        soft_student = tf.nn.softmax(student_log / TEMPERATURE)

        # Distillation loss (KL divergence)
        distill_loss = tf.reduce_mean(
            tf.keras.losses.KLD(soft_teacher, soft_student)) * (TEMPERATURE ** 2)

        # Label-smoothed cross-entropy (smoothing=0.1 prevents overconfidence)
        ce_loss = tf.reduce_mean(
            tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)(labels, student_logits))

        loss = ALPHA * ce_loss + (1 - ALPHA) * distill_loss

      grads = tape.gradient(loss, student.trainable_variables)
      optimizer.apply_gradients(zip(grads, student.trainable_variables))

      train_loss += loss.numpy()
      train_correct += tf.reduce_sum(
          tf.cast(tf.argmax(student_logits, 1) == tf.argmax(labels, 1), tf.int32)).numpy()
      train_total += images.shape[0]

    # Validation
    val_correct = 0
    val_total = 0
    for images, labels in val_generator:
      preds = student(images, training=False)
      val_correct += tf.reduce_sum(
          tf.cast(tf.argmax(preds, 1) == tf.argmax(labels, 1), tf.int32)).numpy()
      val_total += images.shape[0]

    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1}/{EPOCHS} [lr={current_lr:.0e}] - loss: {train_loss/len(train_generator):.4f} "
          f"- acc: {train_correct/train_total:.4f} - val_acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
      best_val_acc = val_acc
      student.save('trained_models/vww_96_labelsmooth_distill.h5')
      print(f"  Saved best model (val_acc={val_acc:.4f})")

  print(f"Training complete. Best val_acc: {best_val_acc:.4f}")


if __name__ == '__main__':
  app.run(main)
