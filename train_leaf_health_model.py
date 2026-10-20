
from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf

# -------------------------------
# 1. Project settings
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "LeafHealthDataset"
MODEL_DIR = BASE_DIR / "models"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 8
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR}"
    )

# -------------------------------
# 2. Find and label leaf images
# -------------------------------

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

healthy_images = []
diseased_images = []

for image_path in DATA_DIR.rglob("*"):
    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in valid_extensions:
        continue

    # Check folder names between the dataset root
    # and the image for a healthy label.
    relative_parts = image_path.relative_to(DATA_DIR).parts[:-1]

    is_healthy = any(
        "healthy" in folder.lower()
        for folder in relative_parts
    )

    if is_healthy:
        healthy_images.append(str(image_path))
    else:
        diseased_images.append(str(image_path))

print("\nDataset summary")
print("----------------")
print("Healthy images:", len(healthy_images))
print("Diseased/damaged images:", len(diseased_images))

if len(healthy_images) < 10 or len(diseased_images) < 10:
    raise ValueError(
        "Not enough images in one or both classes. "
        "Check your dataset folders and labels."
    )

# -------------------------------
# 3. Split into training,
#    validation, and test sets
# -------------------------------

def split_images(images):
    images = images.copy()
    random.shuffle(images)

    n = len(images)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)

    return (
        images[:train_end],
        images[train_end:val_end],
        images[val_end:]
    )

h_train, h_val, h_test = split_images(healthy_images)
d_train, d_val, d_test = split_images(diseased_images)

def combine_groups(healthy, diseased):
    paths = healthy + diseased
    labels = [0] * len(healthy) + [1] * len(diseased)

    combined = list(zip(paths, labels))
    random.shuffle(combined)

    paths, labels = zip(*combined)
    return list(paths), list(labels)

train_paths, train_labels = combine_groups(h_train, d_train)
val_paths, val_labels = combine_groups(h_val, d_val)
test_paths, test_labels = combine_groups(h_test, d_test)

print("\nSplit sizes")
print("Training:", len(train_paths))
print("Validation:", len(val_paths))
print("Testing:", len(test_paths))

# -------------------------------
# 4. Load and prepare images
# -------------------------------

def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)

    return image, tf.cast(label, tf.float32)

def make_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            len(paths),
            seed=SEED
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

train_ds = make_dataset(
    train_paths, train_labels, training=True
)
val_ds = make_dataset(val_paths, val_labels)
test_ds = make_dataset(test_paths, test_labels)

# -------------------------------
# 5. Build the AI model
# -------------------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))

x = tf.keras.applications.mobilenet_v2.preprocess_input(
    inputs
)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# 6. Train the model
# -------------------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# -------------------------------
# 7. Evaluate on unseen images
# -------------------------------

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nTest accuracy: {test_accuracy:.2%}")
print(f"Test loss: {test_loss:.4f}")

# -------------------------------
# 8. Save model and labels
# -------------------------------

model_path = MODEL_DIR / "leaf_health_model.keras"
labels_path = MODEL_DIR / "leaf_health_classes.json"

model.save(model_path)

with open(labels_path, "w", encoding="utf-8") as file:
    json.dump(
        {
            "0": "Healthy",
            "1": "Possibly diseased or damaged"
        },
        file,
        indent=4
    )

print("\nTraining completed!")
print("Model saved to:", model_path)
print("Labels saved to:", labels_path)
