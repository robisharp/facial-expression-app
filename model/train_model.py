"""
Facial Expression Recognition - Model Training
Uses FER2013 dataset with a CNN built in TensorFlow/Keras
Run: python train_model.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE   = 48
BATCH_SIZE = 64
EPOCHS     = 50
DATA_DIR   = "data"          # expects data/train/ and data/test/
MODEL_PATH = "fer_model.h5"
CLASSES    = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ── Data generators ──────────────────────────────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
)
val_gen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_gen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)
val_data = val_gen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)


# ── Model architecture ───────────────────────────────────────────────────────
def build_model(num_classes: int) -> tf.keras.Model:
    """VGG-style CNN tuned for 48×48 grayscale expression images."""
    model = models.Sequential([
        # Block 1
        layers.Conv2D(64, 3, padding="same", activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Classifier head
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


model = build_model(len(CLASSES))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ── Training callbacks ───────────────────────────────────────────────────────
cb = [
    callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
]

# ── Train ────────────────────────────────────────────────────────────────────
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=cb,
)

# ── Plot results ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy"); axes[0].legend()
axes[1].plot(history.history["loss"],         label="Train")
axes[1].plot(history.history["val_loss"],     label="Val")
axes[1].set_title("Loss"); axes[1].legend()
plt.tight_layout()
plt.savefig("training_curves.png")
print("✓ Training complete — model saved to", MODEL_PATH)
print("✓ Curves saved to training_curves.png")
