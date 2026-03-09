"""
Helmet Detection using CNN and Transfer Learning (VGG16)

Business Context
----------------
Construction and industrial environments require strict safety compliance.
One critical rule is wearing safety helmets to prevent head injuries.

This project builds an image classification model to detect whether
a worker is wearing a helmet.

Classes
-------
1 = With Helmet
0 = Without Helmet

Dataset
-------
631 images
- With Helmet: 311
- Without Helmet: 320
Image size: 200x200 RGB
"""

# ============================================================
# 1. Import Libraries
# ============================================================

import os
import random
import math
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
)
from tensorflow.keras.optimizers import Adam, SGD
from keras.applications.vgg16 import VGG16

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    classification_report,
)

warnings.filterwarnings("ignore")

tf.keras.utils.set_random_seed(812)
tf.config.experimental.enable_op_determinism()

# ============================================================
# 2. Load Dataset
# ============================================================

# Update path if running locally
DATA_PATH = "data"

images = np.load(os.path.join(DATA_PATH, "images_proj.npy"))
labels = pd.read_csv(os.path.join(DATA_PATH, "Labels_proj.csv"))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

# ============================================================
# 3. Convert BGR → RGB
# ============================================================

images_rgb = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images])

# ============================================================
# 4. Exploratory Data Analysis
# ============================================================

sns.countplot(x="Label", data=labels)
plt.title("Class Distribution")
plt.show()

# ============================================================
# 5. Data Preprocessing
# ============================================================

# Convert to grayscale
images_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images_rgb])
images_gray = images_gray[..., np.newaxis]

# Resize images
IMG_SIZE = 64

images_rgb_resized = np.array(
    [cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in images_rgb]
)

images_gray_resized = np.array(
    [cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in images_gray.squeeze()]
)

images_gray_resized = images_gray_resized[..., np.newaxis]

# ============================================================
# 6. Train / Validation / Test Split
# ============================================================

X_train_gray, X_temp_gray, X_train_rgb, X_temp_rgb, y_train, y_temp = train_test_split(
    images_gray_resized,
    images_rgb_resized,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels,
)

X_val_gray, X_test_gray, X_val_rgb, X_test_rgb, y_val, y_test = train_test_split(
    X_temp_gray,
    X_temp_rgb,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp,
)

# Normalize
X_train_gray = X_train_gray / 255.0
X_val_gray = X_val_gray / 255.0
X_test_gray = X_test_gray / 255.0

X_train_rgb = X_train_rgb / 255.0
X_val_rgb = X_val_rgb / 255.0
X_test_rgb = X_test_rgb / 255.0

# ============================================================
# 7. Utility Functions
# ============================================================

def model_performance_classification(model, predictors, target):

    pred = model.predict(predictors).reshape(-1) > 0.5
    target = target.to_numpy().reshape(-1)

    acc = accuracy_score(target, pred)
    recall = recall_score(target, pred, average="weighted")
    precision = precision_score(target, pred, average="weighted")
    f1 = f1_score(target, pred, average="weighted")

    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1 Score": f1},
        index=[0],
    )

    return df_perf


# ============================================================
# 8. Model 1 — Simple CNN
# ============================================================

tf.keras.backend.clear_session()

model_1 = Sequential()

model_1.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(64, 64, 1)))
model_1.add(MaxPooling2D((2, 2), padding="same"))

model_1.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model_1.add(MaxPooling2D((2, 2), padding="same"))

model_1.add(Flatten())
model_1.add(Dense(32, activation="relu"))
model_1.add(Dense(1, activation="sigmoid"))

model_1.compile(
    optimizer=SGD(learning_rate=0.01),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history_model1 = model_1.fit(
    X_train_gray,
    y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val_gray, y_val),
)

# ============================================================
# 9. Model 2 — VGG16 Base
# ============================================================

tf.keras.backend.clear_session()

vgg_base = VGG16(weights="imagenet", include_top=False, input_shape=(64, 64, 3))

for layer in vgg_base.layers:
    layer.trainable = False

model_2 = Sequential()

model_2.add(vgg_base)
model_2.add(Flatten())
model_2.add(Dense(1, activation="sigmoid"))

model_2.compile(
    optimizer=SGD(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history_model2 = model_2.fit(
    X_train_rgb,
    y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val_rgb, y_val),
)

# ============================================================
# 10. Model 3 — VGG16 + FFNN
# ============================================================

tf.keras.backend.clear_session()

vgg_base = VGG16(weights="imagenet", include_top=False, input_shape=(64, 64, 3))

for layer in vgg_base.layers:
    layer.trainable = False

model_3 = Sequential()

model_3.add(vgg_base)
model_3.add(Flatten())
model_3.add(Dense(128, activation="relu"))
model_3.add(Dropout(0.5))
model_3.add(Dense(64, activation="relu"))
model_3.add(Dense(1, activation="sigmoid"))

model_3.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history_model3 = model_3.fit(
    X_train_rgb,
    y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val_rgb, y_val),
)

# ============================================================
# 11. Model 4 — VGG16 + FFNN + Data Augmentation
# ============================================================

tf.keras.backend.clear_session()

vgg_base = VGG16(weights="imagenet", include_top=False, input_shape=(64, 64, 3))

for layer in vgg_base.layers:
    layer.trainable = False

model_4 = Sequential()

model_4.add(vgg_base)
model_4.add(Flatten())
model_4.add(Dense(128, activation="relu"))
model_4.add(Dropout(0.5))
model_4.add(Dense(64, activation="relu"))
model_4.add(Dense(1, activation="sigmoid"))

model_4.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

train_datagen = ImageDataGenerator(
    rotation_range=30,
    fill_mode="nearest",
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.4,
    zoom_range=0.5,
    horizontal_flip=True,
)

history_model4 = model_4.fit(
    train_datagen.flow(X_train_rgb, y_train, batch_size=128),
    epochs=20,
    steps_per_epoch=X_train_rgb.shape[0] // 128,
    validation_data=(X_val_rgb, y_val),
)

# ============================================================
# 12. Test Evaluation
# ============================================================

test_perf = model_performance_classification(model_4, X_test_rgb, y_test)

print("Test Performance")
print(test_perf)
