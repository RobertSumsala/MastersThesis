from reportlab.lib.pdfencrypt import padding
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision

from ConnorCode.ImageUtils import ImageEditor
from ConnorCode.CommonUtils import Misc

# =============================================================
# ✅ Enable Mixed Precision Training (saves a LOT of VRAM)
# =============================================================
mixed_precision.set_global_policy("mixed_float16")

'''Trains CustomCNN in the same way the `train_custom_cnn` function does, 
except the train, validation and test image data are fixed.
train_data, val_data, test_data need to be setup before hand of type `ImageData`'''
def train_custom_cnn_fixed_splits(
        train_data,
        val_data,
        img_size=(128, 128),
        num_conv_layers=3,
        filters=32,
        filter_growth=1.2,
        kernel_size=(3, 3),
        use_batch_norm=False,
        epochs=10,
        batch_size=16,
        learning_rate=1e-4,
        channel=None,
        color_space="RGB",
        use_mask=False,
        mask_radius=None,
        correction_technique="none",
        augmentations=None,
        aug_counter=None
):
    print()
    print(f" - CustomCNN: Training custom CNN with fixed splits")
    print(f"   Train={len(train_data)}, Val={len(val_data)}")
    print(f"   img_size={img_size}, layers={num_conv_layers}, filters={filters}, "
          f"kernel={kernel_size}, batch_norm={use_batch_norm}, epochs={epochs}, "
          f"batch_size={batch_size}")

    # -------------------------------------------------
    # Filter only samples with irradiance
    # -------------------------------------------------
    train_data = [img for img in train_data if img.irradiance is not None]
    val_data   = [img for img in val_data if img.irradiance is not None]

    # -------------------------------------------------
    # Generators (ORDER PRESERVED)
    # -------------------------------------------------
    train_idx = np.arange(len(train_data))
    val_idx   = np.arange(len(val_data))

    train_gen = Misc.gen(
        train_idx,
        batch_size=batch_size,
        data=train_data,
        img_size=img_size,
        channel=channel,
        color_space=color_space,
        use_mask=use_mask,
        radius=mask_radius,
        correction_technique=correction_technique,
        augmentations=augmentations,
        aug_counter=aug_counter
    )
    val_gen = Misc.gen(
        val_idx,
        batch_size=batch_size,
        data=val_data,
        img_size=img_size,
        channel=channel,
        color_space=color_space,
        use_mask=use_mask,
        radius=mask_radius,
        correction_technique=correction_technique,
        augmentations=None
    )

    steps_train = max(1, len(train_idx) // batch_size)
    steps_val   = max(1, len(val_idx) // batch_size)

    # =================================================
    # 🔨 Build CNN
    # =================================================
    model = Sequential()
    model.add(Input(shape=(*img_size, 3), dtype="float16"))

    current_filters = filters

    for i in range(num_conv_layers):
        model.add(Conv2D(
            current_filters,
            kernel_size,
            activation="relu",
            padding="same"
        ))

        if use_batch_norm:
            model.add(BatchNormalization())

        if i % 2 == 1:
            model.add(MaxPooling2D((2, 2), padding="same"))

        # Maxing out the filters at 256 to prevent OOM issues
        current_filters = min(int(current_filters * filter_growth), 256)

    model.add(Flatten())
    model.add(Dense(128, activation="relu", dtype="float32"))
    model.add(Dense(1, activation="linear", dtype="float32"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )

    print(" - CustomCNN: Model compiled, starting training...")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_train,
        validation_steps=steps_val,
        epochs=epochs
    )

    # -------------------------------------------------
    # Prepare test set for evaluation (small, OK in RAM)
    # -------------------------------------------------

    print(" - CustomCNN: Training done.")

    return model, history


'''Original version of train CustomCNN function. Doesn't use fixed splits.
Resulting in random-ish results. 
Used in CustomCNNExperiment. Not in further experiments. Not recommended.'''
def train_custom_cnn(
        image_data_list,
        img_size=(128, 128),
        num_conv_layers=3,
        filters=32,
        filter_growth=1.2,
        kernel_size=(3, 3),
        use_batch_norm=False,
        test_size=0.2,
        epochs=10,
        batch_size=16,
        learning_rate=1e-4
):
    print()
    print(f" - CustomCNN: Training custom CNN on {len(image_data_list)} images...")
    print(f" - CustomCNN: Preparing data (img_size={img_size}, layers={num_conv_layers}, "
          f"filters={filters}, kernel_size={kernel_size}, batch_norm={use_batch_norm})")

    # Filter only images with irradiance
    data = [img for img in image_data_list if img.irradiance is not None]

    # Preprocess images
    ImageEditor.preprocess_images(data, img_size)

    # ---- MEMORY OPTIMIZATION 1: convert to float16 ----
    for img in data:
        img.preprocessed_image = img.preprocessed_image.astype("float16")

    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42, shuffle=True)

    # ---- MEMORY OPTIMIZATION 2: Use generators instead of giant NumPy arrays ----
    train_gen = Misc.gen(train_idx, batch_size=batch_size, data=data)
    test_gen = Misc.gen(test_idx, batch_size=batch_size, data=data)

    steps_train = max(1, len(train_idx) // batch_size)
    steps_test = max(1, len(test_idx) // batch_size)

    # =============================================================
    # 🔨 Build dynamic CNN
    # =============================================================
    model = Sequential()

    input_shape = (*img_size, 3)

    # 🔥 Important for mixed precision
    # Ensures first layer is float16 but keeps model stable
    model.add(Input(shape=input_shape, dtype="float16"))

    current_filters = filters  # starting filters

    for i in range(num_conv_layers):
        model.add(Conv2D(
            current_filters,
            kernel_size,
            activation='relu',
            padding='same'
        ))

        if use_batch_norm:
            model.add(BatchNormalization())

        # Pooling every 2 layers — keeps dimensions safe
        if i % 2 == 1:
            model.add(MaxPooling2D((2, 2), padding='same'))

        # Increase filter count for next conv block
        current_filters = int(current_filters * filter_growth)

    model.add(Flatten())

    # ⚠ Mixed precision requires Dense layers to output float32
    # Otherwise the network becomes unstable.
    model.add(Dense(128, activation='relu', dtype="float32"))
    model.add(Dense(1, activation='linear', dtype="float32"))

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    print(f" - CustomCNN: Model compiled, starting training...")

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        steps_per_epoch=steps_train,
        validation_steps=steps_test,
        epochs=epochs
    )

    # For evaluation function later, we still need a small test set in RAM
    X_test = np.array([data[i].preprocessed_image for i in test_idx], dtype="float16")
    y_test = np.array([data[i].irradiance for i in test_idx], dtype="float32")

    print(f" - CustomCNN: Training done.")

    return model, X_test, y_test, history
