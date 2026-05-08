from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from ConnorCode.ImageUtils import ImageEditor
from ConnorCode.CommonUtils import Misc


"""
    Trains a ResNet50 model to predict irradiance (W/m²) from sky images.

    Args:
        image_data_list (list[ImageData]): list of ImageData objects with irradiance filled.
        img_size (tuple): image resize dimensions (default 224x224).
        test_size (float): fraction of data for testing.
        epochs (int): training epochs.
        batch_size (int): batch size.

    Returns:
        model (tf.keras.Model): trained model.
"""
def train_resnet50_regression(image_data_list, img_size=(224, 224), test_size=0.2, epochs=10, batch_size=16):
    print()
    print(f" - RESNET50: Training ResNet50 regression model for {len(image_data_list)} images...")
    print(f" - RESNET50: Preparing training data for ResNet50 regression model...")

    # Filter only images that have irradiance data
    data = [item for item in image_data_list if item.irradiance is not None]

    # Preprocess images
    ImageEditor.preprocess_images(data, img_size)

    # ---- MEMORY OPTIMIZATION 1: convert to float16 ----
    for img in data:
        img.preprocessed_image = img.preprocessed_image.astype("float16")

    # Create index list for train/test splitting
    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)

    # ---- MEMORY OPTIMIZATION 2: Use generators instead of giant NumPy arrays ----
    train_gen = Misc.gen(train_idx, batch_size=batch_size, data=data)
    test_gen  = Misc.gen(test_idx, batch_size=batch_size, data=data)

    steps_train = max(1, len(train_idx) // batch_size)
    steps_test  = max(1, len(test_idx)  // batch_size)

    # Load pretrained ResNet50 (without top)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*img_size, 3))

    # Freeze first two blocks
    for layer in base_model.layers[:50]:
        layer.trainable = False

    # Add regression head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(1, activation='linear')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])

    print()
    print(f" - RESNET50: Training model with generator (optimized RAM)...")

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        steps_per_epoch=steps_train,
        validation_steps=steps_test,
        epochs=epochs,
    )

    # For evaluation function later, we still need a small test set in RAM
    X_test = np.array([data[i].preprocessed_image for i in test_idx], dtype="float16")
    y_test = np.array([data[i].irradiance for i in test_idx], dtype="float32")

    return model, X_test, y_test, history
