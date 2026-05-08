"""
Determines a recommended batch size based on the image resolution.
Larger images require more GPU memory, so the batch size is reduced.

Args:
    img_size (tuple): Image size as (height, width).

Returns:
    int: Recommended batch size.
"""
def recommended_batch_size(img_size):
    h, w = img_size

    # Use the larger dimension as the determinant
    max_dim = max(h, w)

    # Heuristic mapping (safe defaults for ResNet50)
    if max_dim <= 128:
        return 32       # Small images → large batch possible
    elif max_dim <= 224:
        return 16       # Standard ResNet50 input size → moderate batch
    elif max_dim <= 512:
        return 8        # Large, but manageable
    elif max_dim <= 1024:
        return 1        # Very large images → tiny batch
    else:
        return 1        # Extremely large → safest option


"""
Generator that yields batches of images and labels for training/testing.

Purpose:
    - Avoids storing all image data in large NumPy arrays at once, saving RAM/GPU memory.
    - Dynamically loads batches for the model during training/validation.

Notes:
    - Shuffles indices each epoch to randomize batch order.
    - Uses float16 for images to reduce memory usage.
    - Loops infinitely; compatible with Keras `model.fit` using `steps_per_epoch`.
"""
import numpy as np
from ConnorCode.ImageUtils import ImageEditor

def gen(
        indices_subset,
        batch_size,
        data,
        img_size,
        channel=None,
        color_space="RGB",
        use_mask=False,
        radius=None,
        correction_technique="none",
        augmentations=None,
        aug_counter=None
):
    while True:
        np.random.shuffle(indices_subset)
        for i in range(0, len(indices_subset), batch_size):
            batch_idx = indices_subset[i:i + batch_size]

            batch_imgs = []
            batch_labels = []

            for idx in batch_idx:
                img = data[idx].image

                img = ImageEditor.fisheye_correction_using_technique(img, correction_technique, radius=radius)

                if use_mask and radius is not None:
                    img = ImageEditor.crop_to_fisheye_square(img, radius)

                if augmentations is not None:
                    img = ImageEditor.apply_augmentations(img, augmentations, aug_counter=aug_counter)

                img = ImageEditor.preprocess_image(
                    image=img,
                    img_size=img_size,
                    color_space=color_space,
                    channel=channel
                )

                batch_imgs.append(img)
                batch_labels.append(data[idx].irradiance)

            yield np.array(batch_imgs, dtype="float16"), np.array(batch_labels, dtype="float32")


"""
Debug function to check gpu memory state
"""
import tensorflow as tf
def print_gpu_mem(tag):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if not gpus:
        print(f"{tag}: No GPU")
        return

    info = tf.config.experimental.get_memory_info("GPU:0")
    print(f"{tag}: current={info['current'] / 1e9:.2f}GB, peak={info['peak'] / 1e9:.2f}GB")

# "SPEED AND POWER SOLVE EVERYTHING!", Jeremy Clarkson
