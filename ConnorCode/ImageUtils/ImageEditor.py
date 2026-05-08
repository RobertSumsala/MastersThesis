import cv2
import numpy as np
from typing import List
from ConnorCode.ImageUtils.ImageLoader import ImageData


"""
Preprocesses a single image:
1. Resize
2. Convert to selected color space
3. Normalize to 0-1 range
4. Optionally keep only selected channel
"""
def preprocess_image(image, img_size, color_space="RGB", channel=None):
    if image is None:
        raise ValueError("Input image is None")

    # 1️⃣ Resize (still uint8)
    resized = cv2.resize(image, img_size)

    # 2️⃣ Convert color space (expects uint8 0-255)
    converted = convert_color_space(resized, color_space)

    # 3️⃣ Normalize AFTER conversion
    normalized = converted.astype("float32") / 255.0

    # 4️⃣ Keep only selected channel
    if channel is not None:
        normalized = keep_only_channel(normalized, channel, color_space)

    return normalized

#----------------------------------- AUGMENTATIONS -------------------------------------------------------------------------------------------------

def adjust_aug_counter(augmentation, aug_counter):
    if aug_counter is not None:
        aug_counter[augmentation] = aug_counter.get(augmentation, 0) + 1

"""
image: np.array
augmentations with weights: dict like {"rotate": 0.6, "noise": 0.2}, 
weights represent the probability of augmentation being applied,
prob: probability to apply augmentation
"""
import random
def apply_augmentations(image, augmentations, prob=0.15, aug_counter=None):
    if not augmentations or "none" in augmentations:
        adjust_aug_counter("none", aug_counter)
        return image

    # Step 1: Decide if we augment this image
    if random.random() > prob:
        adjust_aug_counter("none", aug_counter)
        return image

    augmented = image.copy()

    # Step 2: Select ONE augmentation based on weights
    aug_names = list(augmentations.keys())
    aug_weights = list(augmentations.values())

    total = sum(aug_weights)

    # if no weights were provided, pick one augmentation randomly
    if total == 0:
        chosen_aug = random.choice(aug_names)
    else:
        normalized_weights = [w / total for w in aug_weights]
        chosen_aug = random.choices(aug_names, weights=normalized_weights, k=1)[0]

    # Step 3: Apply the chosen augmentation
    if chosen_aug == "rotate":
        augmented = rotate_image(augmented)

    elif chosen_aug == "translate":
        augmented = translate_image(augmented)

    elif chosen_aug == "scale":
        augmented = scale_image(augmented)

    elif chosen_aug == "brightness":
        augmented = brightness_adjustment(augmented)

    elif chosen_aug == "clahe":
        augmented = apply_clahe(augmented)

    elif chosen_aug == "noise":
        augmented = add_gaussian_noise(augmented)

    elif chosen_aug == "blur":
        augmented = apply_gaussian_blur(augmented)

    # Step 4: Count it
    adjust_aug_counter(chosen_aug, aug_counter)

    return augmented


def rotate_image(image):
    angle = random.uniform(-15, 15)  # 10–15% approx
    h, w = image.shape[:2]

    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return rotated

def translate_image(image, max_shift_ratio=0.1):
    h, w = image.shape[:2]

    # Compute max pixel shifts
    max_dx = int(w * max_shift_ratio)
    max_dy = int(h * max_shift_ratio)

    # Random shift values
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)

    # Translation matrix
    matrix = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    shifted = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return shifted

def scale_image(image, scale_range=(0.9, 1.1)):
    h, w = image.shape[:2]

    # Random scale factor
    scale = random.uniform(scale_range[0], scale_range[1])

    # Center for scaling
    center = (w // 2, h // 2)

    # Get transformation matrix
    matrix = cv2.getRotationMatrix2D(center, 0, scale)

    scaled = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return scaled

def brightness_adjustment(image, min_factor=0.7, max_factor=1.3):
    factor = random.uniform(min_factor, max_factor)

    # Convert to float to avoid overflow/underflow
    bright = image.astype(np.float32) * factor

    # Clip values to valid range and convert back
    bright = np.clip(bright, 0, 255).astype(np.uint8)

    return bright

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    y_clahe = clahe.apply(y)

    ycrcb_clahe = cv2.merge((y_clahe, cr, cb))
    result = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)

    return result


def add_gaussian_noise(image, mean=0, std=10):
    # Convert to float to avoid overflow
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)

    noisy = image.astype(np.float32) + noise

    # Clip to valid range and convert back
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy

def apply_gaussian_blur(image, kernel_size=15):
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return blurred

#----------------------------------- FISH-EYE CORRECTIONS -------------------------------------------------------------------------------------------------

"""
Convert fisheye image to rectilinear projection using OpenCV.

Parameters
----------
img : np.ndarray
    Input fisheye image.
K : np.ndarray
    Camera intrinsic matrix (3x3). If None, an approximate one is generated.
D : np.ndarray
    Distortion coefficients (4x1). If None, approximate values are used.
balance : float
    0 = crop edges (less black), 1 = keep full FOV (more black).
scale : float
    Output scaling factor to preserve more pixels.

Returns
-------
undistorted : np.ndarray
    Rectilinear projection image.
"""
def _fisheye_to_rectilinear(img, K=None, D=None, balance=0.0, scale=1.5, **kwargs):
    h, w = img.shape[:2]

    # If no calibration provided, create approximate camera parameters
    if K is None:
        f = w  # approximate focal length
        K = np.array([
            [f, 0, w/2],
            [0, f, h/2],
            [0, 0, 1]
        ])

    if D is None:
        # approximate fisheye distortion
        D = np.array([-0.6, 0.2, 0.0, 0.0])

    # Scale output size
    new_size = (int(w * scale), int(h * scale))

    # Adjust camera matrix
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance, new_size=new_size
    )

    # Generate undistortion maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, new_size, cv2.CV_16SC2
    )

    # Remap image
    undistorted = cv2.remap(
        img, map1, map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    return undistorted


"""
Convert fisheye sky image into equal angular sector representation
using polar transformation.
"""
def _fisheye_equal_angular(img, radius=None):
    h, w = img.shape[:2]

    center = (w // 2, h // 2)
    size = max(h, w)

    polar = cv2.warpPolar(
        img,
        (size, size),            # width = azimuth resolution
        center,
        radius,
        cv2.WARP_POLAR_LINEAR
    )

    # Rotate so zenith is at top
    polar = cv2.rotate(polar, cv2.ROTATE_90_CLOCKWISE)

    return polar

"""
Applies Lambert azimuthal equal-area projection to a fisheye sky image.

This transformation remaps the circular fisheye image (hemispherical sky dome)
into a square image while preserving equal solid angles. As a result, each pixel
in the output image represents an approximately equal portion of the sky,
which is particularly important for tasks such as solar irradiance estimation.

Compared to other projections (e.g., polar or rectilinear), this method reduces
over-representation of the horizon and provides a more physically meaningful
distribution of sky regions.
"""
def _fisheye_equal_area(img, radius=None):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    if radius is None:
        radius = min(cx, cy)

    size = 2 * radius

    # Create grid of output coordinates
    y, x = np.indices((size, size))
    nx = (x - radius) / radius
    ny = (y - radius) / radius

    r = np.sqrt(nx**2 + ny**2)

    # Mask for valid circle
    mask = r <= 1

    # Lambert projection
    theta = 2 * np.arcsin(np.clip(r / 2, 0, 1))
    phi = np.arctan2(ny, nx)

    # Map to fisheye radius
    fisheye_r = theta / (np.pi / 2) * radius

    # Source coordinates
    src_x = cx + fisheye_r * np.cos(phi)
    src_y = cy + fisheye_r * np.sin(phi)

    src_x = np.clip(src_x, 0, w - 1).astype(np.int32)
    src_y = np.clip(src_y, 0, h - 1).astype(np.int32)

    # Create output
    output = np.zeros((size, size, 3), dtype=img.dtype)

    # Apply mapping only inside circle
    output[mask] = img[src_y[mask], src_x[mask]]

    return output

"""No correction (control experiment)."""
def _fisheye_none(img, **kwargs):
    return img

def fisheye_correction_using_technique(img, technique="none", **kwargs):
    techniques = {
        "none": _fisheye_none,
        "rectilinear_openCV": _fisheye_to_rectilinear,
        "angular_sectors": _fisheye_equal_angular,
        "equal_area": _fisheye_equal_area
    }

    if technique not in techniques:
        raise ValueError(
            f"Unknown technique '{technique}'. "
            f"Available: {list(techniques.keys())}"
        )

    return techniques[technique](img, **kwargs)

#------------------------------------------------------------------------------------------------------------------------------------

"""
Scans dataset images and returns the first successfully detected fisheye radius.
If detection fails for all checked images, returns fallback_radius.
Note: 530 was determined for the Alpanach dataset, but should never be used anyways,
since at least 1 in 100 images should have measurable radius
"""
def get_initial_fisheye_radius(data, fallback_radius=530, max_checks=50):
    for i in range(min(len(data), max_checks)):
        img = data[i].image

        try:
            r = estimate_fisheye_radius(img)

            if r is not None and r > 0:
                print(f"\n - IMAGE_EDITOR: Detected fisheye radius: {r}")
                return r

        except Exception:
            continue

    print(f"\n - IMAGE_EDITOR: No radius detected. Using fallback radius: {fallback_radius}")
    return fallback_radius

"""
Estimates fisheye circle radius based on non-black pixels.
Assumes outside region is black.

Note: default radius is provided in case we are unable to measure the non-black are,
if the image is too dark.
"""
def estimate_fisheye_radius(image):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Detect non-black pixels
    non_black = np.any(image > 5, axis=2)  # threshold 5 to avoid noise
    Y, X = np.where(non_black)

    if len(X) == 0:
        print("\n - IMAGE_EDITOR: Warning: could not detect fisheye radius.")
        return None

    distances = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    radius = int(np.max(distances))

    return radius

"""
Used to get rid off the black extra space around fisheye images.
We use mean pixels instead of black, since that should influence CNN less
"""
def mask_the_fisheye_circle(image, radius):
    # First get rid of the extra black above, below and next to the circle
    cropped_image = crop_to_fisheye_square(image, radius)

    h, w = cropped_image.shape[:2]
    center = (w // 2, h // 2)

    Y, X = np.ogrid[:h, :w]
    dist_from_center = (X - center[0])**2 + (Y - center[1])**2
    mask = dist_from_center <= radius**2

    masked = cropped_image.copy()
    mean_pixel = np.mean(cropped_image, axis=(0,1))
    masked[~mask] = mean_pixel

    return masked


"""
Crops the image to a square tightly containing the fisheye circle.
Removes thin outer borders but keeps corner black triangles.

Parameters:
    image: np.ndarray (H, W, 3)
    radius: optional precomputed radius

Returns:
    Cropped square image
"""
def crop_to_fisheye_square(image, radius):
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2

    x1 = max(cx - radius, 0)
    x2 = min(cx + radius, w)
    y1 = max(cy - radius, 0)
    y2 = min(cy + radius, h)

    cropped = image[y1:y2, x1:x2]

    return cropped

"""
Preprocesses images stored in a list of ImageData objects.

- Resizes each image to the target input size.
- Normalizes pixel values to [0, 1].
- Saves the preprocessed version in ImageData.preprocessed_image.

Args:
    image_data_list (list[ImageData]): List of ImageData objects.
    img_size (tuple): Target (width, height) for resizing.

Returns:
    None — modifies objects in place.
"""
def preprocess_images(image_data_list, img_size=(224, 224)):
    print()
    print(f" - IMAGE_EDITOR: Preprocessing {len(image_data_list)} images, with img_size {img_size}...")

    count = 0
    for data in image_data_list:
        if data.image is None:
            raise ValueError(f"Missing image for {data.name}")
        resized = cv2.resize(data.image, img_size)
        normalized = resized / 255.0
        data.preprocessed_image = normalized
        count += 1

    print(f" - - - - -IMAGE_EDITOR: {count} number of images preprocessed.")


"""
Converts an RGB image to the specified color space.
Input image must be RGB.
Returns image in the target color space.
"""
def convert_color_space(image, color_space):
    if color_space == "RGB":
        return image

    elif color_space == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    elif color_space == "YCrCb":
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    elif color_space == "CMYK":
        # Manual RGB → CMYK conversion
        img = image.astype(np.float32) / 255.0

        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        K = 1 - np.maximum.reduce([R, G, B])
        C = (1 - R - K) / (1 - K + 1e-8)
        M = (1 - G - K) / (1 - K + 1e-8)
        Y = (1 - B - K) / (1 - K + 1e-8)

        cmyk = np.stack([C, M, Y, K], axis=-1)
        return cmyk

    else:
        raise ValueError(f"Unsupported color space: {color_space}")


"""
Keeps only the selected channel in the given color space
by zeroing out the remaining channels.
Preserves original shape (H, W, C).
"""
def keep_only_channel(image, channel, color_space):
    CHANNEL_INDEX = {
        "RGB": {"R": 0, "G": 1, "B": 2},
        "HSV": {"H": 0, "S": 1, "V": 2},
        "YCrCb": {"Y": 0, "Cr": 1, "Cb": 2},
        "CMYK": {"C": 0, "M": 1, "Y": 2, "K": 3}
    }

    if color_space not in CHANNEL_INDEX:
        raise ValueError(f"Unsupported color space: {color_space}")

    if channel not in CHANNEL_INDEX[color_space]:
        raise ValueError(f"Invalid channel {channel} for {color_space}")

    img = image.copy()
    channel_idx = CHANNEL_INDEX[color_space][channel]

    for i in range(img.shape[-1]):
        if i != channel_idx:
            img[:, :, i] = 0

    return img

"""
Applies a circular mask to remove outer parts of a fisheye image.
keep_ratio=0.9 keeps 90% of the fisheye radius.
"""
def apply_circular_crop(image, keep_ratio=0.9):
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    radius = int(min(cx, cy) * keep_ratio)

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)

    mask = dist_from_center <= radius

    cropped = image.copy()
    cropped[~mask] = 0  # black outside circle

    return cropped


"""
Applies an equal-area (equisolid-angle) fisheye correction to all images
in image_data_list. The corrected image overwrites image_data.image.

This correction ensures that sky regions near the image edges
have equal pixel importance as regions near the center.
"""
def apply_equisolid_fisheye_correction(
        image_data_list: List[ImageData],
        output_size=None,
        keep_ratio_after_cropping=1.0
):
    print()
    print(f" - IMAGE_EDITOR: Applying equisolid-angle fisheye correction...")

    for img_data in image_data_list:
        img = img_data.image

        # Remove outer edges, in case of objects around edges that are not sky
        img = apply_circular_crop(img, keep_ratio=keep_ratio_after_cropping)

        h, w = img.shape[:2]

        # Output image size (default = input size)
        if output_size is None:
            out_h, out_w = h, w
        else:
            out_h, out_w = output_size

        cx, cy = w / 2, h / 2
        max_radius = min(cx, cy)

        # Create coordinate grid for output image
        y, x = np.indices((out_h, out_w), dtype=np.float32)
        x = x - out_w / 2
        y = y - out_h / 2

        r = np.sqrt(x ** 2 + y ** 2)
        r_norm = r / max_radius

        # Limit to valid hemisphere
        mask = r_norm <= 1.0

        # --- Equisolid-angle mapping ---
        # θ from normalized radius
        # light correction
        # theta = r_norm * (np.pi / 2)
        # Strong (edges compressed and center expanded)
        # theta = (r_norm ** 1.8) * (np.pi / 2)

        # Strong (edges expanded and center compressed)
        theta = (r_norm ** 0.5) * (np.pi / 2)

        # Map back to fisheye radius
        r_fisheye = 2 * max_radius * np.sin(theta / 2)

        # Convert back to pixel coordinates
        map_x = cx + (r_fisheye * x / (r + 1e-8))
        map_y = cy + (r_fisheye * y / (r + 1e-8))

        map_x[~mask] = -1
        map_y[~mask] = -1

        corrected = cv2.remap(
            img,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        img_data.image = corrected

    print(f" - IMAGE_EDITOR: Correction completed.")
