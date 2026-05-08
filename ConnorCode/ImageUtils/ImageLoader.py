from pathlib import Path
from typing import List
import cv2
import re

"""
Class for holding an image along with its metadata:
- image: the original loaded image (as a NumPy array)
- timestamp: timestamp of image creation
- name: filename of the image
- irradiance: measured irradiance value (W/m²), optional
- preprocessed_image: resized and normalized version for NN input
"""
class ImageData:
    def __init__(self, image, timestamp, name, irradiance=None, preprocessed_image=None):
        self.image = image
        self.timestamp = timestamp
        self.name = name
        self.irradiance = irradiance
        self.preprocessed_image = preprocessed_image


"""
    Loads all PNG images from the given folder and extracts timestamp information
    from their filenames. Returns a list of ImageData objects containing the image,
    its name, and its timestamp in format 'YYYYMMDD_HH-MM'.

    Expected filename format:
        - includes date in expected format YYYYMMDD_HH-MM
        - no other part of the filename has that format (_8chars_2chars-2chars-)
        - files needs to be *.png
"""
def load_images_with_timestamps(folder_path: str, max_images: int = None) -> List[ImageData]:
    print()
    print(f" - IMAGE_LOADER: Loading images from {folder_path}...")

    folder = Path(folder_path)
    image_data_list = []

    for i, img_path in enumerate(sorted(folder.glob("*.png"))):
        if max_images is not None and i >= max_images:
            break  # Stop after reaching the limit

        filename = img_path.name
        match = re.search(r"_(\d{8})_(\d{2}-\d{2})-", filename)
        if match:
            timestamp = f"{match.group(1)}_{match.group(2)}"
        else:
            timestamp = "unknown"

        image = cv2.imread(str(img_path))  # Load image
        if image is None:
            print(f"⚠️ Could not read image: {filename}")
            continue

        image_data_list.append(ImageData(name=filename, timestamp=timestamp, image=image))

    print(f" - IMAGE_LOADER: Successfully loaded {len(image_data_list)} images.")

    return image_data_list
