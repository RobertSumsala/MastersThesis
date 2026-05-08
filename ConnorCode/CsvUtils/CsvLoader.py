import csv
from dataclasses import dataclass, asdict
from typing import Optional

"""
Holds CNN training parameters. Missing values default to None.
"""
@dataclass
class CNNParams:
    num_layers: Optional[int] = None
    num_epochs: Optional[int] = None
    batch_norm: Optional[bool] = None
    kernel_size: Optional[int] = None
    base_filters: Optional[int] = None
    img_size: Optional[int] = None
    learning_rate: Optional[float] = None

from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class CNNParams:
    """
    Holds CNN training parameters. All fields are required.
    """
    num_layers: int
    num_epochs: int
    batch_norm: bool
    kernel_size: int
    base_filters: int
    img_size: int
    learning_rate: float

"""
Reads a parameter txt file and returns a CNNParams dataclass instance.
Raises an error if any expected parameter is missing.
"""
def load_model_params(file_path: str) -> CNNParams:
    print()
    print(f" - CSV_LOADER: Loading best CNN params from {file_path}...")

    # Expected keys with types
    expected_types = {
        "num_layers": int,
        "num_epochs": int,
        "batch_norm": bool,
        "kernel_size": int,
        "base_filters": int,
        "img_size": int,
        "learning_rate": float
    }

    loaded_values = {}

    try:
        with open(file_path, 'r') as f:
            for line in f:
                if ':' not in line:
                    continue
                key, value = line.strip().split(':', 1)
                key, value = key.strip(), value.strip()

                if key not in expected_types:
                    continue  # ignore unexpected keys

                # Convert to proper type
                if expected_types[key] == int:
                    loaded_values[key] = int(value)
                elif expected_types[key] == float:
                    loaded_values[key] = float(value)
                elif expected_types[key] == bool:
                    loaded_values[key] = value.lower() == "true"

        # Check that all expected keys were found
        missing_keys = [k for k in expected_types if k not in loaded_values]
        if missing_keys:
            raise ValueError(f"Missing parameters in file: {', '.join(missing_keys)}")

        # Create dataclass instance
        params = CNNParams(**loaded_values)

    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found.")

    # Test print: show the dataclass
    print("Loaded CNN parameters:", params)

    return params

"""
Loads irradiance data from a CSV file and assigns it to matching ImageData objects.

Args:
    csv_path (str): Path to the CSV file containing irradiance data.
    image_data_list (list[ImageData]): List of ImageData objects whose 'name'
                                       matches the 'PictureName' field in the CSV.
"""
def load_irradiance_from_csv(csv_path, image_data_list):
    print()
    print(f" - CSV_LOADER: Loading irradiance data from {csv_path}...")

    # Read the CSV into a dictionary: {picture_name: irradiance_value}
    irradiance_map = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            picture_name = row['PictureName'].strip()
            irradiance = float(row['Irradiance'])
            irradiance_map[picture_name] = irradiance

    # Match each ImageData by name and assign irradiance
    matched, unmatched = 0, 0
    for img_data in image_data_list:
        if img_data.name in irradiance_map:
            img_data.irradiance = irradiance_map[img_data.name]
            matched += 1
        else:
            unmatched += 1

    print(f" - CSV_LOADER: Irradiance data added for {matched} images. {unmatched} unmatched.")
