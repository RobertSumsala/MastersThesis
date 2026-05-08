import os
import cv2
import gc
import tensorflow as tf
from ConnorCode.ImageUtils import ImageLoader
from ConnorCode.CsvUtils import CsvLoader
from ConnorCode.CommonUtils import Misc, ModelEvaluator
from ConnorCode.CustomCNN import CustomCNNTrainer
from ConnorCode.TextUtils import TextWriter
from ConnorCode.ImageUtils import ImageEditor


# =============================================================
# 📁 Results paths
# =============================================================
RESULTS_DIR = "../results/fisheyeCorrectionPart2"

RESULTS_TXT_PATH = os.path.join(
    RESULTS_DIR,
    "fisheye_correction_part2_results.txt"
)

NUMBER_OF_TEST_RUNS = 3

def save_test_image_with_tested_correction(test_image, technique, radius):
    print(f"\nTesting {technique} correction on test image...")
    print("\nSaving original test image...")
    cv2.imwrite(os.path.join(RESULTS_DIR, f"original_image_example.png"), test_image)

    print("\nCorrecting image...")
    corrected_image = ImageEditor.fisheye_correction_using_technique(test_image, technique, radius=radius)

    print("\nSaving corrected image...")
    cv2.imwrite(os.path.join(RESULTS_DIR, f"corrected_image_example_using_{technique}.png"), corrected_image)


def test_fisheye_correction_technique(technique):
    print()
    print(f"Testing the {technique} fisheye correction technique")

    # Fixed splits for train, validation, test data
    print("\nSetting up training data...")
    train_image_folder_path = "../solar_dataset_v2/train/images"
    train_image_data_list = ImageLoader.load_images_with_timestamps(train_image_folder_path)

    train_csv_path = "../solar_dataset_v2/train/meteo_data_cleaned.csv"
    CsvLoader.load_irradiance_from_csv(train_csv_path, train_image_data_list)

    print("\nSetting up validation data...")
    val_image_folder_path = "../solar_dataset_v2/val/images"
    val_image_data_list = ImageLoader.load_images_with_timestamps(val_image_folder_path)

    val_csv_path = "../solar_dataset_v2/val/meteo_data_cleaned.csv"
    CsvLoader.load_irradiance_from_csv(val_csv_path, val_image_data_list)

    print("\nSetting up test data...")
    test_image_folder_path = "../solar_dataset_v2/test/images"
    test_image_data_list = ImageLoader.load_images_with_timestamps(test_image_folder_path)

    test_csv_path = "../solar_dataset_v2/test/meteo_data_cleaned.csv"
    CsvLoader.load_irradiance_from_csv(test_csv_path, test_image_data_list)

    print("\nRetrieving best parameters for our model...")
    best_params_path = "../BestParameters/best_parameters_from_optuna.txt"
    best_params = CsvLoader.load_model_params(best_params_path)

    batch_size = Misc.recommended_batch_size((best_params.img_size, best_params.img_size))
    COLOR_SPACE = "HSV"

    DEFAULT_MASK_RADIUS = ImageEditor.get_initial_fisheye_radius(train_image_data_list)

    # ---------------------------------------------------------------------
    # PREP RESULT FILE (overwrite old)
    # ---------------------------------------------------------------------
    os.makedirs(os.path.dirname(RESULTS_TXT_PATH), exist_ok=True)

    # Save a test image that showcases the correction technique
    save_test_image_with_tested_correction(test_image=test_image_data_list[8].image, technique=technique, radius=DEFAULT_MASK_RADIUS)

    with open(RESULTS_TXT_PATH, "w") as f:
        for i in range(NUMBER_OF_TEST_RUNS):
            print(f"Running test {i + 1}")

            # ---------------------------------------------------------------------
            # Train customCNN using HSV ColourSpace for reference - no cropping/masking
            # ---------------------------------------------------------------------
            print("\n" + "=" * 60)
            print(f"Training using regular images for reference...")
            print("=" * 60)

            model, history = CustomCNNTrainer.train_custom_cnn_fixed_splits(
                train_data=train_image_data_list,
                val_data=val_image_data_list,
                img_size=(best_params.img_size, best_params.img_size),
                num_conv_layers=best_params.num_layers,
                filters=best_params.base_filters,
                kernel_size=(best_params.kernel_size, best_params.kernel_size),
                use_batch_norm=best_params.batch_norm,
                epochs=best_params.num_epochs,
                batch_size=batch_size,
                learning_rate=best_params.learning_rate,
                channel=None,
                color_space=COLOR_SPACE,
                use_mask=True,
                mask_radius=DEFAULT_MASK_RADIUS,
                correction_technique="none"
            )

            result_no_masking = ModelEvaluator.evaluate_model_with_generator(
                model=model,
                history=history,
                test_data=test_image_data_list,
                img_size=(best_params.img_size, best_params.img_size),
                batch_size=batch_size,
                channel=None,
                color_space=COLOR_SPACE,
                use_mask=True,
                mask_radius=DEFAULT_MASK_RADIUS,
                correction_technique="none"
            )

            TextWriter.write_line(f, f"\n--- RUN: {i+1} ---\n")
            TextWriter.write_line(f, "\n--- REFERENCE TRAINING OF CUSTOM CNN ---\n")
            TextWriter.write_line(f, "\n--- Correction: None ---\n")
            TextWriter.write_result_block(
                f,
                setting_name="No correction used on images: ",
                setting_value=True,
                result=result_no_masking,
                is_best=False
            )

            # ---------------------------------------------------------------------
            # Train customCNN with corrected images
            # ---------------------------------------------------------------------
            print("\n" + "=" * 60)
            print(f"Training using corrected images...")
            print("=" * 60)

            tf.keras.backend.clear_session()
            gc.collect()

            model, history = CustomCNNTrainer.train_custom_cnn_fixed_splits(
                train_data=train_image_data_list,
                val_data=val_image_data_list,
                img_size=(best_params.img_size, best_params.img_size),
                num_conv_layers=best_params.num_layers,
                filters=best_params.base_filters,
                kernel_size=(best_params.kernel_size, best_params.kernel_size),
                use_batch_norm=best_params.batch_norm,
                epochs=best_params.num_epochs,
                batch_size=batch_size,
                learning_rate=best_params.learning_rate,
                channel=None,
                color_space=COLOR_SPACE,
                use_mask=True,
                mask_radius=DEFAULT_MASK_RADIUS,
                correction_technique=technique
            )

            result_with_masking = ModelEvaluator.evaluate_model_with_generator(
                model=model,
                history=history,
                test_data=test_image_data_list,
                img_size=(best_params.img_size, best_params.img_size),
                batch_size=batch_size,
                channel=None,
                color_space=COLOR_SPACE,
                use_mask=True,
                mask_radius=DEFAULT_MASK_RADIUS,
                correction_technique=technique
            )

            TextWriter.write_line(f, "\n--- CUSTOM CNN USING CORRECTED IMAGES ---\n")
            TextWriter.write_result_block(
                f,
                setting_name="Correction used: ",
                setting_value=technique,
                result=result_with_masking,
                is_best=False
            )

            tf.keras.backend.clear_session()
            gc.collect()

        TextWriter.write_line(f, "\n===== EXPERIMENT FINISHED =====")

    print("\nExperiment completed.")
