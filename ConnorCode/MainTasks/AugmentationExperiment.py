import os
import cv2
from ConnorCode.ImageUtils import ImageLoader
from ConnorCode.CsvUtils import CsvLoader
from ConnorCode.CommonUtils import Misc, ModelEvaluator
from ConnorCode.CustomCNN import CustomCNNTrainer
from ConnorCode.TextUtils import TextWriter
from ConnorCode.ImageUtils import ImageEditor


# =============================================================
# 📁 Results paths
# =============================================================
RESULTS_DIR = "../results/augmentationExperiment"

RESULTS_TXT_PATH = os.path.join(
    RESULTS_DIR,
    "augmentation_experiment_results.txt"
)

NUMBER_OF_TEST_RUNS = 6

def save_test_image_with_tested_augmentation(test_image, augmentation):
    print(f"\nTesting {augmentation} augmentation on test image...")
    print("\nSaving original test image...")
    cv2.imwrite(os.path.join(RESULTS_DIR, f"original_image_example.png"), test_image)

    print("\nCorrecting image...")
    # prob parameter is used for generators, to make sure not all images are augmented,
    # here we want the augmentation to happen, hence set to 1.0
    # the augmentation has to be passed as one member array,
    # as the function expects array, and we want to test only one augmentation here
    augmented_image = ImageEditor.apply_augmentations(test_image, {augmentation: 1.0}, prob=1.0)

    print("\nSaving corrected image...")
    cv2.imwrite(os.path.join(RESULTS_DIR, f"augmented_image_example_using_{augmentation}.png"), augmented_image)


def test_augmentations(augmentations):
    print()
    print(f"Testing these augmentations: {augmentations}...")

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

    if augmentations is not None:
        for augmentation in augmentations:
            save_test_image_with_tested_augmentation(test_image_data_list[8].image, augmentation)

    with open(RESULTS_TXT_PATH, "w") as f:
        TextWriter.write_line(f, f"NOTE: "
                                 f"\nWhen looking at the augmentation statistics bare in mind:"
                                 f"\nDuring training it goes through the train dataset multiple times, "
                                 f"\nthat's why the number of all used augmentations is far greater than the number"
                                 f"\nof images in train dataset. What is important to check is the ratio between "
                                 f"\naugmented none-augmented images.\n")

        for i in range(NUMBER_OF_TEST_RUNS):
            print(f"Running test {i + 1}")

            # ---------------------------------------------------------------------
            # Train customCNN with the best settings with picked augmentations
            # ---------------------------------------------------------------------
            print("\n" + "=" * 60)
            print(f"Training using selected augmentations: {augmentations}")
            print("=" * 60)

            # Reset/init augmentation counter
            aug_counter = {aug_name: 0 for aug_name in augmentations} if augmentations else {}

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
                correction_technique="none",
                augmentations=augmentations,
                aug_counter=aug_counter
            )

            result = ModelEvaluator.evaluate_model_with_generator(
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
            TextWriter.write_line(f, "\n--- TRAINING OF CUSTOM CNN ---\n")
            TextWriter.write_result_block(
                f,
                setting_name="Augmentations",
                setting_value=augmentations,
                result=result,
                is_best=False
            )
            TextWriter.write_line(f, f"Augmentation count (train data only): {aug_counter}")
            TextWriter.write_line(f, f"Number of train images: {len(train_image_data_list)}")
            TextWriter.write_line(f, "-" * 60)

        TextWriter.write_line(f, "\n===== EXPERIMENT FINISHED =====")

    print("\nExperiment completed.")
