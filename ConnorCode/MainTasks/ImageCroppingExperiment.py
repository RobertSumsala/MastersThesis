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
RESULTS_DIR = "../results/imageCropping"

RESULTS_TXT_PATH = os.path.join(
    RESULTS_DIR,
    "image_cropping_analysis_results.txt"
)

def image_crop_test(test_image, img_size, mask_radius):
    print("\nTesting image cropping...")
    print("\nSaving original test image...")
    cv2.imwrite(os.path.join(RESULTS_DIR, f"original_image_example.png"), test_image)

    print("\nCropping image...")
    cropped_image = ImageEditor.crop_to_fisheye_square(test_image, mask_radius)

    print("\nSaving cropped image...")
    cv2.imwrite(os.path.join(RESULTS_DIR, f"cropped_image_example.png"), cropped_image)

    print("\nMasking the image...")
    masked_image = ImageEditor.mask_the_fisheye_circle(test_image, mask_radius)

    print("\nSaving masked image...")
    cv2.imwrite(os.path.join(RESULTS_DIR, f"masked_image_example.png"), masked_image)

    print("\nPreprocessing test image...")
    preprocessed_image = ImageEditor.preprocess_image(
        image=masked_image,
        img_size=(img_size, img_size),
        color_space="RGB",
        channel=None
    )

    print("\nSaving preprocessed test image...")
    preprocessed_image_for_saving = (preprocessed_image * 255).clip(0, 255).astype("uint8")
    cv2.imwrite(os.path.join(RESULTS_DIR, f"preprocessed_image_example.png"), preprocessed_image_for_saving)


def mask_images_and_train_custom_cnn():
    print()
    print(f"Analyzing the influence of masked images on the CNN raining results...")

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

    # Test image cropping functions
    image_crop_test(train_image_data_list[8].image, best_params.img_size, DEFAULT_MASK_RADIUS)

    with open(RESULTS_TXT_PATH, "w") as f:
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
            use_mask=False
        )

        result_no_masking = ModelEvaluator.evaluate_model_with_generator(
            model=model,
            history=history,
            test_data=test_image_data_list,
            img_size=(best_params.img_size, best_params.img_size),
            batch_size=batch_size,
            channel=None,
            color_space=COLOR_SPACE,
            use_mask = False
        )

        TextWriter.write_line(f, "\n--- REFERENCE TRAINING OF CUSTOM CNN USING HSV COLOUR SPACE ---\n")
        TextWriter.write_line(f, "\n--- Image Mask: False ---\n")
        TextWriter.write_result_block(
            f,
            setting_name="No mask used on images: ",
            setting_value=True,
            result=result_no_masking,
            is_best=False
        )

        # ---------------------------------------------------------------------
        # Train customCNN with masked images
        # ---------------------------------------------------------------------
        print("\n" + "=" * 60)
        print(f"Training using masked images...")
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
            mask_radius=DEFAULT_MASK_RADIUS
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
            mask_radius=DEFAULT_MASK_RADIUS
        )

        TextWriter.write_line(f, "\n--- CUSTOM CNN WITH HSV COLOUR SPACE USING MASKED IMAGES ---\n")
        TextWriter.write_line(f, "\n--- Image Mask: True ---\n")
        TextWriter.write_result_block(
            f,
            setting_name="Masked images used: ",
            setting_value=True,
            result=result_with_masking,
            is_best=False
        )

        TextWriter.write_line(f, "\n===== EXPERIMENT FINISHED =====")


    print("\nAnalysis completed.")

