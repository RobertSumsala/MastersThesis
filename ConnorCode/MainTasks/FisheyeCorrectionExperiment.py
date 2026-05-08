import os
import cv2
from ConnorCode.ImageUtils import ImageLoader, GraphMaker, ImageEditor
from ConnorCode.CsvUtils import CsvLoader
from ConnorCode.TextUtils import TextWriter
from ConnorCode.CustomCNN import CustomCNNTrainer
from ConnorCode.CommonUtils import ModelEvaluator
from ConnorCode.CommonUtils import Misc


# =============================================================
# 📁 Results paths
# =============================================================
RESULTS_DIR = "../results/imageCorrectionExperiment"

RESULTS_TXT_PATH = os.path.join(
    RESULTS_DIR,
    "fisheye_correction_experiment.txt"
)

RESULTS_CURVE_PATH = os.path.join(
    RESULTS_DIR,
    "fisheye_correction_learning_curves.png"
)


def compare_original_and_corrected_images():
    print()
    print(f"Running experiment to find out if corrected images without fisheye effect improve results...")

    image_folder_path = "../Solar_data/joined_orig_data/images"
    image_data_list = ImageLoader.load_images_with_timestamps(image_folder_path)

    csv_path = "../Solar_data/joined_orig_data/out_data_joined.csv"
    CsvLoader.load_irradiance_from_csv(csv_path, image_data_list)

    print()
    print(f"Test print:")
    print(f"First image name: {image_data_list[0].name}")
    print(f"First image timestamp: {image_data_list[0].timestamp}")
    print(f"First image irradiance: {image_data_list[0].irradiance}")

    # ---------------------------------------------------------------------
    # EXPERIMENT SETTINGS
    # ---------------------------------------------------------------------
    # Values based on previous experiment `CustomCNNExperiment`
    layers = 6
    batchnorm = True
    kernel = (7, 7)
    img_size = (128, 128)

    # ---------------------------------------------------------------------
    # PREP RESULT FILE (overwrite old)
    # ---------------------------------------------------------------------
    os.makedirs(os.path.dirname(RESULTS_TXT_PATH), exist_ok=True)

    with open(RESULTS_TXT_PATH, "w") as f:
        TextWriter.write_line(f, "===== IMAGE CORRECTION EXPERIMENT RESULTS =====\n")

        # =====================================================================
        # PHASE 1 — TRAINING OUR CUSTOM CNN WITH FISHEYE IMAGES FOR REFERENCE
        # =====================================================================
        TextWriter.write_line(f, "\n--- PHASE 1: TRAINING OUR CUSTOM CNN WITH FISHEYE IMAGE (REGULAR) FOR REFERENCE ---")
        print(f"Training custom cnn without image correction...")

        # Save image as reference
        first_image_data = image_data_list[16]

        output_path = os.path.join(RESULTS_DIR, "image_no_correction_example.png")
        cv2.imwrite(output_path, first_image_data.image)

        # Finding recommended batch size for given image size
        # Trying to avoid memory issues for larger sizes
        recommended_batch_size = Misc.recommended_batch_size(img_size)

        model, X_test, y_test, history_no_image_correction = CustomCNNTrainer.train_custom_cnn(
            image_data_list,
            img_size=img_size,
            num_conv_layers=layers,
            use_batch_norm=batchnorm,
            kernel_size=kernel,
            batch_size=recommended_batch_size
        )

        result_no_image_correction = ModelEvaluator.evaluate_model(model, history_no_image_correction, X_test, y_test, img_size=img_size)

        TextWriter.write_result_block(
            f,
            setting_name="Image correction: ",
            setting_value=False,
            result=result_no_image_correction,
            is_best=False
        )

        # =====================================================================
        # PHASE 2 — TRAINING OUR CUSTOM CNN WITH CORRECTED IMAGES
        # =====================================================================
        TextWriter.write_line(f,"\n--- PHASE 2: TRAINING OUR CUSTOM CNN WITH CORRECTED IMAGES ---")
        print(f"Training custom cnn with image correction...")

        # Fisheye correction
        # Overwrites original image data in the loaded datalist
        # When the data is passed down to model training, it will get preprocessed
        # keep_ratio_after_cropping: float between 0 and 1
        #         Fraction of the fisheye radius to keep.
        #         For example, 0.9 keeps 90% of the radius (cropping ~10% from the edges).
        #         Not calculated, set up manually based on the images used
        ImageEditor.apply_equisolid_fisheye_correction(image_data_list, keep_ratio_after_cropping=0.75)

        # Save image as reference
        first_image_data = image_data_list[16]

        output_path = os.path.join(RESULTS_DIR, "image_after_correction_example.png")
        cv2.imwrite(output_path, first_image_data.image)

        # Finding recommended batch size for given image size
        # Trying to avoid memory issues for larger sizes
        recommended_batch_size = Misc.recommended_batch_size(img_size)

        model, X_test, y_test, history_with_image_correction = CustomCNNTrainer.train_custom_cnn(
            image_data_list,
            img_size=img_size,
            num_conv_layers=layers,
            use_batch_norm=batchnorm,
            kernel_size=kernel,
            batch_size=recommended_batch_size
        )

        result_with_image_correction = ModelEvaluator.evaluate_model(model, history_with_image_correction, X_test, y_test,
                                                                   img_size=img_size)
        TextWriter.write_result_block(
            f,
            setting_name="Image correction: ",
            setting_value=True,
            result=result_with_image_correction,
            is_best=False
        )

        TextWriter.write_line(f, "\n===== EXPERIMENT FINISHED =====")

        GraphMaker.save_learning_curve_comparison(
            history_no_image_correction,
            history_with_image_correction,
            RESULTS_CURVE_PATH,
            "Learning curves comparison - Original vs Corrected Images",
            label_a="No image correction",
            label_b="With image correction"
        )

    print()
    print(f"Experiment finished.")