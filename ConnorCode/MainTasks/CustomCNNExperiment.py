import os
from ConnorCode.ImageUtils import ImageLoader, GraphMaker
from ConnorCode.CsvUtils import CsvLoader
from ConnorCode.CustomCNN import CustomCNNTrainer
from ConnorCode.CommonUtils import ModelEvaluator
from ConnorCode.CommonUtils import Misc
from ConnorCode.TextUtils import TextWriter

# =============================================================
# 📁 Results paths
# =============================================================
RESULTS_DIR = "../results/customCnnExperiment"

RESULTS_TXT_PATH = os.path.join(
    RESULTS_DIR,
    "custom_cnn_experiments.txt"
)

RESULTS_CURVE_PATH = os.path.join(
    RESULTS_DIR,
    "custom_cnn_learning_curve.png"
)

def find_best_settings_for_custom_cnn():
    print()
    print(f"Training and finding the best settings for our custom CNN...")

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
    layers_to_test = [3, 6, 9]
    batchnorm_to_test = [False, True]
    kernels_to_test = [(3, 3), (5, 5), (7, 7)]
    img_sizes_to_test = [(128, 128), (224, 224), (480, 480), (640, 640)]

    print()
    print(f"Running experiment...")
    # ---------------------------------------------------------------------
    # PREP RESULT FILE (overwrite old)
    # ---------------------------------------------------------------------
    os.makedirs(os.path.dirname(RESULTS_TXT_PATH), exist_ok=True)

    with open(RESULTS_TXT_PATH, "w") as f:
        TextWriter.write_line(f, "===== CUSTOM CNN EXPERIMENT RESULTS =====\n")

        # =================================================================
        # PHASE 1 — FIND BEST NUMBER OF LAYERS
        # =================================================================
        TextWriter.write_line(f, "\n--- PHASE 1: Testing Different Numbers of Layers ---")
        print(f"Testing different numbers of layers: {layers_to_test}")

        best_layers = None
        best_result_layers = None

        for layers in layers_to_test:
            model, X_test, y_test, history = CustomCNNTrainer.train_custom_cnn(
                image_data_list,
                num_conv_layers=layers
            )

            result = ModelEvaluator.evaluate_model(model, history, X_test, y_test, img_size=(128, 128))

            TextWriter.write_result_block(
                f,
                setting_name="Number of layers",
                setting_value=layers,
                result=result,
                is_best=False
            )

            if (best_result_layers is None) or (result.test_rmse < best_result_layers.test_rmse):
                best_result_layers = result
                best_layers = layers

        TextWriter.write_result_block(
            f,
            setting_name="Number of layers",
            setting_value=best_layers,
            result=best_result_layers,
            is_best=True
        )

        # =================================================================
        # PHASE 2 — TEST BATCH NORMALIZATION (YES/NO)
        # =================================================================
        TextWriter.write_line(f, "\n--- PHASE 2: Testing Batch Normalization ---")
        print(f"Testing batch normalization: {batchnorm_to_test}")

        best_bn = None
        best_result_bn = None

        for bn in batchnorm_to_test:
            model, X_test, y_test, history = CustomCNNTrainer.train_custom_cnn(
                image_data_list,
                num_conv_layers=best_layers,
                use_batch_norm=bn
            )

            result = ModelEvaluator.evaluate_model(model, history, X_test, y_test, img_size=(128, 128))

            TextWriter.write_result_block(
                f,
                setting_name="BatchNorm setting",
                setting_value=bn,
                result=result,
                is_best=False
            )

            if (best_result_bn is None) or (result.test_rmse < best_result_bn.test_rmse):
                best_result_bn = result
                best_bn = bn

        TextWriter.write_result_block(
            f,
            setting_name="BatchNorm setting",
            setting_value=best_bn,
            result=best_result_bn,
            is_best=True
        )

        # =================================================================
        # PHASE 3 — TEST KERNEL SIZES
        # =================================================================
        TextWriter.write_line(f, "\n--- PHASE 3: Testing Different Kernel Sizes ---")
        print(f"Testing different kernel sizes: {kernels_to_test}")

        best_kernel = None
        best_result_kernel = None

        for kernel in kernels_to_test:
            model, X_test, y_test, history = CustomCNNTrainer.train_custom_cnn(
                image_data_list,
                num_conv_layers=best_layers,
                use_batch_norm=best_bn,
                kernel_size=kernel
            )

            result = ModelEvaluator.evaluate_model(model, history, X_test, y_test, img_size=(128, 128))

            TextWriter.write_result_block(
                f,
                setting_name="Kernel size",
                setting_value=kernel,
                result=result,
                is_best=False
            )

            if (best_result_kernel is None) or (result.test_rmse < best_result_kernel.test_rmse):
                best_result_kernel = result
                best_kernel = kernel

        TextWriter.write_result_block(
            f,
            setting_name="Kernel size",
            setting_value=best_kernel,
            result=best_result_kernel,
            is_best=True
        )

        # =================================================================
        # PHASE 4 — TEST IMAGE SIZES
        # =================================================================
        TextWriter.write_line(f, "\n--- PHASE 4: Testing Image Sizes ---")
        print(f"Testing image sizes: {img_sizes_to_test}")

        best_img_size = None
        best_result_imgsize = None
        best_history_imgsize = None

        for size in img_sizes_to_test:
            # Finding recommended batch size for given image size
            # Trying to avoid memory issues for larger sizes
            recommended_batch_size = Misc.recommended_batch_size(size)
            model, X_test, y_test, history = CustomCNNTrainer.train_custom_cnn(
                image_data_list,
                img_size=size,
                num_conv_layers=best_layers,
                use_batch_norm=best_bn,
                kernel_size=best_kernel,
                batch_size=recommended_batch_size
            )

            result = ModelEvaluator.evaluate_model(model, history, X_test, y_test, img_size=size)

            TextWriter.write_result_block(
                f,
                setting_name="Image size",
                setting_value=size,
                result=result,
                is_best=False
            )

            if (best_result_imgsize is None) or (result.test_rmse < best_result_imgsize.test_rmse):
                best_result_imgsize = result
                best_img_size = size
                best_history_imgsize = history

        TextWriter.write_result_block(
            f,
            setting_name="Image size",
            setting_value=best_img_size,
            result=best_result_imgsize,
            is_best=True
        )

        TextWriter.write_line(f, "\n===== EXPERIMENT FINISHED =====")

        GraphMaker.save_learning_curve(
            best_history_imgsize,
            save_path=RESULTS_CURVE_PATH,
            title=(
                f"Best Custom CNN Learning Curve\n"
                f"Layers={best_layers}, BN={best_bn}, "
                f"Kernel={best_kernel}, ImgSize={best_img_size}"
            )
        )

    print()
    print(f"Experiment finished.")
