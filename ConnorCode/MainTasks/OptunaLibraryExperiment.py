import os
import gc
import optuna
import tensorflow as tf
from ConnorCode.ImageUtils import ImageLoader, GraphMaker
from ConnorCode.CsvUtils import CsvLoader
from ConnorCode.TextUtils import TextWriter
from ConnorCode.OptunaUtils import OptunaObjectives
from ConnorCode.CustomCNN import CustomCNNTrainer
from ConnorCode.CommonUtils import ModelEvaluator
from ConnorCode.CommonUtils import Misc


# =============================================================
# 📁 Results paths
# =============================================================
RESULTS_DIR = "../results/optunaLibraryExperiment"

RESULTS_TXT_PATH = os.path.join(
    RESULTS_DIR,
    "optuna_results.txt"
)

RESULTS_CURVE_PATH = os.path.join(
    RESULTS_DIR,
    "custom_cnn_with_optuna_found_parameters_learning_curve.png"
)

NUMBER_OF_TRIALS = 60

def find_best_settings_for_custom_cnn_using_optuna():
    print()
    print(f"Training and finding the best settings for our custom CNN, using Optuna Library...")

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

    # ---------------------------------------------------------------------
    # PREP RESULT FILE (overwrite old)
    # ---------------------------------------------------------------------
    os.makedirs(os.path.dirname(RESULTS_TXT_PATH), exist_ok=True)

    with open(RESULTS_TXT_PATH, "w") as f:
        # =======================================================
        # PHASE 1 - Create and run study using Optuna Library
        # =======================================================
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: OptunaObjectives.objective_custom_cnn(
                trial,
                train_image_data_list,
                val_image_data_list,
                test_image_data_list
            ),
            n_trials=NUMBER_OF_TRIALS,
            show_progress_bar=True
        )

        # Save results to txt file
        TextWriter.write_line(f, "===== OPTUNA CUSTOM CNN RESULTS =====\n")
        TextWriter.write_line(f, f"Number of trials: {NUMBER_OF_TRIALS}\n")

        # ------------------------------------------------------
        # Check if any trial completed successfully
        # ------------------------------------------------------
        completed_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if len(completed_trials) == 0:
            print("❌ No successful Optuna trials. Aborting Phase 2.")
            TextWriter.write_line(f, "\n❌ No successful Optuna trials. Phase 2 skipped.\n")
            return

        TextWriter.write_line(f, "\n--- BEST PARAMETERS FOUND ---\n")
        for key, value in study.best_params.items():
            TextWriter.write_line(f, f"{key}: {value}")

        TextWriter.write_line(f, "\n--- BEST VALIDATION RMSE ---\n")
        TextWriter.write_line(f, f"{study.best_value}\n")

        print()
        print("Optuna search finished.")
        print("Best parameters:", study.best_params)
        print("Best validation RMSE:", study.best_value)

        # ==============================================================
        # PHASE 2 - Training our CustomCNN with newly found Best Params
        # ==============================================================
        print("\nTraining Custom CNN with newly found best parameters...")
        best = study.best_params
        best_img_size = (best["img_size"], best["img_size"])
        best_number_of_layers = best["num_layers"]
        best_batch_norm_setting = best["batch_norm"]
        best_kernel_size = (best["kernel_size"], best["kernel_size"])
        best_learning_rate = best["learning_rate"]
        best_number_of_filters = best["base_filters"]
        best_number_of_epochs = best["num_epochs"]

        batch_size = Misc.recommended_batch_size(best_img_size)

        # Clear after Optuna search
        tf.keras.backend.clear_session()
        gc.collect()
        model, history = CustomCNNTrainer.train_custom_cnn_fixed_splits(
            train_data=train_image_data_list,
            val_data=val_image_data_list,
            img_size=best_img_size,
            num_conv_layers=best_number_of_layers,
            use_batch_norm=best_batch_norm_setting,
            kernel_size=best_kernel_size,
            batch_size=batch_size,
            epochs=best_number_of_epochs,
            learning_rate=best_learning_rate,
            filters=best_number_of_filters
        )

        result = ModelEvaluator.evaluate_model_with_generator(
            model,
            history,
            test_data=test_image_data_list,
            img_size=best_img_size,
            batch_size=batch_size
        )

        TextWriter.write_line(f, "\n--- NEWLY TRAINED CUSTOM CNN WITH OPTUNA'S BEST PARAMETERS ---\n")
        TextWriter.write_result_block(
            f,
            setting_name="Optuna's best parameters: ",
            setting_value=True,
            result=result,
            is_best=False
        )

        TextWriter.write_line(f, "\n===== EXPERIMENT FINISHED =====")

        GraphMaker.save_learning_curve(
            history,
            save_path=RESULTS_CURVE_PATH,
            title=(
                f"Best Custom CNN Learning Curve\n"
                f"Layers={best_number_of_layers}, BN={best_batch_norm_setting}, "
                f"Kernel={best_kernel_size}, ImgSize={best_img_size, }"
                f"LearningRate={best_learning_rate}, Filters={best_number_of_filters},"
                f"Epochs={best_number_of_epochs}"
            )
        )

    print()
    print(f"Experiment finished.")
