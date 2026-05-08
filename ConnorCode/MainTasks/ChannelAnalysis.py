import os
import gc
import tensorflow as tf
from ConnorCode.ImageUtils import ImageLoader, GraphMaker
from ConnorCode.CsvUtils import CsvLoader
from ConnorCode.CustomCNN import CustomCNNTrainer
from ConnorCode.CommonUtils import ModelEvaluator, Misc
from ConnorCode.TextUtils import TextWriter

# =============================================================
# 📁 Results paths
# =============================================================
RESULTS_DIR = "../results/channelAnalysis"

RESULTS_TXT_PATH = os.path.join(
    RESULTS_DIR,
    "channel_analysis_results.txt"
)

"""
Trains and evaluates the CNN using single-channel inputs
from a given color space.

Parameters:
    color_space_name (str): e.g. "RGB", "CMYK", "HSV"
    channels (list[str]): list of channels to test, e.g. ["R","G","B"]
Returns:
    channel_results (list[dict])
"""
def analyse_channels(color_space_name, channels, train_data, val_data, test_data, best_params, results_file_handle):
    channel_results = []

    print("\n" + "#" * 60)
    print(f"Analysing color space: {color_space_name}")
    print("#" * 60)

    for channel in channels:
        print("\n" + "=" * 50)
        print(f"Training using ONLY {channel} channel ({color_space_name})")
        print("=" * 50)

        TextWriter.write_line(results_file_handle, f"\n\n===== COLOR SPACE: {color_space_name} =====\n")

        tf.keras.backend.clear_session()
        gc.collect()

        batch_size = Misc.recommended_batch_size(
            (best_params.img_size, best_params.img_size)
        )

        model, history = CustomCNNTrainer.train_custom_cnn_fixed_splits(
            train_data=train_data,
            val_data=val_data,
            img_size=(best_params.img_size, best_params.img_size),
            num_conv_layers=best_params.num_layers,
            filters=best_params.base_filters,
            kernel_size=(best_params.kernel_size, best_params.kernel_size),
            use_batch_norm=best_params.batch_norm,
            epochs=best_params.num_epochs,
            batch_size=batch_size,
            learning_rate=best_params.learning_rate,
            channel=channel,
            color_space=color_space_name
        )

        result = ModelEvaluator.evaluate_model_with_generator(
            model=model,
            history=history,
            test_data=test_data,
            img_size=(best_params.img_size, best_params.img_size),
            batch_size=batch_size,
            channel=channel,
            color_space=color_space_name
        )

        print(f"\n📊 Results for channel {channel}:")
        print(f"   RMSE: {result.test_rmse:.2f}")
        print(f"   MAE:  {result.test_mae:.2f}")
        print(f"   R²:   {result.test_r2:.3f}")

        TextWriter.write_line(results_file_handle,f"\n--- RESULTS FOR CHANNEL {channel} ({color_space_name}) ---\n")

        TextWriter.write_result_block(
            results_file_handle,
            setting_name="Channel",
            setting_value=f"{channel} ({color_space_name})",
            result=result,
            is_best=False
        )

        channel_results.append({
            "color_space": color_space_name,
            "channel": channel,
            "rmse": result.test_rmse,
            "mae": result.test_mae,
            "r2": result.test_r2
        })

    return channel_results

"""
Trains and evaluates the CNN using ALL channels
of a given color space (no channel masking).

Parameters:
    color_space_name (str): e.g. "RGB", "HSV", "YCrCb"
Returns:
    result_dict (dict)
"""
def analyse_full_color_space(color_space_name, train_data, val_data, test_data, best_params, results_file_handle):
    print("\n" + "=" * 60)
    print(f"Training using FULL {color_space_name} (all 3 channels)")
    print("=" * 60)

    TextWriter.write_line(results_file_handle,f"\n\n===== FULL COLOR SPACE: {color_space_name} =====\n")

    tf.keras.backend.clear_session()
    gc.collect()

    batch_size = Misc.recommended_batch_size(
        (best_params.img_size, best_params.img_size)
    )

    model, history = CustomCNNTrainer.train_custom_cnn_fixed_splits(
        train_data=train_data,
        val_data=val_data,
        img_size=(best_params.img_size, best_params.img_size),
        num_conv_layers=best_params.num_layers,
        filters=best_params.base_filters,
        kernel_size=(best_params.kernel_size, best_params.kernel_size),
        use_batch_norm=best_params.batch_norm,
        epochs=best_params.num_epochs,
        batch_size=batch_size,
        learning_rate=best_params.learning_rate,
        channel=None,
        color_space=color_space_name
    )

    result = ModelEvaluator.evaluate_model_with_generator(
        model=model,
        history=history,
        test_data=test_data,
        img_size=(best_params.img_size, best_params.img_size),
        batch_size=batch_size,
        channel=None,
        color_space=color_space_name
    )

    print(f"\n📊 Results for FULL {color_space_name}:")
    print(f"   RMSE: {result.test_rmse:.2f}")
    print(f"   MAE:  {result.test_mae:.2f}")
    print(f"   R²:   {result.test_r2:.3f}")

    TextWriter.write_result_block(
        results_file_handle,
        setting_name="Color space (full)",
        setting_value=color_space_name,
        result=result,
        is_best=False
    )

    return {
        "color_space": color_space_name,
        "channel": "FULL",
        "rmse": result.test_rmse,
        "mae": result.test_mae,
        "r2": result.test_r2
    }


"""
MAIN FUNCTION OF THE CHANNEL ANALYSIS EXPERIMENT
"""
def find_channel_carrying_the_most_info():
    print()
    print(f"Analyzing channels and finding the one that carries the most information...")

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

    # ---------------------------------------------------------------------
    # PREP RESULT FILE (overwrite old)
    # ---------------------------------------------------------------------
    os.makedirs(os.path.dirname(RESULTS_TXT_PATH), exist_ok=True)

    COLOR_SPACES = {
        "RGB": ["R", "G", "B"],
        "HSV": ["H", "S", "V"],
        "YCrCb": ["Y", "Cr", "Cb"]
    }
    all_results = []
    full_color_space_results = []

    with open(RESULTS_TXT_PATH, "w") as f:
        for color_space_name, channels in COLOR_SPACES.items():
            # 1️⃣ Single-channel analysis
            channel_results = analyse_channels(
                color_space_name=color_space_name,
                channels=channels,
                train_data=train_image_data_list,
                val_data=val_image_data_list,
                test_data=test_image_data_list,
                best_params=best_params,
                results_file_handle=f,
            )

            # 2️⃣ Full 3-channel analysis
            full_result = analyse_full_color_space(
                color_space_name=color_space_name,
                train_data=train_image_data_list,
                val_data=val_image_data_list,
                test_data=test_image_data_list,
                best_params=best_params,
                results_file_handle=f,
            )

            all_results.append(channel_results)
            full_color_space_results.append(full_result)

            # Save graph per color space
            graph_path = os.path.join(
                RESULTS_DIR,
                f"{color_space_name}_channel_analysis.png"
            )

            GraphMaker.save_channel_rmse_plot(
                channel_results=channel_results,
                save_path=graph_path,
                color_space=color_space_name
            )

        TextWriter.write_line(f, "\n===== EXPERIMENT FINISHED =====")

        # Save global full color space comparison graph
        full_graph_path = os.path.join(
            RESULTS_DIR,
            "FULL_color_space_comparison.png"
        )

        GraphMaker.save_full_color_space_comparison_plot(
            full_results=full_color_space_results,
            save_path=full_graph_path
        )

    print("Channel analysis finished.")
