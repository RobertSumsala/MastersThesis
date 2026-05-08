import os
import cv2
from sklearn.model_selection import KFold
import numpy as np
from ConnorCode.ImageUtils import ImageLoader, GraphMaker
from ConnorCode.CsvUtils import CsvLoader
from ConnorCode.CommonUtils import Misc, ModelEvaluator
from ConnorCode.CustomCNN import CustomCNNTrainer
from ConnorCode.TextUtils import TextWriter
from ConnorCode.ImageUtils import ImageEditor

# =============================================================
# 📁 Results paths
# =============================================================
RESULTS_DIR = "../results/finalEvaluation"

RESULTS_TXT_PATH = os.path.join(
    RESULTS_DIR,
    "final_evaluation_results.txt"
)

RESULTS_CURVE_PATH = os.path.join(
    RESULTS_DIR,
    "1st_fold_learning_curve.png"
)

RESULTS_RESIDUALS_PATH = os.path.join(
    RESULTS_DIR,
    "1st_fold_residuals.png"
)

NUMBER_OF_FOLDS = 10
k_fold = KFold(n_splits=NUMBER_OF_FOLDS, shuffle=True, random_state=42)

def print_kfold_results(
        f,
        mae_scores_train, rmse_scores_train,
        mae_scores_val, rmse_scores_val,
        mae_scores_test, rmse_scores_test
):
    def _log(line):
        TextWriter.write_line(f, line)

    TextWriter.write_line(f, "\n===== K-FOLD RESULTS =====")

    # ---- TRAIN ----
    _log("TRAIN:")
    _log(f"Train MAE:  {np.mean(mae_scores_train):.4f} ± {np.std(mae_scores_train):.4f}")
    _log(f"Train RMSE: {np.mean(rmse_scores_train):.4f} ± {np.std(rmse_scores_train):.4f}")

    # ---- VAL ----
    _log("\nVAL:")
    _log(f"Val MAE:  {np.mean(mae_scores_val):.4f} ± {np.std(mae_scores_val):.4f}")
    _log(f"Val RMSE: {np.mean(rmse_scores_val):.4f} ± {np.std(rmse_scores_val):.4f}")

    # ---- TEST ----
    _log("\nTEST:")
    _log(f"Test MAE:  {np.mean(mae_scores_test):.4f} ± {np.std(mae_scores_test):.4f}")
    _log(f"Test RMSE: {np.mean(rmse_scores_test):.4f} ± {np.std(rmse_scores_test):.4f}")


def evaluate_model_with_best_final_settings():
    print()
    print(f"Evaluating final model with best settings. Using: {NUMBER_OF_FOLDS}k-folds validation...")

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

    # Combine train and validation sets for the k-fold validation
    # As later in the process, always one fold out of the combined set will be used for validation
    # and k-1 folds will be used for training
    full_train_val_data_list = train_image_data_list + val_image_data_list

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

    with open(RESULTS_TXT_PATH, "w") as f:
        TextWriter.write_line(f, f"{NUMBER_OF_FOLDS}K-FOLD VALIDATION RESULTS")

        fold_results = []
        print(f"\nStarting {NUMBER_OF_FOLDS}-fold validation:")
        for fold, (train_idx, val_idx) in enumerate(k_fold.split(full_train_val_data_list)):
            print(f"\n========== FOLD {fold + 1}/{NUMBER_OF_FOLDS} ==========")

            # Split data
            fold_train_data_list = [full_train_val_data_list[i] for i in train_idx]
            fold_val_data_list = [full_train_val_data_list[i] for i in val_idx]

            print(f"Train size: {len(fold_train_data_list)}")
            print(f"Val size: {len(fold_val_data_list)}")
            TextWriter.write_line(f, f"\nFOLD {fold + 1}")
            TextWriter.write_line(f, f"Train size: {len(fold_train_data_list)}")
            TextWriter.write_line(f, f"Val size: {len(fold_val_data_list)}")

            # Train model
            model, history = CustomCNNTrainer.train_custom_cnn_fixed_splits(
                train_data=fold_train_data_list,
                val_data=fold_val_data_list,
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
                augmentations=None,
                aug_counter=None
            )

            # Evaluate on FIXED test set
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

            fold_results.append(result)

            TextWriter.write_line(f, "\n--- TRAINING OF CUSTOM CNN ---\n")
            TextWriter.write_result_block(
                f,
                setting_name="Fold:",
                setting_value=fold + 1,
                result=result,
                is_best=False
            )
            TextWriter.write_line(f, "-" * 60)

            # Save graphs for the first fold as an example
            if fold + 1 == 1:
                print(f"\nSaving graphs for the first fold...")

                # Save the learning curve
                GraphMaker.save_learning_curve(
                    history,
                    save_path=RESULTS_CURVE_PATH,
                    title=f"Best Custom CNN Learning Curve\n"
                )

                # Save residual plot
                GraphMaker.save_residual_plot(
                    y_true=result.y_true,
                    y_pred=result.y_pred,
                    save_path=RESULTS_RESIDUALS_PATH,
                    title=f"Best Custom CNN Residuals\n"
                )
            print(f"\n End of fold: {fold + 1}")

        # Final results after all the runs
        mae_scores_train = [r.train_mae for r in fold_results]
        rmse_scores_train = [r.train_rmse for r in fold_results]
        mae_scores_val = [r.val_mae for r in fold_results]
        rmse_scores_val = [r.val_rmse for r in fold_results]
        mae_scores_test = [r.test_mae for r in fold_results]
        rmse_scores_test = [r.test_rmse for r in fold_results]

        print_kfold_results(f, mae_scores_train, rmse_scores_train, mae_scores_val, rmse_scores_val, mae_scores_test, rmse_scores_test)

        TextWriter.write_line(f, "\n===== EXPERIMENT FINISHED =====")

    print("\nExperiment completed.")
