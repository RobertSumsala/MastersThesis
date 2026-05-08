from ConnorCode.CustomCNN import CustomCNNTrainer
from ConnorCode.CommonUtils import ModelEvaluator
from ConnorCode.CommonUtils import Misc
import tensorflow as tf
import gc
import optuna
import traceback
from ConnorCode.CommonUtils import Misc

"""
Optuna objective for tuning Custom CNN hyperparameters.
Returns validation RMSE (to minimize).
"""
def objective_custom_cnn(trial, train_data, val_data, test_data):
    print(f"\n - OPTUNA: Running trial n.{trial.number + 1}...")
    Misc.print_gpu_mem("Before clear")
    tf.keras.backend.clear_session()
    gc.collect()
    Misc.print_gpu_mem("After clear")

    # -----------------------------
    # Hyperparameters to optimize
    # -----------------------------

    # Number of convolutional layers
    num_layers = trial.suggest_int("num_layers", 3, 9)

    # Epochs
    num_epochs = trial.suggest_int("num_epochs", 5, 15)

    # Batch normalization on/off
    use_batch_norm = trial.suggest_categorical("batch_norm", [True, False])

    # Kernel size
    kernel_size_value = trial.suggest_categorical("kernel_size", [3, 5, 7])
    kernel_size = (kernel_size_value, kernel_size_value)

    # Base number of filters in the first layer
    base_filters = trial.suggest_categorical(
        "base_filters",
        [16, 32, 64, 128]
    )

    # Image size (based on our manual parameter search, we narrowed down the sizes to 128p and 256p
    img_size_value = trial.suggest_categorical("img_size", [128, 256])
    img_size = (img_size_value, img_size_value)

    # Learning rate (log scale)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # ---------------------------------------------------------------------------------------
    # Pruning combinations that will most likely cause OOM issues
    # ---------------------------------------------------------------------------------------
    if img_size_value == 256 and base_filters >= 32 and num_layers >= 7:
        trial.set_user_attr(
            "pruned_reason",
            f"OOM risk: image size == 256, filters >= 32 ({base_filters}), layers>=7 ({num_layers})"
        )
        raise optuna.TrialPruned()

    if img_size_value == 256 and kernel_size_value == 7 and num_layers >= 6:
        trial.set_user_attr(
            "pruned_reason",
            f"OOM risk: image size == 256, kernel size == 7, layers >= 6 ({num_layers})"
        )
        raise optuna.TrialPruned()

    if img_size_value == 256 and use_batch_norm and num_layers >= 8:
        trial.set_user_attr(
            "pruned_reason",
            f"OOM risk: image size == 256, use batch norm == true, layers >= 8 ({num_layers})"
        )
        raise optuna.TrialPruned()

    if img_size_value == 256 and num_epochs >= 12 and num_layers >= 8:
        trial.set_user_attr(
            "pruned_reason",
            f"OOM risk: image size == 256, epochs >= 12 ({num_epochs}), layers >= 8 ({num_layers})"
        )
        raise optuna.TrialPruned()

    if base_filters == 128 and num_layers >= 4:
        trial.set_user_attr(
            "pruned_reason",
            f"OOM risk: filters == 128 ({base_filters}), layers >= 4 ({num_layers})"
        )
        raise optuna.TrialPruned()

    # Adaptive batch size based on image size
    batch_size = Misc.recommended_batch_size(img_size)

    try:
        # -----------------------------
        # Train model
        # -----------------------------
        model, history = CustomCNNTrainer.train_custom_cnn_fixed_splits(
            train_data,
            val_data,
            img_size=img_size,
            num_conv_layers=num_layers,
            use_batch_norm=use_batch_norm,
            kernel_size=kernel_size,
            batch_size=batch_size,
            filters=base_filters,
            epochs=num_epochs,
            learning_rate=learning_rate
        )

        # -----------------------------
        # Evaluate performance
        # -----------------------------
        result = ModelEvaluator.evaluate_model_with_generator(
            model=model,
            history=history,
            test_data=test_data,
            img_size=img_size,
            batch_size=batch_size
        )

    except (ValueError, RuntimeError, MemoryError) :
        print("❌ Trial failed:")
        traceback.print_exc()
        return float("inf")

# Store extra info so you can retrieve them from study later
    trial.set_user_attr("train_rmse", result.train_rmse)
    trial.set_user_attr("val_rmse", result.val_rmse)
    trial.set_user_attr("test_rmse", result.test_rmse)

    # Optuna minimizes the return value
    return result.val_rmse
