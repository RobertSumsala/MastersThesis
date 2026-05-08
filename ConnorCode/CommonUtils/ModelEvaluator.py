from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from ConnorCode.CommonUtils import Misc

@dataclass
class ModelEvaluationResult:
    img_size: tuple

    train_rmse: float
    train_mae: float

    val_rmse: float
    val_mae: float

    test_rmse: float
    test_mae: float
    test_r2: float

    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None


"""
Extracts final train/validation RMSE and MAE from training history
"""
def _extract_train_val_metrics(history):
    train_rmse = np.sqrt(history.history["loss"][-1])
    train_mae  = history.history["mae"][-1]

    val_rmse = np.sqrt(history.history["val_loss"][-1])
    val_mae  = history.history["val_mae"][-1]

    return train_rmse, train_mae, val_rmse, val_mae


"""
Similar version to `evaluate_model`,
except it uses generator for the test data.
Note: For more insight look at the description of `evaluate_model` func below.
"""
def evaluate_model_with_generator(
        model,
        history,
        test_data,
        img_size,
        batch_size,
        channel=None,
        color_space="RGB",
        use_mask=False,
        mask_radius=None,
        correction_technique="none"
):
    print()
    print(" - MODEL: Evaluating model (generator-based)...")

    # ---- Train & validation metrics ----
    train_rmse, train_mae, val_rmse, val_mae = _extract_train_val_metrics(history)

    # -------------------------------------------------
    # Test generator (ORDER PRESERVED)
    # -------------------------------------------------
    test_idx = np.arange(len(test_data))

    test_gen = Misc.gen(
        test_idx,
        batch_size=batch_size,
        data=test_data,
        img_size=img_size,
        channel=channel,
        color_space=color_space,
        use_mask=use_mask,
        radius=mask_radius,
        correction_technique=correction_technique,
        augmentations=None
    )

    steps_test = max(1, int(np.ceil(len(test_idx) / batch_size)))

    # -------------------------------------------------
    # Collect predictions and ground truth
    # -------------------------------------------------
    preds = []
    y_true = []

    for _ in range(steps_test):
        X_batch, y_batch = next(test_gen)
        batch_preds = model.predict(X_batch, verbose=0)

        preds.append(batch_preds)
        y_true.append(y_batch)

    preds = np.vstack(preds).squeeze()
    y_true = np.hstack(y_true).squeeze()

    # Trim in case generator overshoots
    preds = preds[:len(test_data)]
    y_true = y_true[:len(test_data)]

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
    test_rmse = np.sqrt(mean_squared_error(y_true, preds))
    test_mae  = mean_absolute_error(y_true, preds)
    test_r2   = r2_score(y_true, preds)

    print(f"✅ Evaluation Results - image size set to: {img_size}:")
    print(f"   RMSE: {test_rmse:.2f} W/m²")
    print(f"   MAE:  {test_mae:.2f} W/m²")
    print(f"   R²:   {test_r2:.3f}")

    return ModelEvaluationResult(
        img_size=img_size,

        train_rmse=train_rmse,
        train_mae=train_mae,

        val_rmse=val_rmse,
        val_mae=val_mae,

        test_rmse=test_rmse,
        test_mae=test_mae,
        test_r2=test_r2,

        y_true=y_true,
        y_pred=preds
    )


"""
Evaluates a trained model on the test dataset and returns performance metrics.

Args:
    model (tf.keras.Model): trained model to evaluate.
    history: history of the model used for extraction of train and val data
    X_test (np.ndarray): test images.
    y_test (np.ndarray): true irradiance values.
    img_size (tuple): image size used for training/evaluation.

Returns:
    ModelEvaluationResult: dataclass containing RMSE, MAE and R² for train, val and indeed test.
"""
def evaluate_model(model, history, X_test, y_test, img_size):
    print()
    print(f" - MODEL: Evaluating model...")

    # ---- Train & validation metrics ----
    train_rmse, train_mae, val_rmse, val_mae = _extract_train_val_metrics(history)

    # ---- Test metrics ----
    preds = model.predict(X_test)

    test_rmse = np.sqrt(mean_squared_error(y_test, preds))
    test_mae = mean_absolute_error(y_test, preds)
    test_r2 = r2_score(y_test, preds)

    print(f"✅ Evaluation Results - image size set to: {img_size}:")
    print(f"   RMSE: {test_rmse:.2f} W/m²")
    print(f"   MAE:  {test_mae:.2f} W/m²")
    print(f"   R²:   {test_r2:.3f}")

    return ModelEvaluationResult(
        img_size=img_size,

        train_rmse=train_rmse,
        train_mae=train_mae,

        val_rmse=val_rmse,
        val_mae=val_mae,

        test_rmse=test_rmse,
        test_mae=test_mae,
        test_r2=test_r2
    )
