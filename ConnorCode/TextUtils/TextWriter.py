import os

def write_line(file, text):
    file.write(text + "\n")
    print(text)

"""
Writes formatted train/val/test metrics for one experiment setting.
"""
def write_result_block(
    f,
    setting_name,
    setting_value,
    result,
    is_best=False
):
    prefix = f"Selected best {setting_name}" if is_best else setting_name

    write_line(f, f"{prefix}: {setting_value}")

    if is_best:
        write_line(f, f"  Best Train -> RMSE: {result.train_rmse:.2f}, MAE: {result.train_mae:.2f}")
        write_line(f, f"  Best Val   -> RMSE: {result.val_rmse:.2f}, MAE: {result.val_mae:.2f}")
        write_line(
            f,
            f"  Best Test  -> RMSE: {result.test_rmse:.2f}, "
            f"MAE: {result.test_mae:.2f}, R²: {result.test_r2:.3f}"
        )
    else:
        write_line(f, f"  Train -> RMSE: {result.train_rmse:.2f}, MAE: {result.train_mae:.2f}")
        write_line(f, f"  Val   -> RMSE: {result.val_rmse:.2f}, MAE: {result.val_mae:.2f}")
        write_line(
            f,
            f"  Test  -> RMSE: {result.test_rmse:.2f}, "
            f"MAE: {result.test_mae:.2f}, R²: {result.test_r2:.3f}"
        )

    write_line(f, "-" * 60)
