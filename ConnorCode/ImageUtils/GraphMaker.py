import matplotlib.pyplot as plt

"""
Creates and saves a bar plot comparing FULL 3-channel
performance of different color spaces.
"""
def save_full_color_space_comparison_plot(full_results, save_path):
    print()
    print(" - GRAPH_MAKER: Plotting FULL color space comparison graph...")

    color_space_labels = [r["color_space"] for r in full_results]
    rmses = [r["rmse"] for r in full_results]

    # Optional: neutral but distinct colors per color space
    color_map = {
        "RGB": "#666666",
        "HSV": "#999999",
        "YCrCb": "#333333"
    }

    colors = [color_map.get(cs, "#777777") for cs in color_space_labels]

    plt.figure()
    plt.bar(color_space_labels, rmses, color=colors)
    plt.xlabel("Color space (all channels active)")
    plt.ylabel("Test RMSE [W/m²]")
    plt.title("Full color space comparison")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


"""
Creates and saves a bar plot comparing test RMSE values
for individual channels of a given color space.
Each bar is colored according to the channel it represents.
"""
def save_channel_rmse_plot(channel_results, save_path, color_space):
    print()
    print(f" - GRAPH_MAKER: Plotting {color_space} channel <-> RMSE graph...")

    # ------------------------------------------------------------------
    # Label maps (for readable graph names)
    # ------------------------------------------------------------------
    LABEL_MAPS = {
        "RGB": {
            "R": "Red",
            "G": "Green",
            "B": "Blue"
        },
        "CMYK": {
            "C": "Cyan",
            "M": "Magenta",
            "Y": "Yellow",
            "K": "Black"
        },
        "HSV": {
            "H": "Hue",
            "S": "Saturation",
            "V": "Value"
        },
        "YCrCb": {
            "Y": "Luminance (Y)",
            "Cr": "Chrominance Red (Cr)",
            "Cb": "Chrominance Blue (Cb)"
        }
    }

    # ------------------------------------------------------------------
    # Color maps (actual bar colors)
    # ------------------------------------------------------------------
    COLOR_MAPS = {
        "RGB": {
            "R": "#FF0000",
            "G": "#00AA00",
            "B": "#0000FF"
        },
        "CMYK": {
            "C": "#00FFFF",
            "M": "#FF00FF",
            "Y": "#FFFF00",
            "K": "#000000"
        },
        "HSV": {
            "H": "#FF8800",
            "S": "#00CC88",
            "V": "#8888FF"
        },
        "YCrCb": {
            "Y": "#444444",
            "Cr": "#FF4444",
            "Cb": "#4444FF"
        }
    }

    # Safety check
    if color_space not in LABEL_MAPS:
        raise ValueError(f"Unsupported color space: {color_space}")

    label_map = LABEL_MAPS[color_space]
    color_map = COLOR_MAPS[color_space]

    channels = [label_map[r["channel"]] for r in channel_results]
    rmses = [r["rmse"] for r in channel_results]
    colors = [color_map[r["channel"]] for r in channel_results]

    plt.figure()
    plt.bar(channels, rmses, color=colors)

    for i, v in enumerate(rmses):
        plt.text(i, v + 0.01 * max(rmses), f"{v:.2f}", ha='center')

    plt.xlabel(f"{color_space} channel")
    plt.ylabel("Test RMSE [W/m²]")
    plt.title(f"{color_space} Channel-wise Prediction Error")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

"""
Saves residual plot to disk.
Overwrites existing image if present.
"""
def save_residual_plot(y_true, y_pred, save_path, title):
    import matplotlib.pyplot as plt
    import numpy as np

    print()
    print(" - GRAPH_MAKER: Saving residual plot")

    # Compute residuals
    residuals = np.array(y_true) - np.array(y_pred)

    plt.figure(figsize=(8, 5))

    # Scatter plot
    plt.scatter(y_pred, residuals, alpha=0.5)

    # Horizontal line at 0 (perfect prediction)
    plt.axhline(y=0, linestyle="--")

    plt.xlabel("Predicted Irradiance")
    plt.ylabel("Residuals (True - Predicted)")
    plt.title(title)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

"""
Saves training & validation loss curves to disk.
Overwrites existing image if present.
"""
def save_learning_curve(history, save_path, title):
    print()
    print(f" - GRAPH_MAKER: Saving the best learning curve")

    plt.figure(figsize=(8, 5))

    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

"""
Saves a comparison of two training & validation learning curves to disk.
Each history is shown with a consistent color family.
Overwrites existing image if present.
"""
def save_learning_curve_comparison(
    history_a,
    history_b,
    save_path,
    title,
    label_a="Original",
    label_b="New"
):
    print()
    print(" - GRAPH_MAKER: Saving learning curve comparison")

    plt.figure(figsize=(9, 5))

    # --- History A (warm colors) ---
    plt.plot(
        history_a.history["loss"],
        color="tab:red",
        linestyle="-",
        label=f"{label_a} – Train"
    )
    plt.plot(
        history_a.history["val_loss"],
        color="tab:orange",
        linestyle="--",
        label=f"{label_a} – Val"
    )

    # --- History B (cool colors) ---
    plt.plot(
        history_b.history["loss"],
        color="tab:blue",
        linestyle="-",
        label=f"{label_b} – Train"
    )
    plt.plot(
        history_b.history["val_loss"],
        color="tab:green",
        linestyle="--",
        label=f"{label_b} – Val"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
