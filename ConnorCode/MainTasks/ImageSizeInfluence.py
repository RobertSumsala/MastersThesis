from ConnorCode.ImageUtils import ImageLoader
from ConnorCode.CsvUtils import CsvLoader
from ConnorCode.Resnet50Utils import Resnet50Trainer
from ConnorCode.CommonUtils import ModelEvaluator
from ConnorCode.CommonUtils import Misc


def investigate_image_size_influence():
    print()
    print(f"Investigating image size influence...")

    image_folder_path = "../Solar_data/joined_orig_data/images"
    image_data_list = ImageLoader.load_images_with_timestamps(image_folder_path)

    csv_path = "../Solar_data/joined_orig_data/out_data_joined.csv"
    CsvLoader.load_irradiance_from_csv(csv_path, image_data_list)

    print()
    print(f"Test print:")
    print(f"First image name: {image_data_list[0].name}")
    print(f"First image timestamp: {image_data_list[0].timestamp}")
    print(f"First image irradiance: {image_data_list[0].irradiance}")


    # Training rest net with default values (test_size, epochs, batch_size)
    # Size of the image is the only important parameter here
    img_sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
    all_results = []

    for size in img_sizes:
        print("\n" + "=" * 60)
        print(f"🧠 Starting training for image size: {size}")
        print("=" * 60)

        # Train model for given size
        recommended_batch_size = Misc.recommended_batch_size(size)
        model, X_test, y_test, history = Resnet50Trainer.train_resnet50_regression(
            image_data_list=image_data_list,
            img_size=size,
            test_size=0.2,
            epochs=10,
            batch_size=recommended_batch_size
        )

        # Evaluate and store result
        result = ModelEvaluator.evaluate_model(model, history, X_test, y_test, size)
        all_results.append(result)

    # Print summary of all results
    print("\n\n📊 Summary of Image Size Experiments:")
    print("=" * 60)
    for res in all_results:
        print(f"Image Size: {res.img_size}")
        print(f"   RMSE: {res.rmse:.2f} W/m²")
        print(f"   R²: {res.r2:.3f}")
        print("-" * 60)
