# - - - - Sky Image-Based Solar Energy Forecasting - - - - - - - - - - - -
# author: Robert Sumsala
# year: 2025/2026
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import tensorflow as tf
import os
import sys
from datetime import datetime
import time
from MainTasks import ImageSizeInfluence
from MainTasks import CustomCNNExperiment
from MainTasks import FisheyeCorrectionExperiment
from MainTasks import OptunaLibraryExperiment
from MainTasks import ChannelAnalysis
from MainTasks import ImageCroppingExperiment
from MainTasks import FisheyeCorrectionPart2
from MainTasks import AugmentationExperiment
from MainTasks import FinalEvaluation

# =============================================================
# LOG CONSOLE OUTPUT - for debug
# =============================================================

LOG_DIR = "../results/logs"
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(LOG_DIR, f"run_{timestamp}.log")

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(LOG_PATH, "w")

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)


"""
This function serves as a central controller that calls and executes
specific tasks or modules located in the 'MainTasks' directory.

You can manually modify the function call to execute
only selected sub-tasks or individual functions as needed.
Or run the whole workflow, by uncommenting everything.
"""
def main():
    print(tf.config.list_physical_devices('GPU'))

    # Prevents running the code if GPU setup fails
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("GPU not available — aborting run")

    # # Testing the best image size for Resnet50
    # ImageSizeInfluence.investigate_image_size_influence()

    # # Training our own CNN and finding the best settings for it
    # # Testing:
    # # - number_of_layers
    # # - batch_normalization - yes/no
    # # - kernel_size for filters
    # # - image_size
    # # Results are saved into a file, path:
    # CustomCNNExperiment.find_best_settings_for_custom_cnn()

    # # Using our customCNN with the best settings from CustomCNNExperiment
    # # Find out if we can do successful image correction (remove fish eye wrapping)
    # # and compare results with the original images being used
    # FisheyeCorrectionExperiment.compare_original_and_corrected_images()

    # # Running parameter search using Optuna Library,
    # # oppose to our custom search, in one of the previous experiments.
    # # Searching for best:
    # # - number_of_layers
    # # - batch_normalization - yes/no
    # # - kernel_size for filters
    # # - img_size (options narrowed down by our manual parameters search)
    # # - learning_rate (wasn't included in our manual search)
    # # - number_of_filters (wasn't included in our manual search)
    # # Number of trials can be set inside the file as a global parameter
    # OptunaLibraryExperiment.find_best_settings_for_custom_cnn_using_optuna()

    # # Analysing Channels:
    # #  - RGB
    # #  - HSV
    # #  - YCrCb
    # # Determining which channel holds the most information
    # ChannelAnalysis.find_channel_carrying_the_most_info()

    # # Testing the influence of masking images
    # #  - we suspect, getting rid of the black coloured frame will improve results
    # #  - image is first cropped and then masked - the remaining black around the circle
    # #    is replaced by a mean pixel, which influences the CNN less than a black pixel
    # ImageCroppingExperiment.mask_images_and_train_custom_cnn()

    # # Testing additional techniques of getting rid of the fish eye warping effect
    # # and trying if it is even worth it.
    # # You can pick here whichever technique you would like to test.
    # # Fish-eye correction techniques:
    # #  - "none" -> doesn't use any correction
    # #  - "rectilinear_openCV" -> Using OpenCV fisheye model to convert to rectilinear projection.
    # #  - "angular_sectors" -> Converting the fisheye image into a polar representation of the sky
    # #  - "equal_area" -> Applies Lambert azimuthal equal-area projection to a fisheye sky image.
    # #                 -> DO NOT USE - it's computationally too expensive, it wasn't perfected,
    # #                    it is kept here, since it's been tested
    # FisheyeCorrectionPart2.test_fisheye_correction_technique(technique="rectilinear_openCV")

    # # Testing best augmentations
    # # You can pick whatever combinations of available techniques
    # # Available augmentations:
    # # - None -> no augmentations will be used
    # # - "rotate" -> rotation from 10-15% will be randomly applied
    # # - "translate" -> applies random shift (translation), up to 10%
    # # - "scale" -> applies random, minimalistic scaling (zoom in/out - from 90% to 110%)
    # # - "brightness" -> darkens/brightens image randomly on given scale
    # # - "clahe" -> applies CLAHE (adaptive histogram equalization)
    # # - "noise" -> applies gaussian noise
    # # - "blur" -> applies a gaussian blur
    # # Based on our experiments, the best combination of augmentations seems to be:
    # #  - "rotate" + "clahe" + "noise" + "blur"
    # # NOTES:
    # #  - Augmentation can be passed with a weight that represents the chance
    # # it will be picked to augment an image, especially useful when multiple
    # # augmentations are picked to choose from. e.g. {"rotate": 0.6, "clahe": 0.2}.
    # # - Even if passing only one augmentation, it needs to be passed as a dictionary
    # # with a probability being 1.0, to make sure it's applied. e.g. {"rotate": 1.0}
    # augmentations = {"rotate": 0.15,
    #                  "clahe": 0.4,
    #                  "noise": 0.25,
    #                  "blur": 0.25}
    # AugmentationExperiment.test_augmentations(augmentations=augmentations)

    # FINAL EVALUATION OF THE MODEL WITH THE BEST SETTINGS
    # This evaluates our custom CNN with the best settings.
    # It does a K-FOLD (K = 10 by default) validation, and calculates the average MAE and RMSE with deviations.
    # It also saves a learning curve and residual plot for the first run of the validation
    # NOTES:
    #  - best settings should be saved in ../BestParameters/best_parameters_from_optuna.txt in correct form,
    #    or they can be entered manually inside the function where the training of the function is called
    FinalEvaluation.evaluate_model_with_best_final_settings()




if __name__ == "__main__":
    start_time = time.time()
    print(f"LOGGING TO {LOG_PATH}")
    print(f"START TIME: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n\n")

    main()

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print(f"\n\nEND TIME: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    print(f"TOTAL RUNTIME: {hours}h {minutes}m {seconds}s")


# Why the name ConnorCode? Well, one of the existing solutions
# to this irradiance prediction issue was named SkyNet.
# So to be a proper rival to this work, the only valid name was Connor :).
# (both names are references to the Terminator franchise)
