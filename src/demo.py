# %% <- Runs .py files as IPython cells in VSCode
def run_from_ipython():
    try:
        __IPYTHON__  # type: ignore
        return True
    except NameError:
        return False

import os

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

from visualisations.visualiser import Visualiser
import utils.utils as utils

device = utils.SelectDevice()

is_ipython = run_from_ipython()

# Change 'ndims_36' to any of: 'ndims_9', 'ndims_36', 'ndims_49', 'ndims_100'
path_of_run = "models/RENIVariationalAutoDecoder/network_128_5/ndims_36/"
# The Visualiser class contains methods for displaying model outputs
vis = Visualiser(path_of_run, device, training_type="Test", img_size=128)
# A small selection of the test dataset images, choose any idx's between 0 and 20
img_ids = [7, 4, 5, 17]
print("producing figure comparison.png")
vis.comparison(img_ids, save=True, fname=path + "/imgs/comparison")

print("producing figure heatmaps.png")
vis.display_heatmaps(img_ids, save=True, fname=path + "/imgs/heatmaps")

print("producing figure random.png")
vis.random_latent_codes(4, True, path + "/imgs/random")

vis = Visualiser(path_of_run, device, training_type="Train", img_size=512)
print("producing teaser animation.gif")
vis.create_teaser_animation(
    219,
    "bunny",
    200,
    "teaser_animation",
    path + "/imgs/teaser_animation",
    is_ipython=is_ipython,
)

print("producing random interpolations animation.gif")
vis.interpolation_animation(10, None, 30, fname=path + "/imgs/interpolation_animation")

path_of_run = "/models/RENIVariationalAutoDecoder/network_128_5/ndims_9/"
vis = Visualiser(path_of_run, device, img_size=256)
print("producing animation of per parameter latent adjustment.gif")
vis.latent_per_param_adjustments(15, fname=path + "/imgs/latent_per_param_adjustment")

img_ids = [9, 4, 0, 7, 5, 20, 12]
# Loading a new model for environment completion, only 'ndims_36' model provided
path_of_run = (
    "/models/RENIVariationalAutoDecoder/network_128_5/ndims_36/files/in_painting/"
)
vis = Visualiser(
    path_of_run, device, training_type="EnvironmentCompletion", img_size=128
)
print("producing figure environment_completion.png")
vis.show_masked_output(img_ids, True, path + "/imgs/environment_completion")

img_ids = [1, 2, 3, 4, 5]
# Loading a new model for inverse rendering, only 'ndims_9' models provided
path_of_run = "/models/RENIVariationalAutoDecoder/network_128_5/ndims_9/"
vis = Visualiser(
    path_of_run, device, training_type="Inverse", kd_value=0.0, img_size=64
)
print("producing figure inverse_rendering.png")
vis.inverse_comparison(img_ids, True, path + "/imgs/inverse_rendering")

# Loading a new model for unnatural_images, only 'ndims_9' or 'ndims_36' models provided
path_of_run = (
    "models/RENIVariationalAutoDecoder/network_128_5/ndims_36/files/unnatural_images/"
)
vis = Visualiser(path_of_run, device, img_size=128)
img_ids = [10, 11, 5, 8, 2, 1, 0]
print("producing figure comparison_unnatural.png")
vis.comparison(img_ids, False, save=True, fname=path + "/imgs/comparison_unnatural")
