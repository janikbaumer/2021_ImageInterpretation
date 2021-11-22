import h5py
import numpy as np
import matplotlib.pyplot as plt

def set_cloudy_zero(RGB, NIR, CLD):
    ### masking cloudy pixels (set them to NaN)

    # true if no cloud, false if cloud
    no_clouds = CLD < 10

    no_cloud_mask_RGB = np.stack([no_clouds, no_clouds, no_clouds], axis=-1)
    no_cloud_mask_NIR = no_clouds

    # set r, g, b, nir values to nan if there is a cloud
    #RGB[cloud_mask_RGB] = np.nan
    RGB = np.where(no_cloud_mask_RGB, RGB, np.nan)
    NIR = np.where(no_cloud_mask_NIR, NIR, np.nan)

    return RGB, NIR


def remove_outliers(RGB, NIR):
    # NIR
    NIR_std_dev = np.nanstd(NIR)
    NIR_mean = np.nanmean(NIR)

    # only limit upper bound, keep small values
    NIR = np.where(NIR > (NIR_mean+2.5*NIR_std_dev), np.nan, NIR)
    #NIR = np.where(NIR < (NIR_mean-3*NIR_std_dev), np.nan, NIR)

    # same for RGB (separately per channel)
    for channel in range(RGB.shape[-1]):
        # RGB
        RGB_std_dev = np.nanstd(RGB[:, :, channel])
        RGB_mean = np.nanmean(RGB[:, :, channel])

        # only limit upper bound, keep small values
        RGB = np.where(RGB > (RGB_mean + 3 * RGB_std_dev), np.nan, RGB)

    return RGB, NIR


def merging(RGB_previous_masked, RGB_this_masked, NIR_previous_masked, NIR_this_masked):
    # calculate mean for previous and current NIR image (separately for each channel)
    NIR_stack = np.stack([NIR_previous_masked, NIR_this_masked], axis=-1)
    NIR_merged = np.nanmean(NIR_stack, axis=-1)

    # calculate mean for previous and current RGB image (separately for each channel)
    RGB_merged = np.ndarray(RGB_shape)
    for channel in range(RGB_previous_masked.shape[-1]):
        RGB_chnl_stack = np.stack([RGB_previous_masked[:, :, channel], RGB_this_masked[:, :, channel]], axis=-1)
        RGB_chnl_mean = np.nanmean(RGB_chnl_stack, axis=-1)
        RGB_merged[:,:,channel] = RGB_chnl_mean
        #print()
    return RGB_merged, NIR_merged

def normalize(RGB, NIR):
    max_rgb = np.nanmax(RGB)
    min_rgb = np.nanmin(RGB)  # 0 in most cases
    max_nir = np.nanmax(NIR)
    min_nir = np.nanmin(NIR)  # 0 in most cases

    RGB_norm = (RGB-min_rgb)/(max_rgb-min_rgb)
    NIR_norm = (NIR-min_nir)/(max_nir-min_nir)
    print()
    return RGB_norm, NIR_norm

FILEPATH_TRAIN = 'P:\pfshare\data\mikhailu\dataset_rgb_nir_train.hdf5'
FILEPATH_TEST = 'P:\pfshare\data\mikhailu\dataset_rgb_nir_test.hdf5'

DATA_TRAIN = h5py.File(FILEPATH_TRAIN, 'r')
DATA_TEST = h5py.File(FILEPATH_TEST, 'r')
TILES_TRAIN = [1,2,3,4]
TILES_TEST = [0,5]
RGB_shape = (10980, 10980, 3)

for tile in [1]: #TILES_TRAIN:
    print(f'tile: {tile}')
    n_orbits = DATA_TRAIN[f'INPT_{tile}'].shape[0]

    merged_RGB = DATA_TRAIN[f'INPT_{tile}'][0]
    merged_NIR = DATA_TRAIN[f'NIR_{tile}'][0]
    merged_RGB, merged_NIR = remove_outliers(merged_RGB, merged_NIR)
    #merged_RGB, merged_NIR = normalize(merged_RGB, merged_NIR)
    #plt.imshow(merged_RGB)
    #plt.show()
    #print()

    merged_GT = DATA_TRAIN['GT'][tile]
    merged_GT = np.where(merged_GT == -1, np.nan, merged_GT)  # set -1 in labels (GT) to nan

    for orbit in range(1,2): #[1, 4]:  # iterate from 1 to n_orbits  # n_orbits:
        print(f'orbit: {orbit}')
        # todo: check if functions (masking, outlier) have to be called for previous and this (make sure not to run them twice if not needed)

        # work with this iteration plus the previous
        CLD_previous = DATA_TRAIN[f'CLD_{tile}'][orbit-1, ...]
        CLD_this = DATA_TRAIN[f'CLD_{tile}'][orbit, ...]

        RGB_previous = merged_RGB
        RGB_this = DATA_TRAIN[f'INPT_{tile}'][orbit, ...]

        NIR_previous = merged_NIR
        NIR_this = DATA_TRAIN[f'NIR_{tile}'][orbit, ...]

        # masking clouds
        RGB_previous_msk, NIR_previous_msk = set_cloudy_zero(RGB_previous, NIR_previous, CLD_previous)
        RGB_this_msk, NIR_this_msk = set_cloudy_zero(RGB_this, NIR_this, CLD_this)

        # outlier removal for features
        RGB_previous_masked, NIR_previous_masked = remove_outliers(RGB_previous_msk, NIR_previous_msk)
        RGB_this_masked, NIR_this_masked = remove_outliers(RGB_this_msk, NIR_this_msk)

        RGB_previous_norm, NIR_previous_norm = normalize(RGB_previous_masked, NIR_previous_masked)
        RGB_this_norm, NIR_this_norm = normalize(RGB_this_masked, NIR_this_masked)

        # merging of previous and current image
        #RGB_merged, NIR_merged = merging(RGB_previous_masked, RGB_this_masked, NIR_previous_masked, NIR_this_masked)
        merged_RGB, merged_NIR = merging(RGB_previous_norm, RGB_this_norm, NIR_previous_norm, NIR_this_norm)

        # these objects will be used for the iteration of this loop (next orbit)
        # merged_RGB, merged_NIR = normalize(RGB_merged, NIR_merged)  # normalized to value between zero and one

    plt.imshow(merged_RGB)
    plt.show()
    print()