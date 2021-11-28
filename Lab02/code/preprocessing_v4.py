import h5py
import numpy as np
import matplotlib.pyplot as plt

def set_cloudy_nan(RGB, NIR, CLD):
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

def set_nodata_nan(RGB, NIR):

    # get pixels that have nodata over all four input channels
    RGBNIR = np.concatenate([RGB, np.expand_dims(NIR, axis=-1)], axis=-1)
    mask_nodata = (RGBNIR[:,:,0] == 0) & (RGBNIR[:,:,1] == 0) & (RGBNIR[:,:,2] == 0) & (RGBNIR[:,:,3] == 0)
    NIR_out = np.where((RGBNIR[:,:,0] == 0) & (RGBNIR[:,:,1] == 0) & (RGBNIR[:,:,2] == 0) & (RGBNIR[:,:,3] == 0), np.nan, NIR) #NIR[np.logical_not(mask_nodata)]
    RGB_out = np.ndarray(RGB.shape)
    for channel in range(RGB.shape[-1]):
        RGB_out[:, :, channel] = np.where((RGBNIR[:,:,0] == 0) & (RGBNIR[:,:,1] == 0) & (RGBNIR[:,:,2] == 0) & (RGBNIR[:,:,3] == 0), np.nan, RGB[:, :, channel])
    return RGB_out, NIR_out

def remove_outliers(RGB, NIR):
    # NIR
    NIR_std_dev = np.nanstd(NIR)
    NIR_mean = np.nanmean(NIR)

    # only limit upper bound, keep small values
    NIR = np.where(NIR > (NIR_mean+3*NIR_std_dev), np.nan, NIR)
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
    return RGB_norm, NIR_norm

FILEPATH_TRAIN = 'P:\pfshare\data\mikhailu\dataset_rgb_nir_train.hdf5'
FILEPATH_TEST = 'P:\pfshare\data\mikhailu\dataset_rgb_nir_test.hdf5'

DATA_TRAIN = h5py.File(FILEPATH_TRAIN, 'r')
DATA_TEST = h5py.File(FILEPATH_TEST, 'r')
TILES = [0, 1, 2, 3, 4, 5]
TILES_TRAIN = [1, 2, 3, 4]
TILES_TEST = [0, 5]
RGB_shape = (10980, 10980, 3)

for tile in TILES:
    if tile in TILES_TRAIN:
        train = True
    else:
        train = False

    print(f'tile: {tile} / training tile: {train}')

    if train:
        n_orbits = DATA_TRAIN[f'INPT_{tile}'].shape[0]
        merged_RGB = DATA_TRAIN[f'INPT_{tile}'][0]
        merged_NIR = DATA_TRAIN[f'NIR_{tile}'][0]
        if tile == 1:
            merged_GT = DATA_TRAIN['GT'][tile-1] # -1 because it has four GTs and we need to start at index 0 (else: out of bound error)
    else:
        n_orbits = DATA_TEST[f'INPT_{tile}'].shape[0]
        merged_RGB = DATA_TEST[f'INPT_{tile}'][0]
        merged_NIR = DATA_TEST[f'NIR_{tile}'][0]
        if tile == 0:
            merged_GT = DATA_TEST['GT'][0]
        else:  # tile is 5
            merged_GT = DATA_TEST['GT'][1]
    merged_GT = np.where(merged_GT == -1, np.nan, merged_GT)  # set -1 in labels (GT) to nan
    merged_RGB, merged_NIR = remove_outliers(merged_RGB, merged_NIR)

    for orbit in range(1, 2): #[1, 4]:  # iterate from 1 to n_orbits  # n_orbits:
        print(f'orbit: {orbit}')
        # todo: check if functions (masking, outlier) have to be called for previous and this (make sure not to run them twice if not needed)

        # work with this iteration plus the previous
        if train:
            CLD_previous = DATA_TRAIN[f'CLD_{tile}'][orbit-1, ...]
            CLD_this = DATA_TRAIN[f'CLD_{tile}'][orbit, ...]

            RGB_previous = merged_RGB
            RGB_this = DATA_TRAIN[f'INPT_{tile}'][orbit, ...]

            NIR_previous = merged_NIR
            NIR_this = DATA_TRAIN[f'NIR_{tile}'][orbit, ...]

        else:
            CLD_previous = DATA_TEST[f'CLD_{tile}'][orbit-1, ...]
            CLD_this = DATA_TEST[f'CLD_{tile}'][orbit, ...]

            RGB_previous = merged_RGB
            RGB_this = DATA_TEST[f'INPT_{tile}'][orbit, ...]

            NIR_previous = merged_NIR
            NIR_this = DATA_TEST[f'NIR_{tile}'][orbit, ...]

        # masking clouds
        RGB_previous_msk, NIR_previous_msk = set_cloudy_nan(RGB_previous, NIR_previous, CLD_previous)
        RGB_this_msk, NIR_this_msk = set_cloudy_nan(RGB_this, NIR_this, CLD_this)

        # masking no data in RGB and NIR
        RGB_previous_m, NIR_previous_m = set_nodata_nan(RGB_previous_msk, NIR_previous_msk)
        RGB_this_m, NIR_this_m = set_nodata_nan(RGB_this_msk, NIR_this_msk)

        # outlier removal for features
        RGB_previous_masked, NIR_previous_masked = remove_outliers(RGB_previous_m, NIR_previous_m)
        RGB_this_masked, NIR_this_masked = remove_outliers(RGB_this_m, NIR_this_m)

        RGB_previous_norm, NIR_previous_norm = normalize(RGB_previous_masked, NIR_previous_masked)
        RGB_this_norm, NIR_this_norm = normalize(RGB_this_masked, NIR_this_masked)

        # merging of previous and current image
        #RGB_merged, NIR_merged = merging(RGB_previous_masked, RGB_this_masked, NIR_previous_masked, NIR_this_masked)
        merged_RGB, merged_NIR = merging(RGB_previous_norm, RGB_this_norm, NIR_previous_norm, NIR_this_norm)

        # these objects will be used for the iteration of this loop (next orbit)

    #plt.imshow(merged_RGB)
    #plt.show()

    print(f'saving tile {tile}')
    # save to h5 (one file per tile)
    h5f = h5py.File(f'P:/pf/pfstud/jbaumer/tile_{tile}.h5', 'w')
    h5f.create_dataset('RGB', data=merged_RGB)
    h5f.create_dataset('NIR', data=merged_NIR)
    h5f.create_dataset('GT', data=merged_GT)
    h5f.close()

    print()
