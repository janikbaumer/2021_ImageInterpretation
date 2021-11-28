###
# goal of script:
#
# get one final GT per tile
#
# author: Janik Baumer, jbaumer@ethz.ch
###



import h5py
import numpy as np
import matplotlib.pyplot as plt


### FUNCTIONAS AND CLASSES ###
def concat_rgb_nir(arr_rgb, arr_nir):
    res = np.concatenate([arr_rgb, np.expand_dims(arr_nir, axis=-1)], axis=-1)
    return res

def set_cloudy_zero(RGB_NIR, CLD):
    ### masking cloudy pixels (set them to zero)
    # true if no cloud, false if cloud
    no_clouds = np.logical_not(CLD > 10)
    cloud_mask = np.stack([no_clouds, no_clouds, no_clouds, no_clouds], axis=-1)
    # set r, g, b, nir values to 0 if there is a cloud
    RGB_NIR[cloud_mask] = 0
    return RGB_NIR

def set_nolabel_zero(RGB_NIR, GT):
    no_label = GT == -1
    label_mask = np.stack([no_label, no_label, no_label, no_label], axis=-1)
    # set r, g, b, nir values to 0 if there is no label in GT
    RGB_NIR[label_mask] = 0
    return RGB_NIR


#def get_datasets(tile, DATA_TRAIN, DATA_TEST, str_cld, str_inpt, str_nir, str_gt):
def get_datasets(tile, DATA_TRAIN, DATA_TEST, str_gt):
    # train
    if tile in [1, 2, 3, 4]:
        #CLD_dset = DATA_TRAIN[str_cld]
        #INPT_dset = DATA_TRAIN[str_inpt]
        #NIR_dset = DATA_TRAIN[str_nir]
        GT_dset = DATA_TRAIN[str_gt]

    # test
    elif tile in [0, 5]:
        #CLD_dset = DATA_TEST[str_cld]
        #INPT_dset = DATA_TEST[str_inpt]
        #NIR_dset = DATA_TEST[str_nir]
        GT_dset = DATA_TEST[str_gt]
    #return CLD_dset, INPT_dset, NIR_dset, GT_dset
    return GT_dset


FILEPATH_TRAIN = 'P:\pfshare\data\mikhailu\dataset_rgb_nir_train.hdf5'
TRAIN_KEYS_CLD = ['CLD_0', 'CLD_5']
TRAIN_KEYS_INPT = ['INPT_0', 'INPT_5']
TRAIN_KEYS_NIR = ['NIR_0', 'NIR_5']
TRAIN_KEYS_GT = ['GT']

FILEPATH_TEST = 'P:\pfshare\data\mikhailu\dataset_rgb_nir_test.hdf5'
TEST_KEYS_CLD = ['CLD_1', 'CLD_2', 'CLD_3', 'CLD_4']
TEST_KEYS_INPT = ['INPT_1', 'INPT_2', 'INPT_3', 'INPT_4']
TEST_KEYS_NIR = ['NIR_1', 'NIR_2', 'NIR_3', 'NIR_4']
TEST_KEYS_GT = ['GT']

FILEPATH_RGB_CORR = 'C:\scratch\ImgInt_Lab02'

RNG = range(6)

DATA_TRAIN = h5py.File(FILEPATH_TRAIN, 'r')
DATA_TEST = h5py.File(FILEPATH_TEST, 'r')


# loop over 0 to 5
# 0, 4, 5: 30 orbits
# 1, 2, 3: 20 orbits
for tile in RNG:

    str_gt = f'GT'

    # create datasets
    GT_dset = get_datasets(tile, DATA_TRAIN, DATA_TEST, str_gt)
    for n_gt in GT_dset[0]:
        if n_gt == 0:
            GT_dset_stacked = GT_dset
        else:
            GT_dset_stacked = np.concatenate([GT_dset, GT_dset_stacked], axis=0)

    #range(GT_dset_stacked[0]):
    # for each stack, check if all -1 -> set to NaN
    for n in range(0, n_gt-1):
            if (GT[n+1] == -1) == (GT[n] == -1):
                if n = (n_gt - 1):
                    for layer in range(n_gt):
                        GT[n_gt == -1] = np.nan


    GT = GT_dset[0]
    np.save(f'C:\scratch\label_tile_{tile}', GT)

    print(f'saved tile {tile} successfully')

