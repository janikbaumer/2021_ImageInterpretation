###
# goal of script:
# read in images from 6 different tiles (multiple per tile)
# for each time the satellite passed this tile, aggregate imgs together and create one image per tile
# save one img per tile as .h5 file

# note:
# values of inpt: [1, 18892] -> vals (prob 16 bit)
# values of cld: [0, 100] -> probability of being a cloud
# values of nir: [1, 16236] -> vals (prob 16 bit)
# values of gt: [-1, 60] -> canopy height at given pixel (-1: no data)
#
# further notes:
# cloud masks, 100 means cloud, 0 means to cloud -> define threshold at 10 (all prob > 10 is cloud)
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


def get_datasets(tile, DATA_TRAIN, DATA_TEST, str_cld, str_inpt, str_nir, str_gt):
    # train
    if tile in [1, 2, 3, 4]:
        CLD_dset = DATA_TRAIN[str_cld]
        INPT_dset = DATA_TRAIN[str_inpt]
        NIR_dset = DATA_TRAIN[str_nir]
        GT_dset = DATA_TRAIN[str_gt]

    # test
    elif tile in [0, 5]:
        CLD_dset = DATA_TEST[str_cld]
        INPT_dset = DATA_TEST[str_inpt]
        NIR_dset = DATA_TEST[str_nir]
        GT_dset = DATA_TEST[str_gt]
    return CLD_dset, INPT_dset, NIR_dset, GT_dset


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

'''
# plotting
f, axarr = plt.subplots(1, 2)
axarr[0].imshow(DATA_TEST[TEST_KEYS_INPT[1]][0][10_000:11_000, 0:1_000])
axarr[1].imshow(DATA_TEST[TEST_KEYS_CLD[1]][0][10_000:11_000, 0:1_000])
plt.show()
'''

# loop over all images
# for each image, compare with
#  ... corresponding cloud mask -> mask these values
#  ... corresponding no values -> mask these values

# loop over 0 to 5
# 0, 4, 5: 30 orbits
# 1, 2, 3: 20 orbits
for tile in RNG:

    str_cld = f'CLD_{tile}'
    str_inpt = f'INPT_{tile}'
    str_nir = f'NIR_{tile}'
    str_gt = f'GT'

    # create datasets
    CLD_dset, INPT_dset, NIR_dset, GT_dset = \
        get_datasets(tile, DATA_TRAIN, DATA_TEST, str_cld, str_inpt, str_nir, str_gt)

    n_orbits = CLD_dset.shape[0]

    # loop over 20 or 30 imgs in the respective dataset
    for orbit in range(n_orbits):
        print('orbit', orbit)

        # get images / cloud masks
        CLD = CLD_dset[orbit]
        INPT = INPT_dset[orbit]
        NIR = NIR_dset[orbit]
        GT = GT_dset[0]
        '''
        # normalize images (8 or 16 bit imgs)
        INPT = np.asarray(INPT, np.float32) / (2 ** 16 - 1)
        NIR = np.asarray(NIR, np.float32) / (2 ** 16 - 1)
        '''

        # stack rgb and nir images
        RGB_NIR = concat_rgb_nir(INPT, NIR)
        RGB_NIR = set_nolabel_zero(RGB_NIR, GT)
        RGB_NIR = set_cloudy_zero(RGB_NIR, CLD)
        # set_no_data_zero # already implicitly given

        if orbit == 0:  # first loop
            RGB_NIR_new = np.empty((n_orbits, 10980, 10980, 4), dtype='float16')


        RGB_NIR_new[orbit, :, :, :] = RGB_NIR

    # dump RGB_NIR_new array to file
    np.save(f'{FILEPATH_RGB_CORR}_tile_{tile}', RGB_NIR_new)
    # RGB_NIR_new.dump(f'{FILEPATH_RGB_CORR}_tile_{tile}')

    print(f'saved tile {tile}')
    print()
