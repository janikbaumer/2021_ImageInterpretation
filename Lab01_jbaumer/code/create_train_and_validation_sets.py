####################################################################
### Code only works if dataset_ does not yet exist ###
####################################################################


import h5py
import numpy as np
import os.path

"""
Note:
An HDF5 file is a container for two kinds of objects: 
datasets, which are array-like collections of data
and groups, which are folder-like containers that hold datasets and other groups.
The most fundamental thing to remember when using h5py is:
Groups work like dictionaries, and datasets work like NumPy arrays
"""

####################################################################
### Code only works if dataset_train_devel.h5 does not yet exist ###
####################################################################


# VARIABLES
N_IMAGES_TRAIN = 60
N_IMAGES_VAL = 4
FILENAME_TRAIN_FULL = "../datasets/dataset_train.h5"
FILENAME_DEVEL = "../datasets/dataset_train_devel.h5"

FILENAME_VAL = "../datasets/dataset_val.h5"
FILENAME_TRAIN = "../datasets/dataset_train_reduced.h5"

if os.path.isfile(FILENAME_VAL):
      print(f'Any files in current directory with name {FILENAME_VAL} '
            f'must be either renamed or deleted !')
else:
      # read in train set (large)
      data_object_large = h5py.File(FILENAME_TRAIN_FULL,"r")

      # check keys of file object, to find which datasets are stored in this file
      print('keys of File object (train data): \n', data_object_large.keys(), '\n')

      # the following are not yet arrays, but of type "HDF5 dataset"
      dset_CLD = data_object_large['CLD']
      dset_GT = data_object_large['GT']
      dset_NIR = data_object_large['NIR']
      dset_RGB = data_object_large['RGB']



      # They also support array-style slicing.

      # get validation dsets
      dset_CLD_val = dset_CLD[N_IMAGES_TRAIN:N_IMAGES_TRAIN + N_IMAGES_VAL, :,:]  # index 0: n_images, index 1&2: x&y values of pixels
      dset_GT_val = dset_GT[N_IMAGES_TRAIN:N_IMAGES_TRAIN + N_IMAGES_VAL, :, :]
      dset_NIR_val = dset_NIR[N_IMAGES_TRAIN:N_IMAGES_TRAIN + N_IMAGES_VAL, :, :]
      dset_RGB_val = dset_RGB[N_IMAGES_TRAIN:N_IMAGES_TRAIN + N_IMAGES_VAL, :, :]

      # get train dsets
      dset_CLD_train = dset_CLD[0:N_IMAGES_TRAIN,:,:]  # index 0: n_images, index 1&2: x&y values of pixels
      dset_GT_train = dset_GT[0:N_IMAGES_TRAIN,:,:]
      dset_NIR_train = dset_NIR[0:N_IMAGES_TRAIN,:,:]
      dset_RGB_train = dset_RGB[0:N_IMAGES_TRAIN,:,:]


      # create a new file object from new datasets (with reduced number of imgs) - train ds
      with h5py.File(FILENAME_TRAIN, "a") as data_object_small:
            data_object_small['CLD'] = dset_CLD_train
            data_object_small['GT'] = dset_GT_train
            data_object_small['NIR'] = dset_NIR_train
            data_object_small['RGB'] = dset_RGB_train

      # create a new file object from new datasets (with reduced number of imgs) - val ds
      with h5py.File(FILENAME_VAL, "a") as data_object_small:
            data_object_small['CLD'] = dset_CLD_val
            data_object_small['GT'] = dset_GT_val
            data_object_small['NIR'] = dset_NIR_val
            data_object_small['RGB'] = dset_RGB_val

      print('dataset with reduced size successfully saved to current directory ')