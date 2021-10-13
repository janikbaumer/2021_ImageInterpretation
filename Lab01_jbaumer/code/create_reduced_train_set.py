####################################################################
### Code only works if dataset_train_devel.h5 does not yet exist ###
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
N_IMAGES = 2
FILENAME_FULL = "../datasets/dataset_train.h5"
FILENAME_DEVEL = "../datasets/dataset_train_devel.h5"


if os.path.isfile(FILENAME_DEVEL):
      print(f'Any files in current directory with name {FILENAME_DEVEL} '
            f'must be either renamed or deleted !')
else:
      # read in train set (large)
      data_object_large = h5py.File(FILENAME_FULL,"r")

      # check keys of file object, to find which datasets are stored in this file
      print('keys of File object (train data): \n', data_object_large.keys(), '\n')

      # the following are not yet arrays, but of type "HDF5 dataset"
      dset_CLD = data_object_large['CLD']
      dset_GT = data_object_large['GT']
      dset_NIR = data_object_large['NIR']
      dset_RGB = data_object_large['RGB']

      """
      # these types have attributes .shape and .dtype
      print('shape dset_CLD: \n', dset_CLD.shape,
            '\ndtype dset_CLD:\n', dset_CLD.dtype, '\n')
      print('shape dset_GT: \n', dset_GT.shape,
            '\ndtype dset_GT:\n', dset_GT.dtype, '\n')
      print('shape dset_NIR: \n', dset_NIR.shape,
            '\ndtype dset_NIR:\n', dset_NIR.dtype, '\n')
      print('shape dset_RGB:\n', dset_RGB.shape,
            '\ndtype dset_RGB:\n', dset_RGB.dtype, '\n')
      """

      # They also support array-style slicing.
      dset_CLD_new = dset_CLD[0:N_IMAGES,:,:]  # index 0: n_images, index 1&2: x&y values of pixels
      dset_GT_new = dset_GT[0:N_IMAGES,:,:]
      dset_NIR_new = dset_NIR[0:N_IMAGES,:,:]
      dset_RGB_new = dset_RGB[0:N_IMAGES,:,:]



      # create a new file object from new datasets (with reduced number of imgs)
      with h5py.File(FILENAME_DEVEL, "a") as data_object_small:
            data_object_small['CLD'] = dset_CLD_new
            data_object_small['GT'] = dset_GT_new
            data_object_small['NIR'] = dset_NIR_new
            data_object_small['RGB'] = dset_RGB_new

      print('dataset with reduced size successfully saved to current directory ')