### imports
import h5py
import numpy as np
import matplotlib.pylab as plt



### FUNCTIONS
def create_label(CLD,GT,input_image):

    # first img - use index 0
    image_number = 0

    # Assign values 99 (no data) to 3 (for visualization only)
    first_gt_image = np.where(GT[image_number] == 99, 3, GT[image_number])

    #plt.imshow(first_gt_image)
    #plt.show()

    # assign clouds (label 2) based on following condition
    # (threshold 10 choosable - fixed for comparability among groups),
    # to avoid that parts that should be clouds are assigned as palm oil trees
    cloud_positions = np.where(CLD[image_number] > 10)
    first_gt_image[cloud_positions] = 2

    #plt.imshow(first_gt_image)
    #plt.show()

    # remove parts that do not contain data in input image
    # (they also should not contain data in label image)
    idx = np.where(np.max(input_image[0], axis=-1) == 0)
    first_gt_image[idx] = 3
    label_image = first_gt_image

    #plt.imshow(first_gt_image)
    #plt.show()

    # Finally we can draw a small input window and the corresponding label data
    #f, axarr = plt.subplots(ncols=3, nrows=1)
    #axarr[0].imshow(input_image[0, 384:628, 384:628, :3])  # RGB
    #axarr[1].imshow(input_image[0, 384:628, 384:628, -1])  # NIR
    #axarr[2].imshow(first_gt_image[384:628, 384:628])
    #plt.show()

    return label_image


def create_features():
    pass


### VARIABLES

# for real training, change FILE_TRAIN to ../datasets/dataset_train.h5
FILE_TRAIN = '../datasets/dataset_train_devel.h5'
FILE_TEST = '../datasets/dataset_test.h5'



### IMPORT

#Gain access to the data.
#Note: This does *not* load the entire data set into memory.
dset_train = h5py.File(FILE_TRAIN,"r")
dset_test = h5py.File(FILE_TEST, "r")

#The array GT contains the values 0 = background, 1 = palm oil and 99 = no data.


RGB_train = dset_train["RGB"]  # type: #HDF5 dataset / shape (2, 10980, 10980, 3)
NIR_train = dset_train["NIR"]
CLD_train = dset_train["CLD"]
GT_train = dset_train["GT"]

RGB_test = dset_test["RGB"]
NIR_test = dset_test["NIR"]
CLD_test = dset_test["CLD"]
GT_test = dset_test["GT"]

print('keys of dset_train: ', dset_train.keys())
print('keys of dset_test: ', dset_test.keys())

#Let's create an input-label pair:
#first the input by concatenating the RGB and NIR channels.

# expand NIR image in last dimension
NIR_train_expanded = np.expand_dims(NIR_train, axis=-1)
input_image_train = np.concatenate([RGB_train, NIR_train_expanded], axis=-1)

NIR_test_expanded = np.expand_dims(NIR_test, axis=-1)
input_image_test = np.concatenate([RGB_test, NIR_test_expanded], axis=-1)

print(np.shape(input_image_train))
print(type(input_image_train))



### GET FEATURES AND LABELS

# todo: so far, label is only created with one image
#  create with each image in dataset and stack them together


X_train = create_features(RGB_train, RGB_test, input_image_train)


Y_train = create_label(CLD_train, GT_train, input_image_train)

Y_test = create_label(CLD_test, GT_test, input_image_test)


### PREPROCESSING DATA


# ADDING AN NDVI CHANNEL

# FURTHER PREPROCESSING (FEATURE EXTRACTION)


### EVALUATION

# CREATE CONFUSION MATRIX

print(Y_train)
print(Y_test)
print()