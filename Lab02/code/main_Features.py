# %%
### IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import os
import cv2
import pickle as cPickle
import platform

from tqdm import tqdm
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from datetime import datetime
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from numpy import ma

# %%
### CLASSES

from SatelliteSet import SatelliteSet

# %%
### FUNCTIONS

# %%
### INITIALIZING
# Defining variables for future use
## THIS SCRIPT IS FOR FEATURE EXTRACTION

# Initializing file paths, taking os into account

current_os = platform.system()
if current_os == 'Darwin':
    cwd = '..'
    sep = '/'
elif current_os == 'Windows':
    cwd = os.getcwd()
    sep='\\'

DATA_SET = "train" #Name of input image. IDEA: Send one image at a time, be it test or train. Means running code 4 times for training and 2 for test. Attempt to reduce memory usage
FEAT_SET = "features_" + DATA_SET + ".h5"

DATA_SET = "dataset_" + DATA_SET + ".h5"
DATA_FILE = cwd+sep+'datasets'+sep+DATA_SET
FEAT_FILE = cwd+sep+FEAT_SET

# %% Data Loading
if __name__ == "__main__":

    # dealing with division by nan or 0
    np.seterr(divide='ignore', invalid='ignore')

    # create datasets
    dset = SatelliteSet(root=DATA_FILE, windowsize=windSize, test=False)

    # TODO: Load the pretrained-NN here, modify parameters
    model = VGG16(weights='imagenet', include_top=False)

    """
    # create dataloader that samples batches from the dataset
    train_loader = DataLoader(dset,
                              batch_size=batchSize,
                              num_workers=0,
                              shuffle=False)

    """


    """
    Load image into a feature vector x (shape depending on model in use)
    """

    # TODO: Do we do feature extraction on the whole image at once? LETS TRY
    print('STARTING FEATURE EXTRACTION ...')
    features = model.predict(x) #Returns the features as a np array.
    #This still needs to be modified to have the parameters. See documentation. For example, predict with batches or not?

# %% WRITING FEATURES TO HDF5 FILE

    # TODO: Here the feature vectors should be stored and saved, in order to retrieve it later for the training
    #   You need to figure out how the NN outputs the features (probably as np arrays), and then you need to figure
    #   how to save and store them. Make sure to consider how they will be retrieved for training.
    #   QUESTION: do we mask before or after storage? The problem might be when storing everything together that
    #   every batch has different feature vector lengths, because of varying number of good pixels. Solution might
    #   be to add the mask before training.

    h5f_feat = h5py.File(FEAT_FILE, 'w')
    h5f_feat.create_dataset('feature_vectors', data=features)
    h5f_feat.close()

    """
    IN CASE WE WANT TO CREATE A CSV FILE INSTEAD
    np.savetxt('features_'+DATA_SET+'.csv', features, delimiter=',', newline='n'') #Consider using '.gz' for compressed file.
    """

# %%
"""
    train_loader_loop = 0

    for x, y in tqdm(train_loader):  # tqdm: make loops show a smart progress meter by wrapping any iterable with tqdm(iterable)
        train_loader_loop += 1
        # print('train loader loop: ', train_loader_loop)
        x = np.transpose(x, [0, 2, 3,
                             1])  # swap shapes so that afterward shape = (nmbr_imgs_in_batch, size_x, size_y, nmbr_channels)
        x = x.numpy()  # x is not yet ndarray - convert x from pytorch tensor to ndarray

        # change no data (99) to (3), for plotting reasons
        y = np.where(y == 99, 3, y)  # y is already ndarry

        # loop over batch size of train_loader ((all images in this batch (?))
        for i in range(len(x)):

            x_batch = x[i]
            y_batch = y[i]

            # nodata_mask = y_batch!=3
            # x_batch = x_batch[nodata_mask, :]
            # y_batch = y_batch[nodata_mask]

            # if y batch only contains no data (99, resp. 3)
            # so after masking it's an empty array
            # if y_batch.shape != (0,):

            # FEATURE EXTRACTION
            # Grayscale as feature, adds 1 feat
            x_batch_gray = cv2.cvtColor(x_batch[:, :, 0:3], cv2.COLOR_RGB2GRAY)
            x_batch = np.dstack((x_batch, x_batch_gray))

            # NDVI, adds 1 feat
            x_batch_ndvi = calc_ndvi(x_batch[:, :, 3], x_batch[:, :, 0])
            x_batch = np.dstack((x_batch, x_batch_ndvi))

            # Sobel axis 1
            sobel_axis0 = ndimage.sobel(x_batch[:, :, 4], axis=0)
            x_batch = np.dstack((x_batch, sobel_axis0))

            # Sobel axis 2
            sobel_axis1 = ndimage.sobel(x_batch[:, :, 4], axis=1)
            x_batch = np.dstack((x_batch, sobel_axis1))

            # Mean pixel intensity, all four original intensities, adds 1 feat
            x_batch_meanpix = mean_pix(x_batch[:, :, 0:4])
            x_batch = np.dstack((x_batch, x_batch_meanpix))

            # Mean pixel intensity, RGB, adds 1 feat
            x_batch_meanpixRGB = mean_pix(x_batch[:, :, 0:3])
            x_batch = np.dstack((x_batch, x_batch_meanpixRGB))

            # Prewitt horizontal edges, adds 1 feat
            edges_prewitt_horizontal = prewitt_h(x_batch[:, :, 4])
            x_batch = np.dstack((x_batch, edges_prewitt_horizontal))

            # Prewitt vertical edges, adds 1 feat
            edges_prewitt_vertical = prewitt_v(x_batch[:, :, 4])
            x_batch = np.dstack((x_batch, edges_prewitt_vertical))

            # Window-level scaling/shifting, adds 1 feat
            x_batch_wl = window_level_function(x_batch[:, :, 4], 0.8)
            x_batch = np.dstack((x_batch, x_batch_wl))

            # convert nan values to 0
            x_batch = np.nan_to_num(x_batch)

            # could be commented
            # nans = np.any(np.isnan(x_batch))
            # print(nans)
            # print()

            # define shapes
            x_shape = x_batch.shape
            y_shape = y_batch.shape

            # initialize lists in which features / labels are stored
            X_lst = []
            Y_lst = []

            # create feature matrix X_train (can be used for ML model later)
            # for each channel, stack features (e.g. R,G,B,NIR intensities) to column vector
            for chn in range(x_shape[-1]):
                x_batch_chn = x_batch[:, :, chn]
                x_batch_chn = np.resize(x_batch_chn, (x_shape[0] * x_shape[1], 1))
                x_shape_resized = x_batch_chn.shape
                X_lst.append(x_batch_chn)

            y_batch.resize(y_shape[0] * y_shape[1], 1)
            Y_lst.append(y_batch)

            # define feature matrix and label vector
            X_train = np.array(X_lst).T[
                0]  # transpose to have one col per feature, first ele because this dimension was somehow added
            Y_train = np.array(Y_lst).T[
                0].ravel()  # same as above, plus ravel() to convert from col vector to 1D array (needed for some ML models)

            # do masking only if there is no_data
            if np.max(Y_train) == 3:
                Y_train_masked = ma.masked_where(Y_train == 3, Y_train)
                Y_train_masked_compressed = Y_train_masked.compressed()
                n_rows = int(Y_train_masked_compressed.shape[0])

                X_train_masked = ma.zeros(X_train.shape)
                mask_train = Y_train_masked.mask

                # mask each column of X_train and replace X_train column with its masked corresponding
                for col in range(X_train[:].shape[1]):
                    col_vec = X_train[:, col]
                    col_vec_masked = ma.masked_array(col_vec, mask=mask_train)
                    X_train_masked[:, col] = col_vec_masked.compressed().resize((n_rows, 1))
                # stack masked cols together
                # no_data = np.where(Y_train == 3)

                # X_train = X_train_masked
                # Y_train = Y_train_masked

                Y_train = Y_train_masked_compressed
                X_train = X_train_masked[~np.isnan(X_train_masked)]

            # if mask_train.all():
            # break

            # X_train = X_train_flat.
            if not X_train.shape == (0,) and not Y_train.shape == (0,):
                # normalizing and standardizing the feature vectors
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)

 """


