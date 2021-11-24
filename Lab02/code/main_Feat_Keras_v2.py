# %% IMPORTS
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import csv

import platform
from time import time

from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

from numpy import ma

# %% CLASSES
# Repurposed dataset class from the first lab
class SatelliteSetLab02(VisionDataset):

    # test flag: whether data is loaded completely into memory
    def __init__(self, root="../datasets/dataset_train.h5", windowsize=128, test=False):

        super().__init__(root)

        self.wsize = windowsize
        if test:
            h5 = h5py.File(root, 'r', driver="core")  # Store the data in memory
        else:
            h5 = h5py.File(root, 'r')

        self.RGB = h5["RGB"]
        self.NIR = h5["NIR"]
        # self.CLD = h5["CLD"]
        self.GT = h5["GT"]

        if len(self.GT.shape) == 2:
            self.GT = np.expand_dims(self.GT, axis=0)
            self.RGB = np.expand_dims(self.RGB, axis=0)
            self.NIR = np.expand_dims(self.NIR, axis=0)

        self.num_smpls, self.sh_x, self.sh_y = self.GT.shape  # size of each image

        self.pad_x = (self.sh_x - (self.sh_x % self.wsize))
        self.pad_y = (self.sh_y - (self.sh_y % self.wsize))
        self.sh_x = self.pad_x + self.wsize
        self.sh_y = self.pad_y + self.wsize
        self.num_windows = self.num_smpls * self.sh_x / self.wsize * self.sh_y / self.wsize
        self.num_windows = int(self.num_windows)

    def __getitem__(self, index):
        """Returns a data sample from the dataset.
        """
        # determine where to crop a window from all images (no overlap)
        m = index * self.wsize % self.sh_x  # iterate from left to right
        # increase row by windows size everytime m increases
        n = (int(np.floor(index * self.wsize / self.sh_x)) * self.wsize) % self.sh_x
        # determine which batch to use
        b = (index * self.wsize * self.wsize // (self.sh_x * self.sh_y)) % self.num_smpls

        # crop all data at the previously determined position
        RGB_sample = self.RGB[b, n:n + self.wsize, m:m + self.wsize]
        NIR_sample = self.NIR[b, n:n + self.wsize, m:m + self.wsize]
        # CLD_sample = self.CLD[b, n:n + self.wsize, m:m + self.wsize]
        GT_sample = self.GT[b, n:n + self.wsize, m:m + self.wsize]

        # normalize NIR and RGB by maximum possible value
        """
        NIR_sample = np.asarray(NIR_sample, np.float32) / (2 ** 16 - 1)
        RGB_sample = np.asarray(RGB_sample, np.float32) / (2 ** 8 - 1)
        """

        X_sample = np.concatenate([RGB_sample, np.expand_dims(NIR_sample, axis=-1)], axis=-1)

        """
        ### correct gt data ###
        # first assign gt at the positions of clouds
        cloud_positions = np.where(CLD_sample > 10)
        GT_sample[cloud_positions] = 2
        # second remove gt where no data is available - where the max of the input channel is zero
        idx = np.where(np.max(X_sample, axis=-1) == 0)  # points where no data is available
        GT_sample[idx] = 99  # 99 marks the absence of a label and it should be ignored during training
        GT_sample = np.where(GT_sample > 3, 99, GT_sample)
        """

        # pad the data if size does not match
        sh_x, sh_y = np.shape(GT_sample)
        pad_x, pad_y = 0, 0
        if sh_x < self.wsize:
            pad_x = self.wsize - sh_x
        if sh_y < self.wsize:
            pad_y = self.wsize - sh_y

        x_sample = np.pad(X_sample, [[0, pad_x], [0, pad_y], [0, 0]])
        gt_sample = np.pad(GT_sample, [[0, pad_x], [0, pad_y]], 'constant',
                           constant_values=[np.nan])  # pad with NaN to mark absence of data

        # pytorch wants the data channel first - you might have to change this
        x_sample = np.transpose(x_sample, (2, 0, 1))
        return np.asarray(x_sample), gt_sample

    def __len__(self):
        return self.num_windows


# %% FUNCTIONS

def plot_patches(X, Y, num):
    # X_shape = X.shape()
    # Y_shape = Y.shape()
    fig, axs = plt.subplots(num=num, nrows=2, ncols=X.shape[0],
                            figsize=(30, 10), edgecolor='black',
                            tight_layout=True)
    fig.suptitle('Plotted RGB and GT of patches from one batch', fontsize=28)
    for patch in range(len(X)):
        # Plot the RGB images of the batch
        ax1 = axs[0, patch]
        ax2 = axs[1, patch]
        ax1.imshow(X[patch, :, :, 0:3])
        ax1.set_title(f'RGB patch: {patch}', fontsize=14)
        # Plot the GT canopy heights
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)  # Appending an axis for the color bar to the right
        im = ax2.imshow(Y[patch, ...], cmap='inferno')  # Plotting GT in subplot
        ax2.set_title(f'GT patch: {patch}', fontsize=14)  # Giving subplot title
        cbar = fig.colorbar(im, cax=cax, orientation='vertical',
                            ticks=[0.1, 30, 59.9])  # Creating colorbar on the appended axis
        cbar.set_label('True Canopy Height [m]', rotation=90)  # Label the colorbar
        cbar.ax.set_yticklabels(['0', '30', '60'])  # Set colorbar tick labels
    plt.show()


# %% SCRIPT HEADER
### INITIALIZING
# Defining variables
## THIS SCRIPT IS FOR FEATURE EXTRACTION

# Initializing file paths, taking os into account
sep = '\\'
current_os = platform.system()
if current_os == 'Darwin':
    cwd = '..'
    sep = '/'
elif current_os == 'Windows':
    cwd = os.getcwd()
    sep = '\\'

# FILE HIERARCHY INSTRUCTIONS Feature Extraction
#   ...
#       -\datasets
#           -\train
#               -merged_img_tile0.npy
#               -merged_img_tile1.npy
#               -merged_img_tile2.npy
#           -\val
#               -merged_img_tile3.npy
#           -\test
#               -merged_img_tile4.npy
#               -merged_img_tile5.npy
#           -\dev
#               -dev_lab02.h5

# Define root path for data folder
ROOT_PATH = r'C:\Users\Jor Fergus Dal\Desktop\Lav02_tiles_v1'

DATA_SET = "val"  # What dataset do you want to process. Option: "train", "val" & "test"
DATA_FOLDER = ROOT_PATH + sep + 'datasets' + sep + DATA_SET

# Variable and hyperparameters
file_type = '.h5'
windSize = 128
batchSize = 8


# Create the base model ofr feature extraction
feat_extractor = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(windSize, windSize, 3),
    pooling="avg"
)


feat_extractor.trainable = False

# Tile names:
TILES: Dict[str, List[int]] = {
    'train': [1, 2, 3],
    'val': [4],
    'test': [0, 5],
    'dev': [99]
}

# Features by tile, stacked along axis=0
Z_tiles = {
    0: np.array([1, 2]),  # placeholder array
    1: np.array([1, 2]),  # placeholder array
    2: np.array([1, 2]),  # placeholder array
    3: np.array([1, 2]),  # placeholder array
    4: np.array([1, 2]),  # placeholder array
    5: np.array([1, 2]),  # placeholder array
    99: np.array([1, 2]) # placeholder array
}

# GT by tile, stacked along axis=0
Y_tiles = {
    0: np.array([1, 2]),  # placeholder array
    1: np.array([1, 2]),  # placeholder array
    2: np.array([1, 2]),  # placeholder array
    3: np.array([1, 2]),  # placeholder array
    4: np.array([1, 2]),  # placeholder array
    5: np.array([1, 2]),  # placeholder array
    99: np.array([1, 2])  # placeholder array
}

# Number of load loops
num_loadLoops = []

# %% Data Loading
if __name__ == "__main__":

    # dealing with division by nan or 0
    np.seterr(divide='ignore', invalid='ignore')

    # Time before feature extraction
    t_0 = time()

    directory = DATA_FOLDER
    file_num = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(file_type) and not (filename.endswith('features'+file_type)):

            DATA_FILE = DATA_FOLDER + sep + filename

            # create datasets
            dset = SatelliteSetLab02(root=DATA_FILE, windowsize=windSize, test=False)
            # create data loader
            loader = DataLoader(dset, batch_size=batchSize, num_workers=0, shuffle=False, drop_last=True)

            load_loop = 0
            # load batch
            for X, Y in tqdm(loader):
                print(f'\nTile number: {TILES[DATA_SET][file_num]}')  # For traceability
                print(f'Load loop: {load_loop}')  # For traceability

                # Load Data
                X = np.transpose(X, [0, 2, 3,
                                     1])  # swap shapes so that afterward shape = (nmbr_imgs_in_batch, size_x, size_y, nmbr_channels)
                X = X.numpy()  # load the tensor object into a numpy array
                Y = Y.numpy()  # load the tensor object into a numpy array

                # plot RGB and GT of a particular batch for visualisation
                if load_loop == 400:
                    plot_patches(X, Y, file_num)

                # Integrating NIR in RGB channels to reduce channels to 3
                meanRNIR = np.nanmean(np.array([X[:, :, :, 0], X[:, :, :, 3]]), axis=0)
                meanGNIR = np.nanmean(np.array([X[:, :, :, 1], X[:, :, :, 3]]), axis=0)
                meanBNIR = np.nanmean(np.array([X[:, :, :, 2], X[:, :, :, 3]]), axis=0)

                X = np.stack([meanRNIR, meanGNIR, meanBNIR], axis=-1)

                # Replace NaN values in images by 0
                np.nan_to_num(X, copy=False, nan=0.0, posinf=None, neginf=None)

                patch_shape = X.shape[1:-1]

                # Feature extraction
                print('STARTING FEATURE EXTRACTION ...')
                Z = feat_extractor.predict(X)  # Returns the features as a np array.

                # TODO: Reshape features, relate them to ground truth and store them with index.
                #  Remember to aggregate before saving.
                Z_shape = Z.shape
                Y_shape = Y.shape

                Y = np.reshape(Y, (batchSize, -1))

                # append features vectors and GT along rows
                if load_loop == 0:
                    Z_appd = Z
                    Y_appd = Y
                else:
                    Z_appd = np.append(Z_appd, Z, axis=0)
                    Y_appd = np.append(Y_appd, Y, axis=0)

                load_loop += 1

            Z_tiles[TILES[DATA_SET][file_num]] = Z_appd
            Y_tiles[TILES[DATA_SET][file_num]] = Y_appd
            num_loadLoops.append(load_loop)
            file_num += 1
            continue
        else:
            continue

    Z_output = np.empty((2,2))
    Z_output[:] = np.nan
    Y_output = np.empty((2,2))
    Y_output[:] = np.nan

    # Saving batchwise features and GT to drive
    for tile in TILES[DATA_SET]:
        if Y_output.shape == (2,2) and Z_output.shape == (2,2):
            Z_output = Z_tiles[tile]
            Y_output = Y_tiles[tile]
        else:
            Z_output = np.append(Z_output, Z_tiles[tile], axis=0)
            Y_output = np.append(Y_output, Y_tiles[tile], axis=0)
    FEAT_FILE = DATA_FOLDER + sep + DATA_SET + '_features.h5'
    h5f_feat = h5py.File(FEAT_FILE, 'w')
    h5f_feat.create_dataset('Z', data=Z_output)
    h5f_feat.create_dataset('Y', data=Y_output)
    h5f_feat.close()

    # Calc time elapsed
    t_1 = time()
    time_diff = t_1 - t_0
    print(f'Feature extraction ran for {time_diff} seconds')

    # Creating log file to report load loops, number of tiles and elapsed time
    log_arr = [TILES[DATA_SET], num_loadLoops, [np.shape(Z_tiles[tile])[0] for tile in TILES[DATA_SET]], [round(float(time_diff), 2)]*len(TILES[DATA_SET])]
    log_arr = np.array(log_arr)
    np.transpose(log_arr)
    with open(DATA_FOLDER + 'logfile_' + DATA_SET + '.csv', 'w', newline='\n') as csvfile:
        log_writer = csv.writer(csvfile, delimiter=';')
        log_writer.writerow(['TILE', 'Num of batches', 'Num of rows', 'Total elapsed Time'])
        for row in log_arr:
            log_writer.writerow(row)
