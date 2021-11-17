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
import sympy

"""
import torchvision.models as models
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
# from torchsummary import summary
# from efficientnet_pytorch import EfficientNet
import time
import copy
"""

from PIL import Image
from tqdm import tqdm
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from datetime import datetime
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from tensorflow import keras
#¢from keras.wrappers.scikit_learn import KerasRegressor

from numpy import ma


# %%
### CLASSES

# from SatelliteSet import SatelliteSet


# %%
### INITIALIZING
# Defining variables
## THIS SCRIPT IS FOR FEATURE EXTRACTION

# Initializing file paths, taking os into account

current_os = platform.system()
if current_os == 'Darwin':
    cwd = '..'
    sep = '/'
elif current_os == 'Windows':
    cwd = os.getcwd()
    sep = '\\'

# FILE HIERARCHY INSTRUCTIONS Feature Extraction
#   ...cwd
#       -\datasets
#           -\train
#               -merged_img_tile1.npy
#               -merged_img_tile2.npy
#               -merged_img_tile3.npy
#           -\val
#               -merged_img_tile4.npy
#           -\test
#               -merged_img_tile5.npy
#               -merged_img_tile6.npy
#       -main_Features.py
#       -main_Feat_Keras.py
#       -main_Training.py
#       -RMSE_MAE.py


DATA_SET = "train"
DATA_FOLDER = cwd + sep + 'datasets' + sep + DATA_SET

# %% Data Loading
if __name__ == "__main__":

    # dealing with division by nan or 0
    np.seterr(divide='ignore', invalid='ignore')

    # create datasets
    # dset = SatelliteSet(root=DATA_FILE, windowsize=windSize, test=False)

    directory = DATA_FOLDER

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".npy"):

            # Load Image
            X = np.load(DATA_FOLDER + sep + filename)
            Y = np.load(cwd + sep + "datasets" + sep + "label_tile_2.npy")

            # Integrating NIR in RGB channels to reduce channels to 3
            meanRNIR = np.nanmean(np.array([X[:, :, 0], X[:, :, 3]]), axis=0)
            meanGNIR = np.nanmean(np.array([X[:, :, 1], X[:, :, 3]]), axis=0)
            meanBNIR = np.nanmean(np.array([X[:, :, 2], X[:, :, 3]]), axis=0)

            X = np.stack([meanRNIR, meanGNIR, meanBNIR], axis=-1)

            # Creating the masks where there are bad pixels
            mask01 = np.isnan(X[:, :, 0])
            mask02 = np.isnan(X[:, :, 1])
            mask03 = np.isnan(X[:, :, 2])

            # Setting bad pixels equal to 0
            RNIR = X[:, :, 0]
            GNIR = X[:, :, 1]
            BNIR = X[:, :, 2]
            RNIR[mask01] = 0
            GNIR[mask02] = 0
            BNIR[mask03] = 0

            X = np.stack([RNIR, GNIR, BNIR], axis=-1)

            X_masked = X


            # Create the base model ofr feature extraction
            base_model = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights="imagenet",
                input_tensor=None,
                input_shape=(10980, 10980, 3),
                pooling="max",
            )

            base_model.trainable = False

            """
            inputs = keras.Input(shape=(10980, 10980, 3), batch_size=None)
            # We make sure that the base_model is running in inference mode here,
            x = base_model(inputs, training=False)
            features = base_model.predict(X_masked)  # Returns the features as a np array.
            # Convert features of shape `base_model.output_shape[1:]` to vectors
            #x = keras.layers.GlobalAveragePooling2D()(x)
            # A Dense classifier with a single unit (binary classification)
            #outputs = keras.layers.Dense(1)(x)
            #model = keras.Model(inputs, outputs)
            """


            #model.compile(optimizer=keras.optimizers.Adam(), loss='mean_squared_error')
            #model.fit(x=X_masked, y=Y, epochs=20)




            # Feature extraction
            print('STARTING FEATURE EXTRACTION ...')
            features = base_model.predict(X_masked) # Returns the features as a np array.
            # This still needs to be modified to have the parameters. See documentation. For example, predict with batches or not?

            # Saving features to drive
            FEAT_FILE = DATA_FOLDER + sep + filename[0:-4] + '_features.h5'
            h5f_feat = h5py.File(FEAT_FILE, 'w')
            h5f_feat.create_dataset('feature_vectors', data=features)
            h5f_feat.close()
            # dset.close()
            continue
        else:
            continue