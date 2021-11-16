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
import tensorflow as tf
from tensorflow import keras

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
    #dset = SatelliteSet(root=DATA_FILE, windowsize=windSize, test=False)

    # TODO: Load the pretrained-NN here, modify parameters
    model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(10980, 10980, 4),
        pooling="avg"
    )

    # Load Image
    dset = h5py.File(DATA_FILE, 'r')
    NIR = dset['NIR']
    RGB = dset['RGB']
    Y = dset['GT']
    X = np.concatenate([RGB, np.expand_dims(NIR, axis=-1)], axis=-1)

    # TODO: Do we do feature extraction on the whole image at once? LETS TRY
    print('STARTING FEATURE EXTRACTION ...')
    features = model.predict(X) #Returns the features as a np array.
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


