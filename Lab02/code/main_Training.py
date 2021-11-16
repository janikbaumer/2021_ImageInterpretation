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
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from datetime import datetime
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from numpy import ma


# %%
### CLASSES

#from SatelliteSet import SatelliteSet

# %%
### FUNCTIONS
from RMSE_MAE import calculate_RMSE
from RMSE_MAE import calculate_MAE
# %%
### INITIALIZING
# Defining variables for future use
## THIS SCRIPT IS FOR REGRESSOR TRAINING

# Initializing file paths, taking os into account

current_os = platform.system()
if current_os == 'Darwin':
    cwd = '..'
    sep = '/'
elif current_os == 'Windows':
    cwd = os.getcwd()
    sep='\\'

## USER DEFINED PARAMETERS

DATA_SET = "train" #Name of input image. IDEA: Send one image at a time, be it test or train. Means running code 4 times for training and 2 for test. Attempt to reduce memory usage
VAL_SET = "..."

# TODO: Select appropriate parameters. Do we do StandardScaler()?
reg_mod = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))

###
FEAT_SET = "features_" + DATA_SET + ".h5"
FEAT_FILE = cwd+sep+FEAT_SET
DATA_SET = "dataset_" + DATA_SET + ".h5"
DATA_FILE = cwd+sep+'datasets'+sep+DATA_SET
VAL_FILE = cwd+sep+VAL_SET

# %% Data loading
if __name__ == "__main__":

    print('the correct file')
    # dealing with division by nan or 0
    np.seterr(divide='ignore', invalid='ignore')

    # Load image
    dset = h5py.File(DATA_FILE, 'r')
    Y = dset['GT']

    # Load feature vectors
    h5f_feat = h5py_File(FEAT_FILE, 'r')
    X = h5f_feat['feature_vectors'][:]

# %% Training

    reg_mod.fit(X, Y)

    # Close h5 feat file
    h5f_feat.close()

# %%

    now = datetime.now()

    print('VALIDATION STARTING ...')

    # Load validation image
    dset = h5py.File(VAL_FILE, 'r')
    GT = dset['GT']

    # Load validation features
    h5f_val = h5py_File(VAL_FILE, 'r')
    val_features = h5f_val['feature_vectors'][:]


    ## Predict the validation image
    prediction = reg_mod.predict(val_features)

    ## Calculating RMSE
    RMSE = calculate_RMSE(prediction, GT)

    ## Calculating MAE
    MAE = calculate_MAE(prediction, GT)

    # TODO: Save predicitons, and metrics. Use them later to plot images etc.
