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
# FILE HIERARCHY INSTRUCTIONS
#   ...cwd
#       -\dataset
#           -\train
#               -merged_img_tile1.h5
#               -merged_img_tile2.h5
#               -merged_img_tile3.h5
#               -merged_img_tile1_features.h5
#               -merged_img_tile2_features.h5
#               -merged_img_tile3_features.h5
#           -\val
#               -merged_img_tile4.h5
#               -merged_img_tile4_features.h5
#           -\test
#               -merged_img_tile5.h5
#               -merged_img_tile6.h5
#               -merged_img_tile5_features.h5
#               -merged_img_tile6_features.h5
#       -main_Features.py
#       -RMSE_MAE.py


DATA_SET = 'train' #Name of input image. IDEA: Send one image at a time, be it test or train. Means running code 4 times for training and 2 for test. Attempt to reduce memory usage
VAL_SET = "val"

# Initializing the regressor
reg_mod = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))

### Folder paths
DATA_FOLDER = cwd+sep+'datasets'+sep+DATA_SET
VAL_FOLDER = cwd+sep+'datasets'+sep+VAL_SET

# %% Data loading
if __name__ == "__main__":

    print('the correct file')
    # dealing with division by nan or 0
    np.seterr(divide='ignore', invalid='ignore')

    directory = os.fsencode(DATA_FOLDER)

    if DATA_SET == 'train':
        #X = np.array()
        #Y = np.array()
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith("features.h5"):
                FEAT_FILE = DATA_FOLDER+sep+filename
                DATA_FILE = DATA_FOLDER+sep+'RGNIR_final_tile_2.npy'

                # Load ground truth and mask
                #dset = h5py.File(DATA_FILE, 'r')
                Y = np.load(DATA_FOLDER+sep+'label_tile_2.npy')
                for_mask = np.load(DATA_FOLDER+sep+'RGBNIR_final_tile_2.npy')
                for_mask = for_mask[:,:,0]

                # Load feature vectors
                with h5py.File(FEAT_FILE, 'r') as h5f_feat:
                    X = h5f_feat['feature_vectors']

                    # Apply mask to ground truth
                    mask = for_mask==0
                    Y[mask]
                    ## Mask feature vectors...


                    # TODO: Make sure this concatenation works with the dimensions of Y
                    Y = np.concatenate((Y, dset['GT']), axis=1)
                    X = np.concatenate((X, h5f_feat['feature_vectors'][:]), axis=1)

                continue
            else:
                continue

                # Fit the model/train the model
                reg_mod.fit(X, Y)


# %%
    if DATA_SET == 'test':
        VAL_FOLDER = DATA_FOLDER

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith("features.h5"):
            FEAT_FILE = DATA_FOLDER + sep + filename
            DATA_FILE = DATA_FOLDER + sep + filename[0:-12] + '.h5'

            # Load image
            dset = h5py.File(DATA_FILE, 'r')
            Y = dset['GT']

            # Load feature vectors
            h5f_feat = h5py_File(FEAT_FILE, 'r')
            X = h5f_feat['feature_vectors']

            # Mask data...


            continue
        else:
            continue


    ## Predict the validation image
    prediction = reg_mod.predict(X)

    ## Calculating RMSE
    RMSE = calculate_RMSE(prediction, Y)

    ## Calculating MAE
    MAE = calculate_MAE(prediction, Y)

    # TODO: Save predicitons, and metrics. Use them later to plot images etc.
