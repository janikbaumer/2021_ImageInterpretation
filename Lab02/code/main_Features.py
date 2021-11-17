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

from tqdm import tqdm
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from datetime import datetime
#from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
#from tensorflow import keras

from numpy import ma

# %%
### CLASSES

#from SatelliteSet import SatelliteSet

# %%
### FUNCTIONS
def dp(n, left): # returns tuple (cost, [factors])
    if (n, left) in memo: return memo[(n, left)]

    if left == 1:
        return (n, [n])

    i = 2
    best = n
    bestTuple = [n]
    while i * i <= n:
        if n % i == 0:
            rem = dp(n / i, left - 1)
            if rem[0] + i < best:
                best = rem[0] + i
                bestTuple = [i] + rem[1]
        i += 1

    memo[(n, left)] = (best, bestTuple)
    return memo[(n, left)]
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

# FILE HIERARCHY INSTRUCTIONS Feature Extraction
#   ...cwd
#       -\datasets
#           -\train
#               -merged_img_tile1.h5
#               -merged_img_tile2.h5
#               -merged_img_tile3.h5
#           -\val
#               -merged_img_tile4.h5
#           -\test
#               -merged_img_tile5.h5
#               -merged_img_tile6.h5
#       -main_Features.py
#       -main_Training.py
#       -RMSE_MAE.py


DATA_SET = "train"
DATA_FOLDER = cwd+sep+'datasets'+sep+DATA_SET


# %% Data Loading
if __name__ == "__main__":

    # dealing with division by nan or 0
    np.seterr(divide='ignore', invalid='ignore')

    # create datasets
    #dset = SatelliteSet(root=DATA_FILE, windowsize=windSize, test=False)



    directory = DATA_FOLDER

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".npy"):

            # Load Image
            """
            dset = h5py.File(filename, 'r')
            NIR = dset['NIR']
            RGB = dset['RGB']
            Y = dset['GT']
            X = np.concatenate([RGB, np.expand_dims(NIR, axis=-1)], axis=-1)
            """
            X = np.load(DATA_FOLDER+sep+filename)
            print(np.isnan(X))
            print(np.shape(X))

            #plt.imshow(X_masked)
            #plt.show()

            meanRNIR = np.nanmean(np.array([X[:, :, 0], X[:, :, 3]]), axis=0)
            print(meanRNIR)
            print('')
            meanGNIR = np.nanmean(np.array([X[:, :, 1], X[:, :, 3]]), axis=0)
            print(meanGNIR)
            print('')
            meanBNIR = np.nanmean(np.array([X[:, :, 2], X[:, :, 3]]), axis=0)
            print(meanBNIR)

            X = np.stack([meanRNIR, meanGNIR, meanBNIR], axis=-1)
            print(X)

            mask = np.isnan(X[:, :, 0])
            mean01 = np.mean(X[:, :, 0][np.logical_not(mask)])
            mean02 = np.mean(X[:, :, 1][np.logical_not(mask)])
            mean03 = np.mean(X[:, :, 2][np.logical_not(mask)])


            RNIR = X[:,:,0]
            GNIR = X[:, :, 1]
            BNIR = X[:, :, 2]

            RNIR[mask] = mean01
            GNIR[mask] = mean02
            BNIR[mask] = mean03

            X = np.stack([RNIR, GNIR, BNIR], axis=-1)
            print(X)

            X_masked = X
            """
            # TODO: Apply masking
            mask = np.isnan(X)
            X_masked = X[np.logical_not(mask)]
            print(np.shape(X_masked))
            n = np.shape(X_masked)
            n = n[0]/3
            if sympy.isprime(n)
                X_masked.drop[]

            memo={}
            m = dp(int(n), 2)
            print(m)
            """




            # TODO: Load the pretrained-NN here, modify parameters
            model = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights="imagenet",
                input_tensor=None,
                input_shape=np.shape(X_masked),
                pooling="avg"
            )

            # TODO: Reshape X_masked to right size, exactly 3 channels

            # TODO: Do we do feature extraction on the whole image at once? LETS TRY
            print('STARTING FEATURE EXTRACTION ...')
            features = model.predict(X_masked) # Returns the features as a np array.
            # This still needs to be modified to have the parameters. See documentation. For example, predict with batches or not?

            # TODO: Here the feature vectors should be stored and saved, in order to retrieve it later for the training.
            #   The features are returned as a numpy array, and then stored as a hdf5 file.
            #   QUESTION: do we mask before or after storage? The problem might be when storing everything together that
            #   every batch has different feature vector lengths, because of varying number of good pixels. Solution might
            #   be to add the mask before training.

            FEAT_FILE = DATA_FOLDER+cwd+sep+filename[0:-4]+'_features.h5'
            h5f_feat = h5py.File(FEAT_FILE, 'w')
            h5f_feat.create_dataset('feature_vectors', data=features)
            h5f_feat.close()
            #dset.close()
            continue
        else:
            continue



# %% WRITING FEATURES TO HDF5 FILE



    """
    IN CASE WE WANT TO CREATE A CSV FILE INSTEAD
    np.savetxt('features_'+DATA_SET+'.csv', features, delimiter=',', newline='n'') #Consider using '.gz' for compressed file.
    """


