# %%
### IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import torch
import os
import cv2
import pickle as cPickle
import platform
import csv
from time import time

from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from tqdm import tqdm
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import TensorDataset, DataLoader
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
"""
#Custum multiregressor
class MyMultiRegressor:
    def __init__(self, params):


    def partial_fit(self, Z, Y):
        last_model = self.model
        DM_train = xgb.DMatrix(data=Z, label=Y[:, 0])



        return new_model
"""

#Custom class for training
class Lab02FeatDset(TensorDataset):
    def __init__(self, root='../datasets/tile_2_features.h5'):

        h5 = h5py.File(root, 'r')

        self.Z = h5['Z']
        self.Y = h5['Y']

    def __getitem__(self, index):
        Z = self.Z[index,:]
        Y = self.Y[index,:]
        Y = np.nan_to_num(Y)

        return Z, Y

    def __len__(self):
        return len(self.Z)


# %%
### FUNCTIONS
from RMSE_MAE import calculate_RMSE
from RMSE_MAE import calculate_MAE

def plot_pred_gt(pred, Y, is_training):

    Y = Y[400,:]
    pred = pred[400,:]

    Y = np.reshape(Y, (128, 128))
    pred = np.reshape(pred, (128, 128))

    diff = Y - pred

    if is_training:
        DATA_SET = 'validation'
    else:
        DATA_SET = 'test'
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,
                            figsize=(30, 10), edgecolor='black',
                            tight_layout=True)
    fig.suptitle('Plot prediction results on ' + DATA_SET, fontsize=28)

    # Plot the GT canopy heights
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)  # Appending an axis for the color bar to the right
    im1 = ax1.imshow(Y, cmap='inferno')  # Plotting GT in subplot
    ax1.set_title(f'Ground truth', fontsize=14)  # Giving subplot title
    cbar1 = fig.colorbar(im1, cax=cax1, orientation='vertical',
                        ticks=[0.1, 30, 59.9])  # Creating colorbar on the appended axis
    cbar1.set_label('True Canopy Height [m]', rotation=90)  # Label the colorbar
    cbar1.ax.set_yticklabels(['0', '30', '60'])  # Set colorbar tick labels
    # Plot the predicted canopy heights
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)  # Appending an axis for the color bar to the right
    im2 = ax2.imshow(pred, cmap='inferno')  # Plotting GT in subplot
    ax2.set_title(f'Predicted values', fontsize=14)  # Giving subplot title
    cbar2 = fig.colorbar(im2, cax=cax2, orientation='vertical',
                        ticks=[0.1, 30, 59.9])  # Creating colorbar on the appended axis
    cbar2.set_label('Predicted Canopy Height [m]', rotation=90)  # Label the colorbar
    cbar2.ax.set_yticklabels(['0', '30', '60'])  # Set colorbar tick labels
    # Plot the true minus the predicted values
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)  # Appending an axis for the color bar to the right
    im3 = ax3.imshow(pred, cmap='bwr')  # Plotting GT in subplot
    ax3.set_title(f'True minus predicted values', fontsize=14)  # Giving subplot title
    cbar3 = fig.colorbar(im3, cax=cax3, orientation='vertical',
                        ticks=[0.1, 10, 19.9])  # Creating colorbar on the appended axis
    cbar3.set_label('Difference [m]', rotation=90)  # Label the colorbar
    cbar3.ax.set_yticklabels(['0', '10', '20'])  # Set colorbar tick labels
    plt.show()

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
    sep = '\\'

## USER DEFINED PARAMETERS
# FILE HIERARCHY INSTRUCTIONS
#   ...
#       -\dataset
#           -\train
#               -merged_img_tile1.h5
#               -merged_img_tile2.h5
#               -merged_img_tile3.h5
#               -train_features.h5
#           -\val
#               -merged_img_tile4.h5
#               -val_features.h5
#           -\test
#               -merged_img_tile5.h5
#               -merged_img_tile6.h5
#               -test_features.h5
#           -\dev
#               -dev_Lab02.h5
#               -dev_features.h5

ROOT_PATH = r'C:\Users\Jor Fergus Dal\Desktop\Lav02_tiles_v1'

is_training = True

TRAIN_SET = 'train'
VAL_SET = 'val'
TEST_SET = 'test'

# Hyperparameters of the model
"""
objective #determines the loss function to be used
colsample_bytree #percentage of features used per tree. High value can lead to overfitting.
learning_rate #step size shrinkage used to prevent overfitting. Range is [0,1]
max_depth #determines how deeply each tree is allowed to grow during any boosting round.
alpha #L1 regularization on leaf weights. A large value leads to more regularization.
n_estimators #number of trees you want to build.
#subsample #percentage of samples used per tree. Low value can lead to underfitting.
"""
models_list = ['XGBoost', 'RandomForest', 'DecisiontTree']
select_model = 2

if select_model==0:
    xgReg_params = {"objective": "reg:linear", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
                    'max_depth': 5, 'alpha': 10}
    xg_reg = xgb.XGBRegressor(xgReg_params)
    xg_reg.xg_model = xg_reg
    int_model = MultiOutputRegressor(xg_reg)
elif select_model==1:
    RandFor_params = {"n_estimators": 100, "criterion": 'squared_error', "max_depth": 10}
    RandForReg = RandomForestRegressor(RandFor_params)
    int_model = RandForReg
elif select_model==2:
    # TODO: Finish initializing
    DecTree_params = {'criterion': 'squared-error', 'max_depth': 15}
    dec_tree_reg = DecisionTreeRegressor(DecTree_params)
    int_model = dec_tree_reg

### Folder paths
TRAIN_FOLDER = ROOT_PATH + sep + 'datasets' + sep + TRAIN_SET
VAL_FOLDER = ROOT_PATH + sep + 'datasets' + sep + VAL_SET
TEST_FOLDER = ROOT_PATH + sep + 'datasets' + sep + TEST_SET

# %% Model Fitting
if __name__ == "__main__":

    t_0 = time()

    # dealing with division by nan or 0
    np.seterr(divide='ignore', invalid='ignore')

    # Model fitting
    if is_training:
        directory = os.fsencode(TRAIN_FOLDER)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith("features.h5"):
                FEAT_FILE = TRAIN_FOLDER + sep + filename

                # Load feature vectors and GT
                with h5py.File(FEAT_FILE, 'r') as h5f_feat:

                    dset = Lab02FeatDset(FEAT_FILE)

                    dataLoader = DataLoader(dset, batch_size=8, num_workers=0, shuffle=True)
                    model = None
                    for Z, Y in tqdm(dataLoader):
                        Z = Z.numpy()
                        Y = Y.numpy()

                        # TODO: Placeholder statements...
                        #data_DMatrix = xgb.DMatrix(data=Z, label=Y[:,0])  # DMatrix type is optimized for XGBoost

                        # Model fitting
                        #multiRegressor.fit(Z, Y)
                        #xg_reg.train(data_DMatrix, xgb_model=xg_reg)

                        #THis is implementing partial_fit, mor eor less. Now put this in the MultiOutputRegressor
                        """
                        model = xgb.train({
                            'learning_rate': 0.007,
                            'update': 'refresh',
                            'process_type': 'default',
                            'refresh_leaf': True,
                            # 'reg_lambda': 3,  # L2
                            'reg_alpha': 3,  # L1
                            'silent': False,
                        }, dtrain=data_DMatrix,
                            xgb_model=model)
                        """
                        




                continue
            else:
                continue

    # %% Model Evaluation
    if is_training == False:
        VAL_FOLDER = TEST_FOLDER
        DATA_SET = 'test'
    else:
        DATA_SET = 'val'

    directory = os.fsencode(VAL_FOLDER)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith("features.h5"):
            FEAT_FILE = VAL_FOLDER + sep + filename

            # Load image
            dset = h5py.File(FEAT_FILE, 'r')
            Y = dset['Y']
            X = dset['Z']

            # Load data in optimized DMatrix

            pred = multiRegressor.predict(X)




            continue
        else:
            continue

    plot_pred_gt(pred, Y, is_training)

    ## Calculating RMSE
    RMSE = calculate_RMSE(pred, Y)

    ## Calculating MAE
    MAE = calculate_MAE(pred, Y)

    t_1 = time()
    dt = t_1 - t_0

    # TODO: Save predicitons, and metrics. Use them later to plot images etc.
    with open(ROOT_PATH + 'logfile_' + DATA_SET + '.csv', 'w', newline='\n') as csvfile:
        log_writer = csv.writer(csvfile, delimiter=';')
        log_writer.writerow(['RMSE', 'MAE', 'Time Elapsed'])
        log_writer.writerow([RMSE, MAE, round(float(dt), 2)])

