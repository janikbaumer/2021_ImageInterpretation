# Image Interpretation Lab 3 Framework

# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import os
import platform

from time import time
from tqdm import tqdm
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# CLASSES
# from dataset import Dataset

# FILE STRUCTURE
# Current Working Directory
#       imgint_trainset_v2.hdf5
#       imgint_testset_v2.hdf5

current_os = platform.system()
if current_os == 'Windows':
    cwd = os.getcwd()
    print("Current working directory: {0}".format(cwd))
    sep = '\\'

# DATA LOADING
path_train = format(cwd) + sep + 'imgint_trainset_v2.hdf5'
dset_train = h5py.File(path_train)
print('Training set info')
print(dset_train.keys())
data_train = dset_train['data']
gt_train = dset_train['gt']
print(data_train.shape)
print(gt_train.shape)

print('')

path_test = format(cwd) + sep + 'imgint_testset_v2.hdf5'
dset_test = h5py.File(path_test)
print('Test set info')
print(dset_test.keys())
data_test = dset_test['data']
gt_test = dset_test['gt']
print(data_test.shape)
print(gt_test.shape)