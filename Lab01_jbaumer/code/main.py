### IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch

from tqdm import tqdm
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import f1_score, cohen_kappa_score

### CLASSES

class SatelliteSet(VisionDataset):

    # test flag: whether data is loaded completely into memory
    def __init__(self, root="../datasets/dataset_train.h5", windowsize=128,test=False):

        super().__init__(root)

        self.wsize = windowsize
        if test:
            h5 = h5py.File(root, 'r', driver="core") # Store the data in memory
        else:
            h5 = h5py.File(root, 'r')

        self.RGB = h5["RGB"]
        self.NIR = h5["NIR"]
        self.CLD = h5["CLD"]
        self.GT = h5["GT"]

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
        CLD_sample = self.CLD[b, n:n + self.wsize, m:m + self.wsize]
        GT_sample = self.GT[b, n:n + self.wsize, m:m + self.wsize]

        # normalize NIR and RGB by maximumg possible value
        NIR_sample = np.asarray(NIR_sample, np.float32) / (2 ** 16 - 1)
        RGB_sample = np.asarray(RGB_sample, np.float32) / (2 ** 8 - 1)
        X_sample = np.concatenate([RGB_sample, np.expand_dims(NIR_sample, axis=-1)], axis=-1)

        ### correct gt data ###
        # first assign gt at the positions of clouds
        cloud_positions = np.where(CLD_sample > 10)
        GT_sample[cloud_positions] = 2
        # second remove gt where no data is available - where the max of the input channel is zero
        idx = np.where(np.max(X_sample, axis=-1) == 0)  # points where no data is available
        GT_sample[idx] = 99  # 99 marks the absence of a label and it should be ignored during training
        GT_sample = np.where(GT_sample > 3, 99, GT_sample)
        # pad the data if size does not match
        sh_x, sh_y = np.shape(GT_sample)
        pad_x, pad_y = 0, 0
        if sh_x < self.wsize:
            pad_x = self.wsize - sh_x
        if sh_y < self.wsize:
            pad_y = self.wsize - sh_y

        x_sample = np.pad(X_sample, [[0, pad_x], [0, pad_y], [0, 0]])
        gt_sample = np.pad(GT_sample, [[0, pad_x], [0, pad_y]], 'constant',
                           constant_values=[99])  # pad with 99 to mark absence of data

        # pytorch wants the data channel first - you might have to change this
        x_sample = np.transpose(x_sample, (2, 0, 1))
        return np.asarray(x_sample), gt_sample

    def __len__(self):
        return self.num_windows



### FUNCTIONS


### VARIABLES

# for real training, change FILE_TRAIN to ../datasets/dataset_train.h5
FILE_TRAIN = '../datasets/dataset_train_reduced.h5'
FILE_TRAIN = '../datasets/dataset_train_devel.h5'
FILE_VAL = '../datasets/dataset_val.h5'
FILE_TEST = '../datasets/dataset_test.h5'



if __name__ == "__main__":
    '''
    # for plotting
    # for imshow (labels) 0: green/blue-ish (background) | 1: green (palm-oil tree) | 2: white (cloud) | 3: black (no data)
    colormap = [[47, 79, 79], [0, 255, 0], [255, 255, 255], [0, 0, 0]]
    colormap = np.asarray(colormap)  # convert list of lists to numpy array
    '''

    # create datasets
    dset_train = SatelliteSet(root=FILE_TRAIN, windowsize=128, test=False)
    dset_val = SatelliteSet(root=FILE_VAL, windowsize=128, test=True)  # test flag: data only in memory
    dset_test = SatelliteSet(root=FILE_TEST, windowsize=128, test=True)

    # create dataloader that samples batches from the dataset
    train_loader = DataLoader(dset_train,
                              batch_size=8,
                              num_workers=0,
                              shuffle=False)

    val_loader = DataLoader(dset_val,
                            batch_size=4,
                            num_workers=0,
                            shuffle=False)

    test_loader = DataLoader(dset_test,
                             batch_size=8,
                             num_workers=0,
                             shuffle=False)


    # create initialization of certain ML model

    ## Naive Bayes:
    gnb = GaussianNB()


    print('TRAINING STARTING ...')

    train_loader_loop = 0

#    for x_tr, y_tr in tqdm(train_loader):
#        pass

    for x, y in tqdm(train_loader): # tqdm: make loops show a smart progress meter by wrapping any iterable with tqdm(iterable)
        train_loader_loop += 1
        x = np.transpose(x, [0, 2, 3, 1])  # swap shapes so that afterward shape = (nmbr_imgs_in_batch, size_x, size_y, nmbr_channels)
        x = x.numpy()  #  x is not yet ndarray - convert x from pytorch tensor to ndarray

        # change no data (99) to (3), for plotting reasons
        y = np.where(y == 99, 3, y)  # y is already ndarry


        # loop over batch size of train_loader ((all images in this batch (?))
        for i in range(len(x)):
            print(f'training with image {i} of {len(x)} of train loader loop no: {train_loader_loop}')
            x_batch = x[i]
            y_batch = y[i]

            # define shapes
            x_shape = x_batch.shape
            y_shape = y_batch.shape

            # initialize lists in which features / labels are stored
            X_lst = []
            Y_lst = []

            # create feature matrix X_train (can be used for ML model later)
            # for each channel, stack features (e.g. R,G,B,NIR intensities) to column vector
            for chn in range(x_shape[-1]):
                x_batch_chn = x_batch[:,:,chn]
                x_batch_chn.resize(x_shape[0]*x_shape[1], 1)
                x_shape_resized = x_batch_chn.shape
                X_lst.append(x_batch_chn)

            y_batch.resize(y_shape[0]*y_shape[1], 1)
            Y_lst.append(y_batch)

            # define feature matrix and label vector
            X_train = np.array(X_lst).T[0]  # transpose to have one col per feature, first ele because this dimension was somehow added
            Y_train = np.array(Y_lst).T[0].ravel()  # same as above, plus ravel() to convert from col vector to 1D array (needed for some ML models)


            # train model

            ## Naive Bayes
            gnb.fit(X_train, Y_train)

    # HERE gnb IS COMPLETELY TRAINED


    # todo
    # for validation, take full imgs of validations dsets - we anyway take subsamples (128 pxl)
    # predict with gnb model (this case)
    # compare prediction to GT of validation dsets
    # compute confusion matrix per batch, then aggregate to get CM_full

    # based confusion matrices, different models can be compared - use best one on test dset




    print('VALIDATION STARTING ...')

    iterator_val = 0
    f1_full = 0
    kappa_full = 0
    CM_full = np.zeros((4,4))
    for x_va, y_va in tqdm(val_loader): # if windowsize of val_loader is set to full size (10980), then length of this loop = batch_size of dataLoader
        iterator_val += 1
        x_va = np.transpose(x_va, [0, 2, 3, 1])  # swap shapes so that afterward shape = (nmbr_imgs_in_batch, size_x, size_y, nmbr_channels)
        x_va = x_va.numpy()  # x is not yet ndarray - convert x from pytorch tensor to ndarray

        # change no data (99) to (3), for plotting reasons
        y_va = np.where(y_va == 99, 3, y_va)  # y is already ndarry


        # loop over batch size of train_loader ((all images in this batch (?))
        for i in range(len(x_va)):
            # print(f'training with image {i} of {len(x_va)} of train loader loop no: {train_loader_loop}')
            x_va_batch = x_va[i]
            y_va_batch = y_va[i]

            # define shapes
            x_va_shape = x_va_batch.shape
            y_va_shape = y_va_batch.shape

            # initialize lists in which features / labels are stored
            X_val_lst = []
            Y_val_lst = []

            # create feature matrix X_train (can be used for ML model later)
            # for each channel, stack features (e.g. R,G,B,NIR intensities) to column vector
            for chn in range(x_va_shape[-1]):
                x_va_batch_chn = x_va_batch[:,:,chn]
                x_va_batch_chn.resize(x_va_shape[0]*x_va_shape[1], 1)
                x_va_shape_resized = x_va_batch_chn.shape
                X_val_lst.append(x_va_batch_chn)

            y_va_batch.resize(y_va_shape[0]*y_va_shape[1], 1)
            Y_val_lst.append(y_va_batch)


            # define feature matrix and label vector
            X_val = np.array(X_val_lst).T[0]  # transpose to have one col per feature, first ele because this dimension was somehow added
            Y_val = np.array(Y_val_lst).T[0].ravel()  # same as above, plus ravel() to convert from col vector to 1D array (needed for some ML models)

            Y_pred = gnb.predict(X_val)

            cm = confusion_matrix(Y_val, Y_pred, labels=[0, 1, 2, 3])
            #print('cm: \n', cm)
            CM_full = CM_full + cm

            # compute f1-score for each batch
            #f1 = f1_score(Y_val, Y_pred)
            #f1_full += f1
            #print()

            # kappa
            #kappa = cohen_kappa_score(Y_val, Y_pred)
            #kappa_full += kappa
            #print('CM full: \n', CM_full)
            #print()


    print('complete confusion matrix: ', CM_full)
    print()

    # EV add multiple different scores for model evaluation

    # average f1 scores, kappa
    #f1_avg = f1_full / iterator_val
    #kappa_avg = kappa_full / iterator_val





    # Please note that random shuffling (shuffle=True) -> random access.
    # this is slower than sequential reading (shuffle=False)
    # If you want to speed up the read performance but keep the data shuffled, you can reshape the data to a fixed window size
    # e.g. (-1,4,128,128) and shuffle once along the first dimension. Then read the data sequentially.
    # another option is to read the data into the main memory h5 = h5py.File(root, 'r', driver="core")


    # for plotting
    #f, axarr = plt.subplots(ncols=3, nrows=8)






#### discussion with other groups

# models

# KNN
# Decision tree
# rdm forest
# SVM
# naive bayes


# metrics

# kappa
# accuracy
# precision, recall, F1-score


# features (depends on time)

# NDVI
# etc






# todo: preprocessing and evaluation
### PREPROCESSING DATA
# ADDING AN NDVI CHANNEL
# FURTHER PREPROCESSING (FEATURE EXTRACTION)

### EVALUATION
# CREATE CONFUSION MATRIX