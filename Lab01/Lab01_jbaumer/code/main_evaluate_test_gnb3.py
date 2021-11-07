### IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import os
import cv2
import pickle as cPickle

from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import f1_score, cohen_kappa_score
from skimage.filters import prewitt_h, prewitt_v
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.impute import KNNImputer
from scipy import ndimage
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from skmultiflow.trees import HoeffdingTree
from skmultiflow.lazy import KNNClassifier
from numpy import ma


### Functions

def mean_pix(imgmat: np.array) -> np.array:
    (nrow, ncol, nchan) = imgmat.shape
    mean_pix_value = np.zeros((nrow, ncol), dtype=float)
    # dtypeDict = {"uint8":(2**8-1), "uint16":(2**16-1)}
    for i in range(nchan):
        # imgmat[:,:,i] = imgmat[:,:,i]/dtypeDict[str(imgmat[:,:,i].dtype)]
        mean_pix_value += imgmat[:, :, i]
    mean_pix_value /= i
    return mean_pix_value


def window_level_function(image, level):
    minimum = np.min(image[:])  # Minimum intensity value in the image
    maximum = np.max(image[:])  # Maximum intensity value in the image

    window = np.std(image)  # Define window as +/- std

    # Clip or remove everything before or after the window boundaries
    image = np.clip(image, (level - (window / 2)), (level + (window / 2)))
    # Everything below left limit becomes black
    # Everyting above right limit becomes white

    m = (maximum - minimum) / window;  # Slope of the window level transfer function
    b = maximum - (m * (level + (window / 2)))  # y-intercept of the window level transfer function

    image = m * image + b  # The remaining, non-scaled values are adjusted by applying a linear transformation
    return image  # Convert output to unsigned 8-bit integer and return


def calc_ndvi(NIR, RED):
    """
    This function takes the NIR and RED channels of an image and returns the NDVI index.
    Make sure to pass the parameters are matrices of exactly 2 dimensions.
    Also, important that zero divisions are handled properly.
    Before calculations, the channels are normalized.

    Parameters
    ----------
    NIR : nparray
        Array of 2 dim, where dims indicate pixels, and value represents NIR intensity.
    RED : nparray
        Array of 2 dim, where dims indicate pixels, and value represents RED intensity.

    Returns
    -------
    NDVI : nparray
        Array of 2 dim, where dims indicate pixels, and value represents NDVI, a value between -1 and 1.

    """

    NDVI = (NIR.astype(float) - RED.astype(float)) / (NIR + RED)
    NDVI = np.nan_to_num(NDVI)
    # imputer = KNNImputer(n_neighbors = 50)
    # NDVI = imputer.fit_transform(NDVI)
    return NDVI

### CLASSES

class SatelliteSet(VisionDataset):

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
cwd = os.getcwd()
# FILE_TRAIN = '../datasets/dataset_train_reduced.h5'
FILE_TRAIN = '../datasets/dataset_train_devel.h5'
FILE_TRAIN = '../datasets/dataset_train_decide_20imgs.h5'
FILE_VAL = '../datasets/dataset_val.h5'
FILE_TEST = '../datasets/dataset_test.h5'

CLASSES = [0, 1, 2]

if __name__ == "__main__":
    # dealing with division by nan or 0
    np.seterr(divide='ignore', invalid='ignore')

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

    if not os.path.isfile('gnb3_classifier27_19.pkl'):  # if this file does not yet exist (the best model, acc. to test set)
        print('INTIALIZATION STARTING ...')

        # create initialization of certain ML model
        ## Naive Bayes:
        gnb3 = GaussianNB()

        print('TRAINING STARTING ...')

        train_loader_loop = 0

        for x, y in tqdm(train_loader):  # tqdm: make loops show a smart progress meter by wrapping any iterable with tqdm(iterable)
            train_loader_loop += 1
            # print('train loader loop: ', train_loader_loop)
            x = np.transpose(x, [0, 2, 3, 1])  # swap shapes so that afterward shape = (nmbr_imgs_in_batch, size_x, size_y, nmbr_channels)
            x = x.numpy()  # x is not yet ndarray - convert x from pytorch tensor to ndarray

            # change no data (99) to (3), for plotting reasons
            y = np.where(y == 99, 3, y)  # y is already ndarry

            # loop over batch size of train_loader ((all images in this batch (?))
            for i in range(len(x)):
                # print(f'training with image {i} of {len(x)} of train loader loop no: {train_loader_loop}')
                x_batch = x[i]
                y_batch = y[i]


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
                    #stack masked cols together
                    #no_data = np.where(Y_train == 3)

                    #X_train = X_train_masked
                    #Y_train = Y_train_masked


                    Y_train = Y_train_masked_compressed
                    X_train = X_train_masked[~np.isnan(X_train_masked)]

                    #X_train = X_train_flat.
                if not X_train.shape == (0,) and not Y_train.shape == (0,):
                    # normalizing and standardizing the feature vectors
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)

                    # train model
                    # Naive Bayes

                    gnb3.partial_fit(X_train[:, 0:6], Y_train, classes=CLASSES)

    else:  # if those files exist, read them from disk
        print('FILES ALREADY EXISTS - READING MODELS FROM PICKLE FILES ...')

        gnb3 = cPickle.load(open('gnb3_classifier27_19.pkl', 'rb'))

    # HERE models ARE COMPLETELY TRAINED

    now = datetime.now()

    if not os.path.isfile('gnb3_classifier27_19.pkl'):
        #Save the classifiers todo: this might duplicate models even if they already exist

        with open('gnb3_classifier' + now.strftime("%d_%H") + '.pkl', 'wb') as fid:
            cPickle.dump(gnb3, fid)

    print('TESTING STARTING ...')

    iterator_test = 0

    CM_full_gnb3 = np.zeros((3, 3))

    for x_tst, y_tst in tqdm(test_loader):  # if windowsize of val_loader is set to full size (10980), then length of this loop = batch_size of dataLoader
        iterator_test += 1
        x_tst = np.transpose(x_tst, [0, 2, 3, 1])  # swap shapes so that afterward shape = (nmbr_imgs_in_batch, size_x, size_y, nmbr_channels)
        x_tst = x_tst.numpy()  # x is not yet ndarray - convert x from pytorch tensor to ndarray

        # change no data (99) to (3), for plotting reasons
        y_tst = np.where(y_tst == 99, 3, y_tst)  # y is already ndarry

        # loop over batch size of train_loader ((all images in this batch (?))
        for i in range(len(x_tst)):
            # print(f'training with image {i} of {len(x_va)} of train loader loop no: {train_loader_loop}')
            x_tst_batch = x_tst[i]
            y_tst_batch = y_tst[i]

            # FEATURE EXTRACTION
            # Grayscale as feature, adds 1 feat
            x_tst_batch_gray = cv2.cvtColor(x_tst_batch[:, :, 0:3], cv2.COLOR_RGB2GRAY)
            x_tst_batch = np.dstack((x_tst_batch, x_tst_batch_gray))

            # NDVI, adds 1 feat
            x_tst_batch_ndvi = calc_ndvi(x_tst_batch[:, :, 3], x_tst_batch[:, :, 0])
            x_tst_batch = np.dstack((x_tst_batch, x_tst_batch_ndvi))

            # Sobel axis 1
            sobel_axis0_tst = ndimage.sobel(x_tst_batch[:, :, 4], axis=0)
            x_tst_batch = np.dstack((x_tst_batch, sobel_axis0_tst))

            # Sobel axis 2
            sobel_axis1_tst = ndimage.sobel(x_tst_batch[:, :, 4], axis=1)
            x_tst_batch = np.dstack((x_tst_batch, sobel_axis1_tst))

            # Mean pixel intensity, all four original intensities, adds 1 feat
            x_batch_meanpix_tst = mean_pix(x_tst_batch[:, :, 0:4])
            x_tst_batch = np.dstack((x_tst_batch, x_batch_meanpix_tst))

            # Mean pixel intensity, RGB, adds 1 feat
            x_batch_meanpixRGB_tst = mean_pix(x_tst_batch[:, :, 0:3])
            x_tst_batch = np.dstack((x_tst_batch, x_batch_meanpixRGB_tst))

            # Prewitt horizontal edges, adds 1 feat
            edges_prewitt_horizontal_tst = prewitt_h(x_tst_batch[:, :, 4])
            x_tst_batch = np.dstack((x_tst_batch, edges_prewitt_horizontal_tst))

            # Prewitt vertical edges, adds 1 feat
            edges_prewitt_vertical_tst = prewitt_v(x_tst_batch[:, :, 4])
            x_tst_batch = np.dstack((x_tst_batch, edges_prewitt_vertical_tst))

            # Window-level scaling/shifting, adds 1 feat
            x_batch_wl_tst = window_level_function(x_tst_batch[:, :, 4], 0.8)
            x_tst_batch = np.dstack((x_tst_batch, x_batch_wl_tst))

            # convert nan values to 0
            x_tst_batch = np.nan_to_num(x_tst_batch)

            # define shapes
            x_tst_shape = x_tst_batch.shape
            y_tst_shape = y_tst_batch.shape

            # initialize lists in which features / labels are stored
            X_test_lst = []
            Y_test_lst = []

            # create feature matrix X_va (can be used for ML model later)
            # for each channel, stack features (e.g. R,G,B,NIR intensities) to column vector
            for chn in range(x_tst_shape[-1]):
                x_tst_batch_chn = x_tst_batch[:, :, chn]
                x_tst_batch_chn.resize(x_tst_shape[0] * x_tst_shape[1], 1)
                x_tst_shape_resized = x_tst_batch_chn.shape
                X_test_lst.append(x_tst_batch_chn)

            y_tst_batch.resize(y_tst_shape[0] * y_tst_shape[1], 1)
            Y_test_lst.append(y_tst_batch)

            # define feature matrix and label vector
            X_test = np.array(X_test_lst).T[
                0]  # transpose to have one col per feature, first ele because this dimension was somehow added
            Y_test = np.array(Y_test_lst).T[
                0].ravel()  # same as above, plus ravel() to convert from col vector to 1D array (needed for some ML models)

            # do masking only if there is no_data
            if np.max(Y_test) == 3:
                Y_test_masked = ma.masked_where(Y_test == 3, Y_test)
                Y_test_masked_compressed = Y_test_masked.compressed()
                n_rows = int(Y_test_masked_compressed.shape[0])

                X_test_masked = ma.zeros(X_test.shape)
                mask_test = Y_test_masked.mask

                # mask each column of X_test and replace X_test column with its masked corresponding
                for col in range(X_test[:].shape[1]):
                    col_vec = X_test[:, col]
                    col_vec_masked = ma.masked_array(col_vec, mask=mask_test)
                    X_test_masked[:, col] = col_vec_masked.compressed().resize((n_rows, 1))

                Y_test = Y_test_masked_compressed
                X_test = X_test_masked[~np.isnan(X_test_masked)]


            if not X_test.shape == (0,) and not Y_test.shape == (0,):
                Y_pred_gnb3 = gnb3.predict(X_test[:, 0:6])


                cm_gnb3 = confusion_matrix(Y_test, Y_pred_gnb3, labels=CLASSES)
                CM_full_gnb3 = CM_full_gnb3 + cm_gnb3


    print('CONFUSION MATRICES COMPUTED - SAVING CONFUSION MATRICES ...')

    np.savetxt('cm_full_gnb3_testSet.csv', CM_full_gnb3, delimiter=',')
