### IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import os
import cv2
import pickle as cPickle


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
from skimage.filters import prewitt_h,prewitt_v
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.impute import KNNImputer
from scipy import ndimage
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

### Functions

def mean_pix(imgmat: np.array) -> np.array:
    (nrow,ncol,nchan)= imgmat.shape    
    mean_pix_value = np.zeros((nrow, ncol), dtype=float)
    #dtypeDict = {"uint8":(2**8-1), "uint16":(2**16-1)}
    for i in range(nchan):
        #imgmat[:,:,i] = imgmat[:,:,i]/dtypeDict[str(imgmat[:,:,i].dtype)]
        mean_pix_value += imgmat[:,:,i]
    mean_pix_value /= i
    return mean_pix_value

def window_level_function(image, level):
    
    
    minimum = np.min(image[:]) #Minimum intensity value in the image
    maximum = np.max(image[:]) #Maximum intensity value in the image
    
    window = np.std(image)  # Define window as +/- std
    
    #Clip or remove everything before or after the window boundaries
    image = np.clip(image, (level-(window/2)), (level+(window/2))) 
    #Everything below left limit becomes black
    #Everyting above right limit becomes white
    
    m = (maximum-minimum)/window;  # Slope of the window level transfer function
    b = maximum - (m * (level + (window/2))) # y-intercept of the window level transfer function

    image = m * image + b # The remaining, non-scaled values are adjusted by applying a linear transformation
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

    
    NDVI = (NIR.astype(float)-RED.astype(float))/(NIR+RED)
    NDVI = np.nan_to_num(NDVI)
    #imputer = KNNImputer(n_neighbors = 50)
    #NDVI = imputer.fit_transform(NDVI)
    return NDVI


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
cwd = os.getcwd()
FILE_TRAIN = '../datasets/dataset_train_reduced.h5'
FILE_TRAIN = '../datasets/dataset_train_devel.h5'
FILE_TRAIN = '../datasets/dataset_train_decide_20imgs.h5'
FILE_VAL =  '../datasets/dataset_val.h5'
FILE_TEST = '../datasets/dataset_test.h5'



if __name__ == "__main__":
    '''
    # for plotting
    # for imshow (labels) 0: green/blue-ish (background) | 1: green (palm-oil tree) | 2: white (cloud) | 3: black (no data)
    colormap = [[47, 79, 79], [0, 255, 0], [255, 255, 255], [0, 0, 0]]
    colormap = np.asarray(colormap)  # convert list of lists to numpy array
    '''

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






    if not os.path.isfile('gnb1_classifier26_13.pkl'):  # if this file does not yet exist

        print('INTIALIZATION STARTING ...')

        # create initialization of certain ML model

        ## Naive Bayes:
        gnb1 = GaussianNB()
        gnb2 = GaussianNB()
        gnb3 = GaussianNB()
        gnb4 = GaussianNB()
        gnb5 = GaussianNB()

        ## Support Vector Machine: SEEMS NOT TO WORK
        # svm = svm.SVC()

        ## Stochastic Gradient Descent SEEMS NOT TO WORK
        # sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)

        '''
        ## Decision tree:
        tree1 = tree.DecisionTreeClassifier()
        tree2 = tree.DecisionTreeClassifier()
        tree3 = tree.DecisionTreeClassifier()
        tree4 = tree.DecisionTreeClassifier()
        tree5 = tree.DecisionTreeClassifier()


        ## Random Forest:??

        ## Ensemble:??

        ## K-Nearest neighbor:
        n_neigh = 5
        neigh1 = KNeighborsClassifier(n_neighbors=n_neigh)
        neigh2 = KNeighborsClassifier(n_neighbors=n_neigh)
        neigh3 = KNeighborsClassifier(n_neighbors=n_neigh)
        neigh4 = KNeighborsClassifier(n_neighbors=n_neigh)
        neigh5 = KNeighborsClassifier(n_neighbors=n_neigh)
        '''

        print('TRAINING STARTING ...')

        train_loader_loop = 0

        for x, y in tqdm(train_loader): # tqdm: make loops show a smart progress meter by wrapping any iterable with tqdm(iterable)
            train_loader_loop += 1
            # print('train loader loop: ', train_loader_loop)
            x = np.transpose(x, [0, 2, 3, 1])  # swap shapes so that afterward shape = (nmbr_imgs_in_batch, size_x, size_y, nmbr_channels)
            x = x.numpy()  #  x is not yet ndarray - convert x from pytorch tensor to ndarray

            # change no data (99) to (3), for plotting reasons
            y = np.where(y == 99, 3, y)  # y is already ndarry


            # loop over batch size of train_loader ((all images in this batch (?))
            for i in range(len(x)):
                #print(f'training with image {i} of {len(x)} of train loader loop no: {train_loader_loop}')
                x_batch = x[i]
                y_batch = y[i]


                # FEATURE EXTRACTION
                #Grayscale as feature, adds 1 feat
                x_batch_gray = cv2.cvtColor(x_batch[:,:,0:3], cv2.COLOR_RGB2GRAY)
                x_batch = np.dstack((x_batch, x_batch_gray))

                #NDVI, adds 1 feat
                x_batch_ndvi = calc_ndvi(x_batch[:,:,3], x_batch[:,:,0])
                x_batch = np.dstack((x_batch, x_batch_ndvi))

                #Sobel axis 1
                sobel_axis0 = ndimage.sobel(x_batch[:,:,4], axis=0)
                x_batch = np.dstack((x_batch, sobel_axis0))

                #Sobel axis 2
                sobel_axis1 = ndimage.sobel(x_batch[:,:,4], axis=1)
                x_batch = np.dstack((x_batch, sobel_axis1))

                #Mean pixel intensity, all four original intensities, adds 1 feat
                x_batch_meanpix = mean_pix(x_batch[:,:,0:4])
                x_batch = np.dstack((x_batch, x_batch_meanpix))

                #Mean pixel intensity, RGB, adds 1 feat
                x_batch_meanpixRGB = mean_pix(x_batch[:,:,0:3])
                x_batch = np.dstack((x_batch, x_batch_meanpixRGB))

                #Prewitt horizontal edges, adds 1 feat
                edges_prewitt_horizontal = prewitt_h(x_batch[:,:,4])
                x_batch = np.dstack((x_batch, edges_prewitt_horizontal))

                #Prewitt vertical edges, adds 1 feat
                edges_prewitt_vertical = prewitt_v(x_batch[:,:,4])
                x_batch = np.dstack((x_batch, edges_prewitt_vertical))

                #Window-level scaling/shifting, adds 1 feat
                x_batch_wl = window_level_function(x_batch[:,:,4], 0.8)
                x_batch = np.dstack((x_batch, x_batch_wl))

                # convert nan values to 0
                x_batch = np.nan_to_num(x_batch)

                # could be commented
                nans = np.any(np.isnan(x_batch))
                print(nans)
                print()


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
                    x_batch_chn = np.resize(x_batch_chn, (x_shape[0]*x_shape[1], 1))
                    x_shape_resized = x_batch_chn.shape
                    X_lst.append(x_batch_chn)

                y_batch.resize(y_shape[0]*y_shape[1], 1)
                Y_lst.append(y_batch)

                # define feature matrix and label vector
                X_train = np.array(X_lst).T[0]  # transpose to have one col per feature, first ele because this dimension was somehow added
                Y_train = np.array(Y_lst).T[0].ravel()  # same as above, plus ravel() to convert from col vector to 1D array (needed for some ML models)

                # normalizing and standardizing the feature vectors
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)

                # train model

                ## Naive Bayes

                gnb1.fit(X_train[:,0:4], Y_train)
                gnb2.fit(X_train[:,0:5], Y_train)
                gnb3.fit(X_train[:,0:6], Y_train)
                gnb4.fit(X_train[:,0:8], Y_train)
                gnb5.fit(X_train, Y_train)

                '''
                tree1.fit(X_train[:,0:4], Y_train)
                tree2.fit(X_train[:,0:5], Y_train)
                tree3.fit(X_train[:,0:6], Y_train)
                tree4.fit(X_train[:,0:8], Y_train)
                tree5.fit(X_train, Y_train)
                
                neigh1.fit(X_train[:,0:4], Y_train)
                neigh2.fit(X_train[:,0:5], Y_train)
                neigh3.fit(X_train[:,0:6], Y_train)
                neigh4.fit(X_train[:,0:8], Y_train)
                neigh5.fit(X_train, Y_train)
                '''

    else:  # if those files exist, read them from disk
        print('FILES ALREADY EXISTS - READING MODELS FROM PICKLE FILES ...')
        gnb1 = cPickle.load(open('gnb1_classifier26_13.pkl', 'rb'))
        gnb2 = cPickle.load(open('gnb2_classifier26_13.pkl', 'rb'))
        gnb3 = cPickle.load(open('gnb3_classifier26_13.pkl', 'rb'))
        gnb4 = cPickle.load(open('gnb4_classifier26_13.pkl', 'rb'))
        gnb5 = cPickle.load(open('gnb5_classifier26_13.pkl', 'rb'))

        # todo: add other models to be read
        # tree and neigh
    # HERE models ARE COMPLETELY TRAINED
    
    if not os.path.isfile('gnb1_classifier26_13.pkl'):
        #Save the classifiers todo: this might duplicate models even if they already exist
        now = datetime.now()
        with open('gnb1_classifier' + now.strftime("%d_%H") + '.pkl', 'wb') as fid:
            cPickle.dump(gnb1, fid)
        with open('gnb2_classifier' + now.strftime("%d_%H") + '.pkl', 'wb') as fid:
            cPickle.dump(gnb2, fid)
        with open('gnb3_classifier' + now.strftime("%d_%H") + '.pkl', 'wb') as fid:
            cPickle.dump(gnb3, fid)
        with open('gnb4_classifier' + now.strftime("%d_%H") + '.pkl', 'wb') as fid:
            cPickle.dump(gnb4, fid)
        with open('gnb5_classifier' + now.strftime("%d_%H") + '.pkl', 'wb') as fid:
            cPickle.dump(gnb5, fid)


    '''
    with open('tree1_classifier' + now.strftime("%d_%H") + '.pkl', 'wb') as fid:
        cPickle.dump(tree1, fid)
    with open('tree2_classifier' + now.strftime("%d_%H") + '.pkl', 'wb') as fid:
        cPickle.dump(tree2, fid)
    with open('tree3_classifier' + now.strftime("%d_%H") + '.pkl', 'wb') as fid:
        cPickle.dump(tree3, fid)
    with open('tree4_classifier' + now.strftime("%d_%H") + '.pkl', 'wb') as fid:
        cPickle.dump(tree4, fid)
    with open('tree5_classifier' + now.strftime("%d_%H") + '.pkl', 'wb') as fid:
        cPickle.dump(tree5, fid)
        
    with open('neigh1_classifier' + now.strftime("%d_%H") + '.pkl', 'wb') as fid:
        cPickle.dump(neigh1, fid)
    with open('neigh2_classifier' + now.strftime("%d_%H") + '.pkl', 'wb') as fid:
        cPickle.dump(neigh2, fid)
    with open('neigh3_classifier' + now.strftime("%d_%H") + '.pkl', 'wb') as fid:
        cPickle.dump(neigh3, fid)
    with open('neigh4_classifier' + now.strftime("%d_%H") + '.pkl', 'wb') as fid:
        cPickle.dump(neigh4, fid)
    with open('neigh5_classifier' + now.strftime("%d_%H") + '.pkl', 'wb') as fid:
        cPickle.dump(neigh5, fid)
    '''
        
    

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
    
    CM_full_gnb1 = np.zeros((4,4))
    CM_full_gnb2 = np.zeros((4, 4))
    CM_full_gnb3 = np.zeros((4, 4))
    CM_full_gnb4 = np.zeros((4, 4))
    CM_full_gnb5 = np.zeros((4, 4))

    '''
    CM_full_tree1 = np.zeros((4, 4))
    CM_full_tree2 = np.zeros((4, 4))
    CM_full_tree3 = np.zeros((4, 4))
    CM_full_tree4 = np.zeros((4, 4))
    CM_full_tree5 = np.zeros((4, 4))

    CM_full_neigh1 = np.zeros((4, 4))
    CM_full_neigh2 = np.zeros((4, 4))
    CM_full_neigh3 = np.zeros((4, 4))
    CM_full_neigh4 = np.zeros((4, 4))
    CM_full_neigh5 = np.zeros((4, 4))
    '''

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

            # FEATURE EXTRACTION
            # Grayscale as feature, adds 1 feat
            x_va_batch_gray = cv2.cvtColor(x_va_batch[:, :, 0:3], cv2.COLOR_RGB2GRAY)
            x_va_batch = np.dstack((x_va_batch, x_va_batch_gray))

            # NDVI, adds 1 feat
            x_va_batch_ndvi = calc_ndvi(x_va_batch[:, :, 3], x_va_batch[:, :, 0])
            x_va_batch = np.dstack((x_va_batch, x_va_batch_ndvi))

            # Sobel axis 1
            sobel_axis0_va = ndimage.sobel(x_va_batch[:, :, 4], axis=0)
            x_va_batch = np.dstack((x_va_batch, sobel_axis0_va))

            # Sobel axis 2
            sobel_axis1_va = ndimage.sobel(x_va_batch[:, :, 4], axis=1)
            x_va_batch = np.dstack((x_va_batch, sobel_axis1_va))

            # Mean pixel intensity, all four original intensities, adds 1 feat
            x_batch_meanpix_va = mean_pix(x_va_batch[:, :, 0:4])
            x_va_batch = np.dstack((x_va_batch, x_batch_meanpix_va))

            # Mean pixel intensity, RGB, adds 1 feat
            x_batch_meanpixRGB_va = mean_pix(x_va_batch[:, :, 0:3])
            x_va_batch = np.dstack((x_va_batch, x_batch_meanpixRGB_va))

            # Prewitt horizontal edges, adds 1 feat
            edges_prewitt_horizontal_va = prewitt_h(x_va_batch[:, :, 4])
            x_va_batch = np.dstack((x_va_batch, edges_prewitt_horizontal_va))

            # Prewitt vertical edges, adds 1 feat
            edges_prewitt_vertical_va = prewitt_v(x_va_batch[:, :, 4])
            x_va_batch = np.dstack((x_va_batch, edges_prewitt_vertical_va))

            # Window-level scaling/shifting, adds 1 feat
            x_batch_wl_va = window_level_function(x_va_batch[:, :, 4], 0.8)
            x_va_batch = np.dstack((x_va_batch, x_batch_wl_va))

            # convert nan values to 0
            x_va_batch = np.nan_to_num(x_va_batch)

            # define shapes
            x_va_shape = x_va_batch.shape
            y_va_shape = y_va_batch.shape

            # initialize lists in which features / labels are stored
            X_val_lst = []
            Y_val_lst = []

            # create feature matrix X_train (can be used for ML model later)
            # for each channel, stack features (e.g. R,G,B,NIR intensities) to column vector
            for chn in range(x_va_shape[-1]):
                x_va_batch_chn = x_va_batch[:, :, chn]
                x_va_batch_chn.resize(x_va_shape[0]*x_va_shape[1], 1)
                x_va_shape_resized = x_va_batch_chn.shape
                X_val_lst.append(x_va_batch_chn)

            y_va_batch.resize(y_va_shape[0]*y_va_shape[1], 1)
            Y_val_lst.append(y_va_batch)


            # define feature matrix and label vector
            X_val = np.array(X_val_lst).T[0]  # transpose to have one col per feature, first ele because this dimension was somehow added
            Y_val = np.array(Y_val_lst).T[0].ravel()  # same as above, plus ravel() to convert from col vector to 1D array (needed for some ML models)

            ## TODO
            #Make prediciton per model (5x3 models) and confusion matrices, rememeber to aggregate per model
            #SAVE CONFUSION MATRICES!!!!
            Y_pred_gnb1 = gnb1.predict(X_val[:, 0:4])
            Y_pred_gnb2 = gnb2.predict(X_val[:, 0:5])
            Y_pred_gnb3 = gnb3.predict(X_val[:, 0:6])
            Y_pred_gnb4 = gnb4.predict(X_val[:, 0:8])
            Y_pred_gnb5 = gnb5.predict(X_val)

            '''
            Y_pred_tree1 = tree1.predict(X_val[:, 0:4])
            Y_pred_tree2 = tree2.predict(X_val[:, 0:5])
            Y_pred_tree3 = tree3.predict(X_val[:, 0:6])
            Y_pred_tree4 = tree4.predict(X_val[:, 0:8])
            Y_pred_tree5 = tree5.predict(X_val)

            Y_pred_neigh1 = neigh1.predict(X_val[:, 0:4])
            Y_pred_neigh2 = neigh2.predict(X_val[:, 0:5])
            Y_pred_neigh3 = neigh3.predict(X_val[:, 0:6])
            Y_pred_neigh4 = neigh4.predict(X_val[:, 0:8])
            Y_pred_neigh5 = neigh5.predict(X_val)
            '''

            #Y_pred_svm = svm.predict(X_val)
            #Y_pred_sgd = sgd.predict(X_val)

    
            cm_gnb1 = confusion_matrix(Y_val, Y_pred_gnb1, labels=[0, 1, 2, 3])
            CM_full_gnb1 = CM_full_gnb1 + cm_gnb1
            cm_gnb2 = confusion_matrix(Y_val, Y_pred_gnb2, labels=[0, 1, 2, 3])
            CM_full_gnb2 = CM_full_gnb2 + cm_gnb2
            cm_gnb3 = confusion_matrix(Y_val, Y_pred_gnb3, labels=[0, 1, 2, 3])
            CM_full_gnb3 = CM_full_gnb3 + cm_gnb3
            cm_gnb4 = confusion_matrix(Y_val, Y_pred_gnb4, labels=[0, 1, 2, 3])
            CM_full_gnb4 = CM_full_gnb4 + cm_gnb4
            cm_gnb5 = confusion_matrix(Y_val, Y_pred_gnb5, labels=[0, 1, 2, 3])
            CM_full_gnb5 = CM_full_gnb5 + cm_gnb5

            '''
            cm_tree1 = confusion_matrix(Y_val, Y_pred_tree1, labels=[0, 1, 2, 3])
            CM_full_tree1 = CM_full_tree1 + cm_tree1
            cm_tree2 = confusion_matrix(Y_val, Y_pred_tree2, labels=[0, 1, 2, 3])
            CM_full_tree2 = CM_full_tree2 + cm_tree2
            cm_tree3 = confusion_matrix(Y_val, Y_pred_tree3, labels=[0, 1, 2, 3])
            CM_full_tree3 = CM_full_tree3 + cm_tree3
            cm_tree4 = confusion_matrix(Y_val, Y_pred_tree4, labels=[0, 1, 2, 3])
            CM_full_tree4 = CM_full_tree4 + cm_tree4
            cm_tree5 = confusion_matrix(Y_val, Y_pred_tree5, labels=[0, 1, 2, 3])
            CM_full_tree5 = CM_full_tree5 + cm_tree5


            cm_neigh1 = confusion_matrix(Y_val, Y_pred_neigh1, labels=[0, 1, 2, 3])
            CM_full_neigh1 = CM_full_neigh1 + cm_neigh1
            cm_neigh2 = confusion_matrix(Y_val, Y_pred_neigh2, labels=[0, 1, 2, 3])
            CM_full_neigh2 = CM_full_neigh2 + cm_neigh2
            cm_neigh3 = confusion_matrix(Y_val, Y_pred_neigh3, labels=[0, 1, 2, 3])
            CM_full_neigh3 = CM_full_neigh3 + cm_neigh3
            cm_neigh4 = confusion_matrix(Y_val, Y_pred_neigh4, labels=[0, 1, 2, 3])
            CM_full_neigh4 = CM_full_neigh4 + cm_neigh4
            cm_neigh5 = confusion_matrix(Y_val, Y_pred_neigh5, labels=[0, 1, 2, 3])
            CM_full_neigh5 = CM_full_neigh5 + cm_neigh5
            '''

            # compute f1-score for each batch
            #f1 = f1_score(Y_val, Y_pred)
            #f1_full += f1
            #print()

            # kappa
            #kappa = cohen_kappa_score(Y_val, Y_pred)
            #kappa_full += kappa
            #print('CM full: \n', CM_full)
            #print()


    print('Confusion matrices computed')
    print('SAVING CONFUSION MATRICES ...')

    np.savetxt('cm_full_gnb1.csv', CM_full_gnb1, delimiter=',')
    np.savetxt('cm_full_gnb2.csv', CM_full_gnb2, delimiter=',')
    np.savetxt('cm_full_gnb3.csv', CM_full_gnb3, delimiter=',')
    np.savetxt('cm_full_gnb4.csv', CM_full_gnb4, delimiter=',')
    np.savetxt('cm_full_gnb5.csv', CM_full_gnb5, delimiter=',')


    '''
    np.savetxt('cm_full_tree1.csv', CM_full_tree1, delimiter=',')
    np.savetxt('cm_full_tree2.csv', CM_full_tree2, delimiter=',')
    np.savetxt('cm_full_tree3.csv', CM_full_tree3, delimiter=',')
    np.savetxt('cm_full_tree4.csv', CM_full_tree4, delimiter=',')
    np.savetxt('cm_full_tree5.csv', CM_full_tree5, delimiter=',')

    np.savetxt('cm_full_neigh1.csv', CM_full_neigh1, delimiter=',')
    np.savetxt('cm_full_neigh2.csv', CM_full_neigh2, delimiter=',')
    np.savetxt('cm_full_neigh3.csv', CM_full_neigh3, delimiter=',')
    np.savetxt('cm_full_neigh4.csv', CM_full_neigh4, delimiter=',')
    np.savetxt('cm_full_neigh5.csv', CM_full_neigh5, delimiter=',')
    '''



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
