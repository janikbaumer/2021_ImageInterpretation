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
                              num_workers=8,
                              shuffle=False)

    val_loader = DataLoader(dset_val,
                            batch_size=8,
                            num_workers=8,
                            shuffle=False)

    test_loader = DataLoader(dset_test,
                             batch_size=8,
                             num_workers=8,
                             shuffle=False)


    # create initialization of certain ML model

    ## SVC
    # clf = svm.SVC(kernel='linear', C=1)

    ## Naive Bayes:
    gnb = GaussianNB()





    #print(f"Number of mislabeled points out of "
    #      f"a total {X_val.shape[0]} points : {(y_val != y_pred).sum()}")

#    for x,y in tqdm(dset_train):
#        print('globi')
#        pass


    print('TRAINING STARTING')

    train_loader_loop = 0
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

            ## SVC
            # clf.fit(X_train, Y_train)

            ## Naive Bayes
            gnb.fit(X_train, Y_train)



            print()



    print('VALIDATION STARTING')

    val_loader_loop = 0
    # at this point, all subsamples have been gone through the training process of the image
    for x, y in tqdm(val_loader):
        val_loader_loop += 1
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
            X_val = np.array(X_lst).T[0]  # transpose to have one col per feature, first ele because this dimension was somehow added
            Y_val = np.array(Y_lst).T[0].ravel()  # same as above, plus ravel() to convert from col vector to 1D array (needed for some ML models)


            ### todo: with these X_val, Y_val, the model can be evaulated, (ev with confusion matrix???)



    # Please note that random shuffling (shuffle=True) -> random access.
    # this is slower than sequential reading (shuffle=False)
    # If you want to speed up the read performance but keep the data shuffled, you can reshape the data to a fixed window size
    # e.g. (-1,4,128,128) and shuffle once along the first dimension. Then read the data sequentially.
    # another option is to read the data into the main memory h5 = h5py.File(root, 'r', driver="core")


    # for plotting
    #f, axarr = plt.subplots(ncols=3, nrows=8)

    # for averaging confusion matrix

'''
#    # loop over all batches
#    for x, y in tqdm(train_loader): # tqdm: make loops show a smart progress meter by wrapping any iterable with tqdm(iterable)
#        # for averaging confusion matrix
#        counter += 1

#        # since pytorch originally wanted the data channel first
#        x = np.transpose(x, [0, 2, 3, 1])  # swap shapes so that afterward shape = (nmbr_imgs_in_batch, size_x, size_y, nmbr_channels)
#        x = x.numpy()  #  x is not yet ndarray - convert x from pytorch tensor to ndarray

#        # change no data (99) to 3, for plotting reasons
#        y = np.where(y == 99, 3, y)  # y is already ndarry


#        # loop over batch size of train_loader ((all images in this batch (?))
#        for i in range(len(x)):
#            x_batch = x[i]
#            y_batch = y[i]

#            # define shapes
#            x_shape = x_batch.shape
#            y_shape = y_batch.shape

#            # initialize lists in which features / labels are stored
#            X_lst = []
#            Y_lst = []


#            ##### todo CONTINUE HERE REFACTORING CODE
#            # create feature matrix X_train (can be used for ML model later)
#            # for each channel, stack features (e.g. R,G,B,NIR intensities) to column vector

#            for chn in range(x_shape[-1]):
#                x_batch_chn = x_batch[:,:,chn]
#                x_batch_chn.resize(x_shape[0]*x_shape[1], 1)
#                x_shape_resized = x_batch_chn.shape
#                X_lst.append(x_batch_chn)

#            y_batch.resize(y_shape[0]*y_shape[1], 1)
#            Y_lst.append(y_batch)

#            # define feature matrix and label vector
#            X_train = np.array(X_lst).T[0]  # transpose to have one col per feature, first ele because this dimension was somehow added
#            Y_train = np.array(Y_lst).T[0].ravel()  # same as above, plus ravel() to convert from col vector to 1D array (needed for some ML models)
#            print()



#            ### HERE ERRORS MIGHT BE STARTING ...


#            # Split dataset into training set and test set
#            X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2,
                                                                random_state=42)  # 80% training and 20% test

#            # Train the model using the training sets

#            #clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train.ravel())
#            #clf.score(X_val, y_val)

            #clf.fit(X_train, y_train)

#            # Predict the response for test dataset
#            #y_pred = clf.predict(X_val)

#            # Naive Bayes:
#            gnb = GaussianNB()
#            y_pred = gnb.fit(X_train, y_train).predict(X_val)
#            print(f"Number of mislabeled points out of "
#                  f"a total {X_val.shape[0]} points : {(y_val != y_pred).sum()}")



#            # confusion matrix
#            # truth: vertical axis / predicted: horizontal axis
#            try:
#                 conf_mat = conf_mat + confusion_matrix(y_val, y_pred, labels=[0,1,2,3]) # labels 0 = background, 1 = palm oil, 2 = clouds and 3 = no data
#            except NameError:
#                conf_mat = np.zeros((4,4))

#            print('Confusion matrix: \n', conf_mat)

#            # for plotting
#            #axarr[i, 0].imshow(x[i, :, :, :3])  # first column: RGB
#            #axarr[i, 1].imshow(x[i, :, :, -1])  # second column: NIR
#            #axarr[i, 2].imshow(colormap[y[i]] / 255)  # third column:
#            #plt.show()

#        # for plotting
#        #plt.show()
#        # quit()

#    print(f'\ncnt (looped over train_loader) = {counter} times')
#    print('AVG confusion matrix: \n', conf_mat/counter)

'''






















'''
#Gain access to the data.
#Note: This does *not* load the entire data set into memory.
dset_train = h5py.File(FILE_TRAIN,"r")
dset_test = h5py.File(FILE_TEST, "r")

#The array GT contains the values 0 = background, 1 = palm oil and 99 = no data.


RGB_train = dset_train["RGB"]  # type: #HDF5 dataset / shape (2, 10980, 10980, 3)
NIR_train = dset_train["NIR"]
CLD_train = dset_train["CLD"]
GT_train = dset_train["GT"]

RGB_test = dset_test["RGB"]
NIR_test = dset_test["NIR"]
CLD_test = dset_test["CLD"]
GT_test = dset_test["GT"]

print('keys of dset_train: ', dset_train.keys())
print('keys of dset_test: ', dset_test.keys())

#Let's create an input-label pair:
#first the input by concatenating the RGB and NIR channels.

# expand NIR image in last dimension
NIR_train_expanded = np.expand_dims(NIR_train, axis=-1)
input_image_train = np.concatenate([RGB_train, NIR_train_expanded], axis=-1)

NIR_test_expanded = np.expand_dims(NIR_test, axis=-1)
input_image_test = np.concatenate([RGB_test, NIR_test_expanded], axis=-1)

print(np.shape(input_image_train))
print(type(input_image_train))



### GET FEATURES AND LABELS

# todo: so far, label is only created with one image
#  create with each image in dataset and stack them together


X_train = create_features(RGB_train, RGB_test, input_image_train)


Y_train = create_label(CLD_train, GT_train, input_image_train)

Y_test = create_label(CLD_test, GT_test, input_image_test)


### PREPROCESSING DATA


# ADDING AN NDVI CHANNEL

# FURTHER PREPROCESSING (FEATURE EXTRACTION)


### EVALUATION

# CREATE CONFUSION MATRIX

print(Y_train)
print(Y_test)
print()

'''
