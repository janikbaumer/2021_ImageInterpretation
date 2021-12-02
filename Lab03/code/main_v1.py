import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from torch.utils.data import DataLoader
from sklearn import model_selection

import torch.utils.data
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

# fix random seed for reproducibility
numpy.random.seed(42)

# load the dataset but only keep the top n words, zero the rest
##### CLASSES ######
class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, time_downsample_factor=1, num_channel=4):

        self.num_channel = num_channel
        self.time_downsample_factor = time_downsample_factor
        self.eval_mode = False
        # Open the data file
        self.data = h5py.File(path, "r", libver='latest', swmr=True)

        data_shape = self.data["data"].shape
        target_shape = self.data["gt"].shape
        self.num_samples = data_shape[0]

        if len(target_shape) == 3:
            self.eval_mode = True
            self.num_pixels = target_shape[0]*target_shape[1]*target_shape[2]
        else:
            self.num_pixels = target_shape[0]

        label_idxs = np.unique(self.data["gt"])
        self.n_classes = len(label_idxs)
        self.temporal_length = data_shape[-2]//time_downsample_factor

        print('Number of pixels: ', self.num_pixels)
        print('Number of classes: ', self.n_classes)
        print('Temporal length: ', self.temporal_length)
        print('Number of channels: ', self.num_channel)
        print(self.data['gt'].shape)
        print(self.data['data'].shape)

    def return_labels(self):
        return self.data["gt"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        X = self.data["data"][idx]
        target = self.data["gt"][idx]

        # Convert numpy array to torch tensor
        X = torch.from_numpy(X)
        target = torch.from_numpy(np.array(target)).float()

        # if self.eval_mode:
        #     X = X.view()
        #     target = target.view()

        # Temporal down-sampling
        X = X[...,0::self.time_downsample_factor, :self.num_channel]

        # keep values between 0-1
        X = X * 1e-4

        return X.float(), target.long()

####### FUNCTIONS ##########
def plot_bands(X):
    x = np.arange(X.shape[0])
    for i, band in enumerate(plotbands):
        plt.plot(x, X[:,i])

    plt.savefig("bands.png", dpi=300, format="png", bbox_inches='tight')


####### VARIABLES ##########
colordict = {'B04': '#a6cee3', 'NDWI': '#1f78b4', 'NDVI': '#b2df8a', 'RATIOVVVH': '#33a02c', 'B09': '#fb9a99',
             'B8A': '#e31a1c', 'IRECI': '#fdbf6f', 'B07': '#ff7f00', 'B12': '#cab2d6', 'B02': '#6a3d9a', 'B03': '#0f1b5f',
             'B01': '#b15928', 'B10': '#005293', 'VH': '#98c6ea', 'B08': '#e37222', 'VV': '#a2ad00', 'B05': '#69085a',
             'B11': '#007c30', 'NDVVVH': '#00778a', 'BRIGHTNESS': '#000000', 'B06': '#0f1b5f'}
plotbands = ["B02", "B03", "B04", "B08"]

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
          26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51]

label_names = ['Unknown', 'Apples', 'Beets', 'Berries', 'Biodiversity area', 'Buckwheat',
               'Chestnut', 'Chicory', 'Einkorn wheat', 'Fallow', 'Field bean', 'Forest',
               'Gardens', 'Grain', 'Hedge', 'Hemp', 'Hops', 'Legumes', 'Linen', 'Lupine',
               'Maize', 'Meadow', 'Mixed crop', 'Multiple', 'Mustard', 'Oat', 'Pasture', 'Pears',
               'Peas', 'Potatoes', 'Pumpkin', 'Rye', 'Sorghum', 'Soy', 'Spelt', 'Stone fruit',
               'Sugar beet', 'Summer barley', 'Summer rapeseed', 'Summer wheat', 'Sunflowers',
               'Tobacco', 'Tree crop', 'Vegetables', 'Vines', 'Wheat', 'Winter barley',
               'Winter rapeseed', 'Winter wheat']

if __name__ == "__main__":
    DATA_PATH = "../data/imgint_trainset_reduced.hdf5"
    BATCH_SIZE = 8
    traindataset = Dataset(DATA_PATH, time_downsample_factor=1)
    X_train, y_train = traindataset[0]  # X and y have type torch.Tensor / X: torch.Size([71,4]) and y: torch.Size([]), so y has no (empty) size
    #X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
    print('shape of X: ', X_train.shape)
    print('shape of y: ', y_train.shape)

    gt_list = traindataset.return_labels()
    labels, pix_counts = np.unique(gt_list, return_counts=True)

    inds = pix_counts.argsort()
    pix_counts_sorted = pix_counts[inds]
    labels_sorted = labels[inds]

    label_names_sorted = [label_names[labels.tolist().index(x)] for x in labels_sorted]

    # fig = plt.figure()
    # plt.bar(label_names_sorted, pix_counts_sorted)
    # plt.xticks(rotation=90)
    # plt.savefig("hist.png", dpi=300, format="png", bbox_inches='tight')

    dloader = DataLoader(traindataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)

    n_pxl = traindataset.num_pixels
    n_chn = traindataset.num_channel
    n_classes = traindataset.n_classes
    print()

    # create the model
    embedding_vector_length = 32
    model = Sequential()
    # Turns positive integers (indexes) into dense vectors of fixed size

    # model.add(Embedding(n_pxl, embedding_vector_length, input_length=n_chn))
    model.add(LSTM(100, input_length=n_chn))
    model.add(Dense(n_classes, activation='sigmoid'))

    # todo ev change metrics to custom f1 score
    model.compile(optimizer='adam', metrics=['accuracy']) #loss: default
    model.build()
    print(model.summary())
    print()


    for X_train, y_train in tqdm(dloader):
        # convert tensor to numpy
        X_train = X_train.numpy()
        y_train = y_train.numpy()
        model.fit(X_train, y_train, epochs=3, batch_size=64)
        # X has shape torch.Size([BATCH_SIZE, TEMPORAL_LENGTH, NUM_CHANNELS])
        # Y has shape torch.Size([BATCH_SIZE])
        #print(X.shape, Y.shape)
        #print()
        # todo work with input and output (ev sep into train and validation)

    # Final evaluation of the model
    scores = model.evaluate(X_val, y_val, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))


'''      
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
    
    
    # Load feature vectors and GT
    print('Starting Training')
    dset = Lab02FeatDset(FEAT_FILE)

    dataLoader = DataLoader(dset, batch_size=batchSize, num_workers=0, shuffle=True)
    model = None
    for Z, Y in tqdm(dataLoader):
        Z = Z.numpy()
        Y = Y.numpy()
'''