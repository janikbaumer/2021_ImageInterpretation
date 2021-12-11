import torch.utils.data
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import math

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
            self.eval_mode=True
            self.num_pixels = target_shape[0]*target_shape[1]*target_shape[2]
        else:
            self.num_pixels = target_shape[0]

        label_idxs = np.unique(self.data["gt"])
        self.n_classes = len(label_idxs)
        #self.temporal_length = data_shape[-2]//time_downsample_factor
        self.temporal_length = math.ceil(data_shape[-2]/time_downsample_factor)

        print('Number of pixels: ', self.num_pixels)
        print('Number of classes: ', self.n_classes)
        print('Temporal length: ', self.temporal_length)
        print('Number of channels: ', self.num_channel)

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
        #target.apply_(mapping_dict.get)

        # if self.eval_mode:
        #     X = X.view()
        #     target = target.view()

        # Temporal down-sampling
        X = X[...,0::self.time_downsample_factor, :self.num_channel]

        # keep values between 0-1
        X = X * 1e-4

        return X.float(), target.long()

colordict = {'B04': '#a6cee3', 'NDWI': '#1f78b4', 'NDVI': '#b2df8a', 'RATIOVVVH': '#33a02c', 'B09': '#fb9a99',
             'B8A': '#e31a1c', 'IRECI': '#fdbf6f', 'B07': '#ff7f00', 'B12': '#cab2d6', 'B02': '#6a3d9a', 'B03': '#0f1b5f',
             'B01': '#b15928', 'B10': '#005293', 'VH': '#98c6ea', 'B08': '#e37222', 'VV': '#a2ad00', 'B05': '#69085a',
             'B11': '#007c30', 'NDVVVH': '#00778a', 'BRIGHTNESS': '#000000', 'B06': '#0f1b5f'}
plotbands = ["B02", "B03", "B04", "B08"]

label_IDs_old = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
          26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51]

label_names_old = ['Unknown', 'Apples', 'Beets', 'Berries', 'Biodiversity area', 'Buckwheat',
               'Chestnut', 'Chicory', 'Einkorn wheat', 'Fallow', 'Field bean', 'Forest',
               'Gardens', 'Grain', 'Hedge', 'Hemp', 'Hops', 'Legumes', 'Linen', 'Lupine',
               'Maize', 'Meadow', 'Mixed crop', 'Multiple', 'Mustard', 'Oat', 'Pasture', 'Pears',
               'Peas', 'Potatoes', 'Pumpkin', 'Rye', 'Sorghum', 'Soy', 'Spelt', 'Stone fruit',
               'Sugar beet', 'Summer barley', 'Summer rapeseed', 'Summer wheat', 'Sunflowers',
               'Tobacco', 'Tree crop', 'Vegetables', 'Vines', 'Wheat', 'Winter barley',
               'Winter rapeseed', 'Winter wheat']

dictionary_toDel = dict(zip(label_IDs_old, label_names_old))

label_IDs = [21, 51, 20, 27, 38, 49, 50, 45, 30, 48, 42, 46, 36]
lst_0_12 = list(range(len(label_IDs)))
label_names = ['Meadow', 'Winter wheat', 'Maize', 'Pasture', 'Sugar beet', 'Winter barley', 'Winter rapeseed', 'Vegetables', 'Potatoes',
      'Wheat', 'Sunflowers', 'Vines', 'Spelt']
mapping_dict = dict(zip(label_IDs, lst_0_12))


def plot_bands(X):
    x = np.arange(X.shape[0])
    for i, band in enumerate(plotbands):
        plt.plot(x, X[:, i])

    plt.savefig("bands.png", dpi=300, format="png", bbox_inches='tight')

if __name__ == "__main__":

    data_path = "/scratch2/tmehmet/imgint_testset_v2.hdf5"
    traindataset = Dataset(data_path)
    X,y = traindataset[0]

    print(X.shape)
    print(y.shape)

    gt_list = traindataset.return_labels()
    labels, pix_counts = np.unique(gt_list, return_counts=True)
    print(labels)
    print(pix_counts)

    inds = pix_counts.argsort()
    pix_counts_sorted = pix_counts[inds]
    labels_sorted = labels[inds]
    print(labels_sorted)


    label_names_sorted = [label_names[label_IDs.index(x)] for x in labels_sorted]
    print(label_names_sorted)

    fig = plt.figure()
    plt.bar(label_names_sorted, pix_counts_sorted)
    plt.xticks(rotation=90)
    plt.savefig("hist_test.png", dpi=300, format="png", bbox_inches='tight')
