from torchvision.datasets.vision import VisionDataset
import h5py
import numpy as np


class SatelliteSet(VisionDataset):

    def __init__(self, windowsize=128, test=False):
        self.wsize = windowsize
        super().__init__(None)
        self.num_smpls, self.sh_x, self.sh_y = 3,10980,10980  # size of each image

        self.pad_x = (self.sh_x - (self.sh_x % self.wsize))
        self.pad_y = (self.sh_y - (self.sh_y % self.wsize))
        self.sh_x = self.pad_x + self.wsize
        self.sh_y = self.pad_y + self.wsize
        self.num_windows = self.num_smpls * self.sh_x / self.wsize * self.sh_y / self.wsize
        self.num_windows = int(self.num_windows)
        self.has_data = False

    # ugly fix for working with windows
    # Windows cannot pass the h5 file to sub-processes, so each process must access the file itself.
    def load_data(self):
        h5 = h5py.File("dataset_test.h5", 'r')
        self.RGB = h5["RGB"]
        self.NIR = h5["NIR"]
        self.CLD = h5["CLD"]
        self.GT = h5["GT"]
        self.has_data = True

    def __getitem__(self, index):
        if not self.has_data:
            self.load_data()

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


if __name__ == "__main__":
    colormap = [[47, 79, 79], [0, 255, 0], [255, 255, 255], [0, 0, 0]]
    colormap = np.asarray(colormap)
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import torch



    dset = SatelliteSet( windowsize = 512,test=False)
    # create dataloader that samples batches from the dataset
    train_loader = torch.utils.data.DataLoader(dset,
                                               batch_size=8,
                                               num_workers=8,
                                               shuffle=True)

    # Please note that random shuffling (shuffle=True) -> random access.
    # this is slower than sequential reading (shuffle=False)
    # If you want to speed up the read performance but keep the data shuffled, you can reshape the data to a fixed window size
    # e.g. (-1,4,128,128) and shuffle once along the first dimension. Then read the data sequentially.
    # another option is to read the data into the main memory h5 = h5py.File(root, 'r', driver="core")

    # plot some examples
    f, axarr = plt.subplots(ncols=3, nrows=8)

    for x, y in tqdm(train_loader):
        x = np.transpose(x, [0, 2, 3, 1])
        y = np.where(y == 99, 3, y)
        for i in range(len(x)):
            axarr[i, 0].imshow(x[i, :, :, :3])
            axarr[i, 1].imshow(x[i, :, :, -1])
            axarr[i, 2].imshow(colormap[y[i]] / 255)

        plt.show()
        quit()
