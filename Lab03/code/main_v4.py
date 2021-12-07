"""
Andreas Brown, Fergus Dal, Janik Baumer
Image Interpretation, Lab 03

based on: https://mofanpy.com/tutorials/
"""
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.utils.data
import torch
from torch import nn
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import Dataset
from dataset import plot_bands
from sklearn.metrics import confusion_matrix

# import variables
from dataset import colordict, plotbands, label_IDs, label_names, mapping_dict

# fix random seed for reproducibility
torch.manual_seed(1)
np.random.seed(42)

#DOWNLOAD_MNIST = True   # set to True if haven't download the data

'''
# Mnist digital dataset
train_data = dsets.MNIST(
    root='./mnist/',
    train=True,                         # this is training data
    transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,            # download it if you don't have it
)

# plot one example
print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()
'''


############# LOAD DATA #############
PATH_TRAIN = '../data/imgint_trainset_v2.hdf5'
PATH_TEST = '../data/imgint_testset_v2.hdf5'
traindataset = Dataset(PATH_TRAIN)
testdataset = Dataset(PATH_TEST)

### show some test shapes
X, y = traindataset[0]
print(X.shape)
print(y.shape)


### some stuff from TA
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


### some stuff from myself
n_pxl = traindataset.num_pixels
n_chn = traindataset.num_channel
n_classes = traindataset.n_classes
temp_len = traindataset.temporal_length


###########################################
########### HYPERPARAMETERS ###############
###########################################

EPOCHS = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = temp_len  # todo get rid of magic number         # rnn time step / image height
INPUT_SIZE = 4  # todo get rid of magic number      # rnn input size / image width
LR = 0.001               # learning rate
NUM_LAYERS = 1

'''
### some plotting
fig = plt.figure()
plt.bar(label_names_sorted, pix_counts_sorted)
plt.xticks( rotation=90)
plt.savefig("hist_test.png", dpi=300, format="png", bbox_inches='tight')
'''

# Data Loader for easy mini-batch return in training
train_loader = DataLoader(dataset=traindataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=testdataset, batch_size=BATCH_SIZE, shuffle=True)
# Data Loader for easy mini-batch return in training - FROM DEMO
# train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


# convert test data into Variable, pick 2000 samples to speed up testing - FROM DEMO
# test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
# test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.   # shape (2000, 28, 28) value in range(0,1)
# test_y = test_data.test_labels.numpy()[:2000]    # covert to numpy array


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        '''
        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=NUM_LAYERS,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        '''
        self.rnn = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )

        #self.rnn = nn.GRU()


        self.out = nn.Linear(64, n_classes)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        #r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        r_out, h_n = self.rnn(x, None)

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()      # the target label is not one-hotted

# training and testing
for epoch in range(EPOCHS):
    nsamples_break = 5000
    for step, (b_x_train, b_y_train) in tqdm(enumerate(train_loader)):    # gives batch data
        b_y_train.apply_(mapping_dict.get)
        if step > nsamples_break:
            break
        b_x_train = b_x_train.view(-1, TIME_STEP, INPUT_SIZE)       # reshape x to (batch, time_step, input_size), # the size -1 is inferred from other dimensions

        output_train = rnn(b_x_train)                               # rnn output (of batch of traindata)
        loss = loss_func(output_train, b_y_train)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 50 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())


    accuracies = []
    cm_total = np.zeros((n_classes, n_classes))
    for step, (b_x_test, b_y_test) in tqdm(enumerate(test_loader)):
        b_x_test = b_x_test.view(-1, 71, 4)

        output_test = rnn(b_x_test)                   # (samples, time_step, input_size)
        pred_y = torch.max(output_test, 1)[1].data.numpy()

        accuracy = float((pred_y == b_y_test.data.numpy().flatten()).astype(int).sum()) / float(b_y_test.data.numpy().flatten().size)
        print('Epoch: ', epoch, 'test accuracy: %.2f' % accuracy)
        accuracies.append(accuracy)

        # confusion matrix
        cm = confusion_matrix(b_y_test.data.numpy().reshape(-1), pred_y, labels=list(range(n_classes)))
        cm_total = cm_total + cm
        # todo: collect all accuracies over this loop (test_loader) -> compute average (or more generally: call aggregation function) over all acc. of this loop
        #  get aggregated accuracy score
        #  or other metrics (confusion matrix etc)
        #  start comparing values (different model / different time_sample_factors / ... )
    avg_acc = np.mean(accuracies)
    print('avg accuracy: \n', avg_acc)
    print('cm total: \n', cm_total)
'''
# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
'''