"""
Andreas Brown, Fergus Dal, Janik Baumer
Image Interpretation, Lab 03
"""
import os.path

import torch
from torch import nn
#import torchvision.datasets as dsets
#import torchvision.transforms as transforms
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
from aggregate import calc_metrics
from time import time
# import variables
from dataset import colordict, plotbands, label_IDs, label_names, mapping_dict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

'''
# plot one example
print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()
'''
'''
### show some test shapes
X, y = traindataset[0]
print(X.shape)
print(y.shape)
'''
'''
### some plotting
fig = plt.figure()
plt.bar(label_names_sorted, pix_counts_sorted)
plt.xticks( rotation=90)
plt.savefig("hist_test.png", dpi=300, format="png", bbox_inches='tight')
'''

###########################################
########### HYPERPARAMETERS ###############
###########################################

# to vary, values are now got from the loop
###MODEL_TYPE = 'LSTM'          # try GRU or LSTM or RNN
###NUM_LAYERS = 2              # try 1 and 2
###TDS_FACTOR = 1              # time downsampling factor, try 1 (no downsampling), 4, (16)

# not to vary
LR = 0.001
INPUT_SIZE = 4
EPOCHS = 3  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
HIDDEN_SIZE = 128  # ev try 64 and 128

NSAMPLES_BREAK = 50


### LISTS FOR LOOP ###
MODELS_LST = ['RNN', 'GRU', 'LSTM']
NUM_LAYERS_LST = [1, 2]
TDS_FACTOR_LST = [1, 4]

for MODEL_TYPE in MODELS_LST:
    for NUM_LAYERS in NUM_LAYERS_LST:
        for TDS_FACTOR in TDS_FACTOR_LST:
            if NUM_LAYERS == 2 and TDS_FACTOR == 4:
                continue
            else:
                ###########################################
                ############### VARIOUS ###################
                ###########################################

                # fix random seed for reproducibility
                torch.manual_seed(1)
                np.random.seed(42)

                # set device for pytorch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

                ############# LOAD DATA #############
                PATH_TRAIN = r'C:\Users\jbaumer\PycharmProjects\2021_ImageInterpretation\Lab03\data\imgint_trainset_v3.hdf5'
                PATH_VAL = r'C:\Users\jbaumer\PycharmProjects\2021_ImageInterpretation\Lab03\data\imgint_validationset_v3.hdf5'
                PATH_TEST = r'C:\Users\jbaumer\PycharmProjects\2021_ImageInterpretation\Lab03\data\imgint_testset_v3.hdf5'
                traindataset = Dataset(PATH_TRAIN, time_downsample_factor=TDS_FACTOR)
                validationdataset = Dataset(PATH_VAL, time_downsample_factor=TDS_FACTOR)
                testdataset = Dataset(PATH_TEST, time_downsample_factor=TDS_FACTOR)

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

                ### get some parameters
                n_pxl_train = traindataset.num_pixels
                n_chn_train = traindataset.num_channel
                n_classes_train = traindataset.n_classes
                temp_len_train = traindataset.temporal_length

                n_pxl_val = validationdataset.num_pixels
                n_chn_val = validationdataset.num_channel
                n_classes_val = validationdataset.n_classes
                temp_len_val = validationdataset.temporal_length


                TIME_STEP = temp_len_train  # try time sample factor 16, 4, 1
                PATH_MODEL = f'../models/model_{MODEL_TYPE}_nlayers_{NUM_LAYERS}_templength_{TIME_STEP}.pkl'

                # Data Loader for easy mini-batch return in training
                train_loader = DataLoader(dataset=traindataset, batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(dataset=validationdataset, batch_size=BATCH_SIZE, shuffle=True)
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

                        self.rnn = nn.RNN(
                            input_size=INPUT_SIZE,
                            hidden_size=HIDDEN_SIZE,
                            num_layers=NUM_LAYERS,
                            batch_first=True,
                        )

                        self.out = nn.Linear(HIDDEN_SIZE, n_classes_train)

                    def forward(self, x):
                        # x shape (batch, time_step, input_size)
                        # r_out shape (batch, time_step, output_size)
                        # h_n shape (n_layers, batch, hidden_size)
                        # h_c shape (n_layers, batch, hidden_size)
                        #r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
                        r_out, h_n = self.rnn(x)  # final hidden state for each layer

                        # choose r_out at the last time step
                        out = self.out(h_n[-1, :, :])  # takes final hidden state of all layers
                        return out

                class GRU(nn.Module):
                    def __init__(self):
                        super(GRU, self).__init__()

                        self.rnn = nn.GRU(
                            input_size=INPUT_SIZE,
                            hidden_size=HIDDEN_SIZE,
                            num_layers=NUM_LAYERS,
                            batch_first=True,
                        )

                        self.out = nn.Linear(HIDDEN_SIZE, n_classes_train)

                    def forward(self, x):
                        # x shape (batch, time_step, input_size)
                        # r_out shape (batch, time_step, output_size)
                        # h_n shape (n_layers, batch, hidden_size)
                        # h_c shape (n_layers, batch, hidden_size)
                        #r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
                        r_out, h_n = self.rnn(x)  # final hidden state for each layer

                        # choose r_out at the last time step
                        out = self.out(h_n[-1, :, :])  # takes final hidden state of all layers
                        return out


                class LSTM(nn.Module):
                    def __init__(self):
                        super(LSTM, self).__init__() # call init function from parent class

                        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
                            input_size=INPUT_SIZE,
                            hidden_size=HIDDEN_SIZE,         # rnn hidden unit
                            num_layers=NUM_LAYERS,  # number of rnn layer
                            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
                        )

                        self.out = nn.Linear(HIDDEN_SIZE, n_classes_train)

                        #self.hn = torch.randn(2, 3, 20)
                        #self.cn = torch.randn(2, 4, 30)

                    def forward(self, x):
                        # x shape (batch, time_step, input_size)
                        # r_out shape (batch, time_step, output_size)
                        # h_n shape (n_layers, batch, hidden_size)
                        # h_c shape (n_layers, batch, hidden_size)
                        output, (self.hn, self.cn) = self.rnn(x)
                        #r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
                        #r_out, h_c = self.rnn(x, None)   # None represents zero initial hidden state

                        # choose r_out at the last time step
                        out = self.out(output[:, -1, :])  # apply the Linear classification layer in the end, take last temporal element
                        return out

                def get_metrics(y_true, y_pred):
                    precision = precision_score(y_true=y_true, y_pred=y_pred, average=None, zero_division=0)
                    recall = recall_score(y_true=y_true, y_pred=y_pred, average=None, zero_division=0)
                    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=None, zero_division=0)
                    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
                    return precision, recall, f1, accuracy



                #################################
                ########### MAIN CODE ###########
                #################################
                if os.path.isfile(PATH_MODEL):
                    # load trained model
                    print('MODEL ALREADY EXISTS, LOADING TRAINED MODEL...')
                    model = torch.load(f=PATH_MODEL)
                    model = model.to(device)

                else:
                    print('MODEL DOES NOT YET EXISTS, TRAINING MODEL FROM SCRATCH')
                    if MODEL_TYPE == 'GRU':
                        model = GRU()
                        model = model.to(device)
                    elif MODEL_TYPE == 'LSTM':
                        model = LSTM()
                        model = model.to(device)
                    else:
                        model = RNN()
                        model = model.to(device)

                    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.001)   # optimize all cnn parameters
                    loss_func = nn.CrossEntropyLoss()                                             # the target label is not one-hotted
                    loss_func = loss_func.to(device)
                    print(model)
                    print(optimizer)
                    print(loss_func)


                    model.train()

                    print('TRAINING STARTING ...')
                    t_train_start = time()

                    # training
                    for epoch in range(EPOCHS):
                        for step, (b_x_train, b_y_train) in tqdm(enumerate(train_loader)):    # gives batch data
                            b_x_train = b_x_train.to(device)
                            b_y_train = b_y_train.to(device)

                            #if step > NSAMPLES_BREAK:
                            #    break

                            # set gradients to zero for this step (not to have residuals from last loop)
                            optimizer.zero_grad()                           # clear gradients for this training step

                            b_y_train.apply_(mapping_dict.get)
                            b_x_train = b_x_train.view(-1, temp_len_train, INPUT_SIZE)       # reshape x to (batch, time_step, input_size), # the size -1 is inferred from other dimensions

                            output_train = model(b_x_train)                             # rnn output (of batch of traindata)
                            loss = loss_func(output_train, b_y_train)                   # cross entropy loss
                            loss.backward()                                             # backpropagation, compute gradients
                            optimizer.step()                                            # apply gradients - update parameters (weights)

                            if step % 50 == 0:
                                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
                    t_train_end = time()
                    t_train = t_train_end-t_train_start
                    print('training time in s: ', t_train)

                    print('MODEL TRAINED SUCCESSFULLY, SAVING THE MODEL...')
                    torch.save(obj=model, f=PATH_MODEL)
                    print('MODEL SAVED SUCCESSFULLY')



                print('VALIDATION STARTING')
                model.eval()  # same as model.train(mode=False)

                # lists to store each predictions and labels in a list
                pred_arr = np.empty(0)
                target_arr = np.empty(0)
                t_val_start = time()
                for step, (b_x_val, b_y_val) in tqdm(enumerate(val_loader)):
                    b_x_val = b_x_val.to(device)
                    b_y_val = b_y_val.to(device)

                    if step > NSAMPLES_BREAK:
                        break

                    b_y_val.apply_(mapping_dict.get)
                    b_x_val = b_x_val.view(-1, temp_len_val, INPUT_SIZE)

                    output_val = model(b_x_val)                                 # (samples, time_step, input_size)
                    pred_y = torch.max(output_val, 1)[1].data.numpy()           # prediction
                    pred_arr = np.append(pred_arr, pred_y)
                    target_arr = np.append(target_arr, b_y_val)


                # after this loop, we have pred_arr, target_arr with all values from all batches (so from whole validation set)
                t_val_end = time()
                t_val = t_val_end-t_val_start
                print('validation time: ', t_val)

                confusion_matrix = confusion_matrix(y_true=target_arr, y_pred=pred_arr) #  , labels=list(range(n_classes_val)))
                precision, recall, f1, accuracy = get_metrics(y_true=target_arr, y_pred=pred_arr)

                print('SAVING PREDICTIONS...')
                BASE_PATH_RESULTS = f'../results/model_{MODEL_TYPE}_nlayers_{NUM_LAYERS}_templength_{TIME_STEP}'

                # save validation time
                np.savetxt(fname=f'{BASE_PATH_RESULTS}_time_val.txt', X=(t_val,))

                # save confusion matrix
                np.savetxt(fname=f'{BASE_PATH_RESULTS}_confusion_matrix.txt', X=confusion_matrix)

                # save target array and predicted array
                np.savetxt(fname=f'{BASE_PATH_RESULTS}_y_true.txt', X=target_arr)
                np.savetxt(fname=f'{BASE_PATH_RESULTS}_y_pred.txt', X=pred_arr)

                # save f1 scores, accuracy, precisions, recalls
                np.savetxt(fname=f'{BASE_PATH_RESULTS}_f1.txt', X=f1)
                np.savetxt(fname=f'{BASE_PATH_RESULTS}_accuracies.txt', X=(accuracy,))
                np.savetxt(fname=f'{BASE_PATH_RESULTS}_precisions.txt', X=precision)
                np.savetxt(fname=f'{BASE_PATH_RESULTS}_recalls.txt', X=recall)

                print()


                """
                    accuracy = float((pred_y == b_y_test.data.numpy().flatten()).astype(int).sum()) / float(b_y_test.data.numpy().flatten().size)
                    print('Epoch: ', epoch, 'test accuracy: %.2f' % accuracy)
                    accuracies_test.append(accuracy)
                
                    # confusion matrix
                    cm = confusion_matrix(b_y_test.data.numpy().reshape(-1), pred_y, labels=list(range(n_classes_train)))
                    cm_total = cm_total + cm
                    # todo: collect all accuracies over this loop (test_loader) -> compute average (or more generally: call aggregation function) over all acc. of this loop
                    #  get aggregated accuracy score
                    #  or other metrics (confusion matrix etc)
                    #  start comparing values (different model / different time_sample_factors / ... )
                avg_acc_test = np.mean(accuracies_test)
                print('avg accuracy: \n', avg_acc_test)
                print('cm total: \n', cm_total)
                '''
                # print 10 predictions from test data
                test_output = rnn(test_x[:10].view(-1, 28, 28))
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                print(pred_y, 'prediction number')
                print(test_y[:10], 'real number')
                '''
                """
print()