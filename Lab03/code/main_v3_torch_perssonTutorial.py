import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create Fully Connected Network
class NN(nn.Module):  # inherit from nn.Module
    def __init__(self, input_size, num_classes):  # input size will be 28x28=784
        super(NN, self).__init__()  # call init function of class we inherited from (nn.Module)
        self.fc1 = nn.Linear(input_size, 50)  # nn.Linear(in_features, out_features,...)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# test if model does what it's supposed to do
model = NN(784, 10)
x = torch.randn(64, 784)
print(model(x).shape)  # gets 64x10

# set device (cpu or gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparams
input_size = 784

input_size = 28  # todo
sequence_length = 28  # todo
num_layers = 2
hidden_size = 256

num_classes = 10
learning_rate = 0.001
batch_size = 1  # 64
num_epochs = 1

# Create a RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # input shape: batch_size x time_sequence x features
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward prop
        out, hidden_state = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)  # pass through linear layer
        return out

# load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  # shuffle: when it has gone through each img in training set, it shuffles the batches before going to the next epoch

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)  # shuffle: when it has gone through each img in training set, it shuffles the batches before going to the next epoch

# initialize network
#model = NN(input_size=input_size, num_classes=num_classes).to(device)
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# train network
for epoch in range(num_epochs):  # one epoch means that network has seen all the images in the dataset
    for batch_idx, (data, targets) in enumerate(train_loader):  # data: image, targets: label
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        #print('data shape: ', data.shape)  # torch.Size([batchSize, nChannels (1 for blackwhite, 3 for rgb), width, height])
        #print('targets shape: ', targets.shape)  # torch.Size([BatchSize])

        # get correct shape # only for fully connected
        # data = data.reshape(data.shape[0], -1)  # torch.Size([batchSize, nCh*w*h])

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()  # set all gradients to zero for each batch
        loss.backward()

        # gradient descent or adam step
        optimizer.step()  # update weights depending on gradients (calc in loss.backwards())
# check accuracy on training & test to see how good our model

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('check acc on train data')
    else:
        print('check acc on test data')

    num_correct = 0
    num_samples = 0
    model.eval()  # ?

    with torch.no_grad():  # to check accuracy, gradients do not have to be computed
        i = 0
        for x, y in loader:
            print('loop: ', i)
            i += 1
            x = x.to(device=device)
            y = y.to(device=device)

            x = x.reshape(x.shape[0], -1).squeeze(1)

            scores = model(x)  # the output of the model with input x, shape: 64x10

            _, predictions = scores.max(1)
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)
            print()
            print(f'got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

            model.train()
            #acc = model.train()
            #return acc

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)