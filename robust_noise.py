import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as F
import sys
import os

plt.style.use('seaborn')
seed = 3689
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#generating training and testing data
#a few dimensions where the two classes are separated and the rest are noise dimensions
#basically want to allow the possibility of learning adhoc features which may not be
#important for classification, but may be present in training data as it is limited
#This may be the cause behind sharp and flat minima(thats a hypothesis)
d = 1000
k = 5
d1 = 100
d2 = d-d1

sigma = 0.02
train_per_class = 1000
train_length = train_per_class*k
test_per_class = 1000
test_length = test_per_class*k
mu = np.random.multivariate_normal(np.zeros((d1)), np.eye(d1), k)
for i in range(k):
    mu[i] = mu[i]/np.linalg.norm(mu[i])

training_set = np.zeros((train_length, d))
train_labels = np.zeros((train_length))
val_set = np.zeros((test_length, d))
val_labels = np.zeros((test_length))

for i in range(k):
    training_set[i*train_per_class:(i+1)*train_per_class, :d1] = np.random.multivariate_normal(mu[i], sigma*np.eye(d1), train_per_class)
    training_set[i*train_per_class:(i+1)*train_per_class, d1:] = np.random.multivariate_normal(np.zeros((d2)),np.eye(d2), train_per_class)
    train_labels[i*train_per_class:(i+1)*train_per_class] = np.zeros((train_per_class)) + i
    val_set[i*test_per_class:(i+1)*test_per_class, :d1] = np.random.multivariate_normal(mu[i], sigma*np.eye(d1), test_per_class)
    val_set[i*test_per_class:(i+1)*test_per_class, d1:] = np.random.multivariate_normal(np.zeros((d2)), np.eye(d2), test_per_class)
    val_labels[i*test_per_class:(i+1)*test_per_class] = np.zeros((test_per_class)) + i

#uncomment this code for d=2 and k=2 to look at the classes
# colors = []
# for i in range(train_length):
#     if train_labels[i]==0:
#         colors.append('y')
#     else:
#         colors.append('r')
#
# plt.scatter(training_set[:,0], training_set[:, 1], c=colors)
# plt.show()

BN_reg = True
"""Use this if want to turn shuffle off in trainloader to look at the
regularization effect of Batch Normalization"""
if not BN_reg:
    p = np.random.permutation(train_length)
    training_set = training_set[p]
    train_labels = train_labels[p]

training_set = torch.tensor(training_set).type(torch.FloatTensor)
train_labels = torch.tensor(train_labels).type(torch.LongTensor)
val_set = torch.tensor(val_set).type(torch.FloatTensor)
val_labels = torch.tensor(val_labels).type(torch.LongTensor)

class MyCustomDataset(Dataset):
    """ Linearly separable dataset."""

    def __init__(self, train_data, test_data, train=True, transform=None):
        """
        Args:
            train_data: training data
            test_data: testing data
            train: whether training data required or nor
        """
        self.train_data = train_data
        self.test_data = test_data
        self.train = train
        self.transform = transform

    def __len__(self):
        if self.train:
            return (self.train_data[0].shape[0])
        else:
            return (self.test_data[0].shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.train:
            sample = (self.train_data[0][idx, :], self.train_data[1][idx])
        else:
            sample = (self.test_data[0][idx, :], self.test_data[1][idx])

        if self.transform:
            sample = self.transform(sample)

        return sample


dataset_train = MyCustomDataset([training_set,train_labels],[],True)
dataset_test = MyCustomDataset([],[val_set, val_labels],False)

BN_momentum = 0.1
num_layers = 2
neurons = 100
is_bias = True
BN=False
gamma_1= True
beta_1 = True
act = 'linear'
lr = 0.1
batch_size = 32
tot_epochs = 200
change_epoch = 20
store_dir = True

if BN:
    if BN_reg:
        dir = './Robust_noise/BN/with_regularization/' + str(train_per_class) + '/'
    else:
        dir = './Robust_noise/BN/without_regularization/' + str(train_per_class) + '/'
else:
    dir = './Robust_noise/non-BN/' + str(train_per_class) + '/'

CHECK_FOLDER = os.path.isdir(dir)

# If folder doesn't exist, then create it.
if not CHECK_FOLDER:
    os.makedirs(dir)
    print("created folder : ", dir)

if store_dir:
    sys.stdout = open(dir + "details.txt", "w")

class TestNet(nn.Module):
    def __init__(self, n_layers, d, neurons, k, is_bias, act):
        super(TestNet, self).__init__()
        self.linears = nn.ModuleList()
        if BN:
            if gamma_1:
                self.gamma = nn.ParameterList()
            if beta_1:
                self.beta = nn.ParameterList()
        self.running_mean = []
        self.running_var = []
        for i in range(n_layers):
            if i==0:
                if n_layers==1:
                    self.linears.append(nn.Linear(d, k, bias=is_bias))
                else:
                    self.linears.append(nn.Linear(d, neurons, bias=is_bias))
                    self.running_mean.append(torch.zeros(1,neurons).to(device))
                    self.running_var.append(torch.zeros(1,neurons).to(device))
                    if BN:
                        if gamma_1:
                            self.gamma.append(nn.Parameter(torch.ones(1, neurons).to(device)))
                        if beta_1:
                            self.beta.append(nn.Parameter(torch.zeros(1, neurons).to(device)))
            elif i==n_layers-1:
                self.linears.append(nn.Linear(neurons, k, bias=is_bias))
            else:
                self.linears.append(nn.Linear(neurons, neurons, bias=is_bias))
                self.running_mean.append(torch.zeros(1, neurons).to(device))
                self.running_var.append(torch.zeros(1, neurons).to(device))
                if BN:
                    if gamma_1:
                        self.gamma.append(nn.Parameter(torch.ones(1, neurons).to(device)))
                    if beta_1:
                        self.beta.append(nn.Parameter(torch.zeros(1, neurons).to(device)))
        self.layers = n_layers
        self.act = act

    def forward(self, x, update_mean_var=0):
        for i, l in enumerate(self.linears):
            x = l(x)
            mean = torch.mean(x, dim=0, keepdim=True)
            var = torch.var(x, dim=0, keepdim=True)
            if BN:
                if i!=self.layers-1:
                    if self.training:
                        x = (x-mean)/(torch.sqrt(var + 1e-5))
                        if update_mean_var:
                            self.running_mean[i] = BN_momentum*mean + (1-BN_momentum)*self.running_mean[i]
                            self.running_var[i] = BN_momentum*var + (1-BN_momentum)*self.running_var[i]
                    else:
                        x = (x-self.running_mean[i])/(torch.sqrt(self.running_var[i] + 1e-5))
                    if gamma_1 and beta_1:
                        x = self.gamma[i]*x + self.beta[i]
                    elif gamma_1:
                        x = self.gamma[i] * x
                    elif beta_1:
                        x = x + self.beta[i]

            if i!=self.layers-1 and self.act!='linear':
                if self.act=='relu':
                    x = F.relu(x)
                elif self.act=='hardtanh':
                    x = F.hardtanh(x)
        return x

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    if train_length%batch_size==0:
        train_steps = int(train_length/batch_size)
    else:
        train_steps = int(train_length/batch_size) + 1
    net = TestNet(num_layers, d, neurons, k, is_bias, act)
    net.to(device)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    old_sgrad = None
    old_grad = None

    #don't shuffle the examples if want to eliminate the regularization effect of BN
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                            shuffle=BN_reg, num_workers=0)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size = batch_size, shuffle=True, num_workers=0)

    criterion = torch.nn.functional.cross_entropy
    curr_index = 0
    optimizer = optim.SGD(net.parameters(), lr)
    best_val_acc = 0
    check = False
    best_epoch_past = 0
    for epoch in range(tot_epochs):  # loop over the dataset multiple times
        net = net.train()
        running_loss = 0.0
        acc = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = net(inputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            indices = torch.argmax(outputs, 1)
            acc = acc + torch.sum(indices == labels)
            running_loss += loss.item()
            if i % 500 == 499:
                t = 1
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / i))

        train_loss.append(running_loss / (i + 1))
        train_acc.append(acc.item() / train_length)
        print('[%d] Train loss: %.3f, Train acc: %.3f' %
              (epoch + 1, running_loss / (i + 1), acc.item() / train_length))
        net = net.eval()
        acc = 0.0
        total_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                total_loss += loss.item()
                indices = torch.argmax(outputs, 1)
                acc = acc + torch.sum(indices == labels)

        print('[%d] Validation loss: %.3f, Validation acc: %.3f' %
              (epoch + 1, total_loss / (i + 1), acc.item() / test_length))

        test_loss.append(total_loss / (i + 1))
        test_acc.append(acc.item() / test_length)
        if acc.item() / test_length > best_val_acc:
            best_val_acc = acc.item() / test_length
            best_epoch_past = 0
        else:
            best_epoch_past += 1

        if best_epoch_past == change_epoch:
            lr = lr * 0.1
            if lr < 1e-3:
                break
            print("At epoch " + str(epoch) + ", lr changed to " + str(lr))
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.1
            best_epoch_past = 0

print('Best_val_acc is ' + str(best_val_acc))

plt.plot(list(range(len(train_loss))), train_loss, label='train_loss')
plt.plot(list(range(len(train_loss))), test_loss, label='test_loss')
plt.legend(loc='upper right')
if store_dir:
    plt.savefig(dir + 'loss.png')
    plt.close()
else:
    plt.show()

plt.plot(list(range(len(train_loss))), train_acc, label='train_acc')
plt.plot(list(range(len(train_loss))), test_acc, label='test_acc')
plt.legend(loc='upper right')
if store_dir:
    plt.savefig(dir + 'acc.png')
    plt.close()
else:
    plt.show()

