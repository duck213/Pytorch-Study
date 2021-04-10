import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)  # for reproducibility
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# there are two hypothesis sigmoid options
#hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W)+b)))  # first option
#hypothesis = torch.sigmoid(x_train.matmul(W) + b)  # second option

# 1. cost function
#losses = -(y_train * torch.log(hypothesis)) + (1 - y_train) * torch.log(1 - hypothesis)
#cost = losses.mean()

#F.binary_cross_entropy(hypothesis, y_train)  # binary ross entropy to lighten the burden

# optimizer setting
optimizer = optim.SGD([W, b], lr=1)

# 1. training with low-level binary cross entropy loss

epoch_num = 1000

for epoch in range(epoch_num + 1):
    hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W)+b)))
    cost = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis)).mean()

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch {:4d}/{} Cost: {:.6f}'.format(epoch, epoch_num, cost.item()))


# 2. Training with F.binary_cross_entropy

for epoch in range(epoch_num + 1):
    hypothesis =  torch.sigmoid(x_train.matmul(W) + b)
    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch {:4d}/{} Cost: {:.6f}'.format(epoch, epoch_num, cost.item()))


# 3. training with Real Data using low-level binary cross entropy loss

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data_rd = xy[:, 0:-1]
y_data_rd = xy[:, [-1]]
x_train_rd = torch.FloatTensor(x_data_rd)
y_train_rd = torch.FloatTensor(y_data_rd)

W_rd = torch.zeros((8,1), requires_grad=True)
b_rd = torch.zeros(1, requires_grad=True)
optimizer_rd = optim.SGD([W_rd, b_rd], lr=1)

rd_epochs = 100

for epoch in range(rd_epochs + 1):
    hypothesis = 1 / (1 + torch.exp(-(x_train_rd.matmul(W_rd) + b_rd)))
    cost = -(y_train_rd * torch.log(hypothesis) +
             (1 - y_train_rd) * torch.log(1 - hypothesis)).mean()

    optimizer_rd.zero_grad()
    cost.backward()
    optimizer_rd.step()

    if epoch % 10 == 0:
        print('epoch {:4d}/{} Cost: {:.6f}'.format(epoch, rd_epochs, cost.item()))

# 4. training with Real Data using F.binary_cross_entropy

for epoch in range(rd_epochs + 1):
    hypothesis = torch.sigmoid(x_train_rd.matmul(W_rd) + b)
    cost = F.binary_cross_entropy(hypothesis, y_train_rd)

    optimizer_rd.zero_grad()
    cost.backward()
    optimizer_rd.step()

    if epoch % 10 == 0:
        print('epoch {:4d}/{} Cost: {:.6f}'.format(epoch, rd_epochs, cost.item()))


# 5. High-level Implementation with nn.Module

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = BinaryClassifier()

optim_model = optim.SGD(model.parameters(), lr=1)
nb_epochs = 100
for epoch in range(nb_epochs + 1):
    hps = model(x_train_rd)
    cst = F.binary_cross_entropy(hps, y_train_rd)

    optim_model.zero_grad()
    cst.backward()
    optim_model.step()

    if epoch % 10 == 0:
        prediction = hps >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train_rd

        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print('Epoch {:4d}/{} Cost: {:.6f} Acccuracy {:2.2f}%'
              .format(epoch, nb_epochs, cst.item(), accuracy * 100))


