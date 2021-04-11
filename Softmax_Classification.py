import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)  # for reproducibility

x_mat = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],
           [1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
y_mat = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_mat)
y_train = torch.LongTensor(y_mat)

W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer setting
optimizer = optim.SGD([W, b], lr=0.1)

# 1. training with low-level binary cross entropy loss

epoch_num = 1000
for epoch in range(epoch_num + 1):
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)
    y_one_hot = torch.zeros_like(hypothesis)
    y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
    cost = (y_one_hot * -torch.log(F.softmax(hypothesis, dim=1))).sum(dim=1).mean()

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch {:4d}/{} Cost: {:.6f}'.format(epoch, epoch_num, cost.item()))


# 2. training with F.cross_entropy

for epoch in range(epoch_num + 1):
    z = x_train.matmul(W) + b

    cost_F = F.cross_entropy(z, y_train)

    optimizer.zero_grad()
    cost_F.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch {:4d}/{} Cost: {:.6f}'.format(epoch, epoch_num, cost_F.item()))


# 3. High-level Implementation with nn.Module

class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,3)

    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifier()

optim_model = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    hps = model(x_train)
    cst = F.cross_entropy(hps, y_train)

    optim_model.zero_grad()
    cst.backward()
    optim_model.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cst.item()))


