import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1) #  for reproducibility


# Data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

# Model initialization
W = torch.zeros(1)
lr = 0.1

# variable declare
epoch_num = 10

for epoch in range(epoch_num + 1):
    hypothesis = x_train * W
    cost = torch.mean((hypothesis - y_train) ** 2)
    gradient = torch.sum((W * x_train - y_train) * x_train)

    print('epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'
          .format(epoch, epoch_num, W.item(), cost.item()))

    W -= lr * gradient


# Data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

# Model initialization
W = torch.zeros(1, requires_grad=True)

# optimizer
optimizer = optim.SGD([W], lr=0.15)

# variable declare
epoch_num = 10

for epoch in range(epoch_num + 1):
    hypothesis = x_train * W  # H(x) calculation
    cost = torch.mean((hypothesis - y_train) ** 2)


    print('epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'
          .format(epoch, epoch_num, W.item(), cost.item()))

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()