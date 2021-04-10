import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)  # for reproducibility


# Data definition
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

# Model initialization
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer setting
optimizer = optim.SGD([W, b], lr=0.01)

# variable declare
epoch_num = 1000

for epoch in range(epoch_num + 1):
    hypothesis = x_train * W + b
    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch {:4d}/{} W: {:.3f}, b: {:.3f}, Cost: {:.6f}'
              .format(epoch, epoch_num, W.item(), b.item(), cost.item()))


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])



# Model initialization
model = LinearRegressionModel()

# optimizer setting
optimizer = optim.SGD(model.parameters(), lr=0.01)

# variable declare
epoch_num = 1000

for epoch in range(epoch_num + 1):
    prediction = model(x_train)  # H(x) calculation
    cost = F.mse_loss(prediction, y_train)  # cost calculation with Mean Squared Error

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        params = list(model.parameters())
        W = params[0].item()
        b = params[1].item()
        print('epoch {:4d}/{} W: {:.3f}, b: {:.3f}, Cost: {:.6f}'
              .format(epoch, epoch_num, W, b, cost.item()))

