import torch
import torch.nn as nn
import matplotlib.pyplot as plt


N = 100
x = torch.randn(N,1)
y = (x + torch.randn(N,1))/2


ANN_regressor = nn.Sequential(
    nn.Linear(1,1),
    nn.ReLU(),
    nn.Linear(1,1)
)

epochs = 500
l_rate = .01
lossFun = nn.MSELoss()
optimizer = torch.optim.SGD(ANN_regressor.parameters(), lr = l_rate)
y_pred=0

for epoch in range(epochs):
    y_pred = ANN_regressor(x)
    loss = lossFun(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(x, y, "bo", label="Real")
plt.plot(x, y_pred.detach(), "ro", label="predicted")
plt.legend()
plt.show()
