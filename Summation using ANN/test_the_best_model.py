import statistics

import numpy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

operands = []
for i in range(-10, 11):
    for j in range(-10, 11):
        temp = [i, j]
        operands.append(temp)

operands = np.array(operands)
operands = operands.reshape((-1, 2))
operands = numpy.vstack((operands, operands))

sum_results = []

for operand_pair in operands:
    sum_results.append(operand_pair[0] + operand_pair[1])

sum_results = np.array(sum_results)
sum_results = sum_results.reshape((-1, 1))

train_X, temp_X, train_y, temp_y = train_test_split(operands, sum_results, train_size=0.8, shuffle=True)
dev_X, test_X, dev_y, test_y = train_test_split(temp_X, temp_y, train_size=0.5, shuffle=True)

train_dataset = TensorDataset(torch.tensor(train_X, dtype=torch.float), torch.tensor(train_y, dtype=torch.float))
dev_dataset = TensorDataset(torch.tensor(dev_X, dtype=torch.float), torch.tensor(dev_y, dtype=torch.float))
test_dataset = TensorDataset(torch.tensor(test_X, dtype=torch.float), torch.tensor(test_y, dtype=torch.float))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, drop_last=True)
dev_loader = DataLoader(dev_dataset, batch_size=len(dev_dataset.tensors[0]), shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset.tensors[0]), shuffle=False)


class SumationANN(nn.Module):
    def __init__(self, n_layers, n_nodes):
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        super().__init__()
        self.input = nn.Linear(2, n_nodes)
        self.layers = []
        for layer_i in range(n_layers):
            self.layers.append(nn.Linear(n_nodes, n_nodes))
        self.output = nn.Linear(n_nodes, 1)

    def forward(self, Xx):
        Xx = nn.functional.relu(self.input(Xx))
        for layer in self.layers:
            Xx = nn.functional.relu(layer(Xx))
        return self.output(Xx)


best_regressor = torch.load("best_model.pt")
x, y = next(iter(test_loader))
test_preds = best_regressor(x)
test_preds = torch.tensor(test_preds, dtype=torch.int)
best_test_acc = (len(torch.where(test_preds == y)[0]) / len(y))*100
print(f"Best Model has test accuracy : {best_test_acc}")

plt.plot(y.detach(), 'bo')
plt.plot(test_preds.detach(), 'gx')
plt.show()
