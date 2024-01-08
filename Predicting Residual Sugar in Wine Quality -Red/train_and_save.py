import random
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

data = pd.read_csv('winequality-red.csv', sep=';')

data2 = data.copy(deep=True)

residual_sugar_replaced_i = []
residual_sugar_replaced_val = []

i = 0
while i < 100:
    random_index = random.randint(0, 1598)
    original_val = data2["residual sugar"][random_index]

    if original_val not in residual_sugar_replaced_val:
        data2["residual sugar"][random_index] = np.NaN
        residual_sugar_replaced_i.append(random_index)
        residual_sugar_replaced_val.append(original_val)
        i = i+1

idvs = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]

data = data.iloc[residual_sugar_replaced_i, :]
test_X = data.iloc[:, idvs].values
test_y = data.iloc[:, 3].values
test_y = np.reshape(test_y, (-1, 1))

test_set = TensorDataset(torch.tensor(test_X).float(), torch.tensor(test_y).float())
test_loader = DataLoader(test_set, shuffle=True, batch_size=len(test_set.tensors[0]))


data2 = data2.dropna()
X = data2.iloc[:, idvs].values
y = data2.iloc[:, 3].values
y = np.reshape(y, (-1, 1))

train_set = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
train_loader = DataLoader(train_set, batch_size=1, drop_last=True, shuffle=True)


class ResidualSugarPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(np.shape(X)[1], 30)
        self.hidden_layers = []
        for h_layer_i in range(5):
            self.hidden_layers.append(nn.Linear(30, 30))
        self.output = nn.Linear(30, 1)

    def forward(self, X):
        X = nn.functional.relu(self.input(X))
        for layer in self.hidden_layers:
            X = nn.functional.relu(layer(X))
        return self.output(X)


predictor = ResidualSugarPredictor()
optimizer = torch.optim.Adam(params=predictor.parameters(), lr=.001)
lossFun = nn.MSELoss()
print("\n\n\n\n")
for epoch_i in range(2000):
    epoch_accuracies = []
    losses = []
    for X, y in train_loader:
        preds = predictor(X)
        loss = lossFun(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = torch.mean(torch.div(torch.abs(torch.sub(y, preds)), y)).item()
        final_acc = 100.00 - (100.00 * acc)
        epoch_accuracies.append(final_acc)
        losses.append(loss.item())
    epoch_accuracy = np.mean(epoch_accuracies)
    final_loss = np.mean(losses)
    print(f"Epoch {epoch_i+1}   Train accuracy: {epoch_accuracy} %   loss: {final_loss}")

predictor.eval()
X, y = next(iter(test_loader))
test_results = predictor(X)

plt.plot(y.detach().numpy(), 'bo')
plt.plot(test_results.detach().numpy(), 'gx')
plt.show()

torch.save(predictor, "residual_sugar.pt")
