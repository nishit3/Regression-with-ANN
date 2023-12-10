import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy

data = pd.read_csv('winequality-red.csv', sep=';')
data = data[data['total sulfur dioxide'] < 200]              # Removing Outliers
data = data.apply(scipy.stats.zscore)

feature_matrix = data.iloc[:, [data.columns.get_loc(str(column_name)) for column_name in data.keys().drop('residual sugar')]].values
target = data.iloc[:, data.columns.get_loc('residual sugar')].values
target = np.reshape(target, (-1, 1))

train_X, test_X, train_y, test_y = train_test_split(feature_matrix, target, train_size=0.8)
train_X = torch.tensor(train_X).float()
test_X = torch.tensor(test_X).float()
train_y = torch.tensor(train_y).float()
test_y = torch.tensor(test_y).float()


train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)
epochs = np.linspace(1, 1000, num=1000)
results = []
train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=len(torch.detach(test_X)))


classifier = nn.Sequential(
        nn.Linear(len(data.keys().drop('residual sugar')), 16),
        nn.ReLU(),
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        # nn.Sigmoid()
    )

optimizer = torch.optim.SGD(classifier.parameters(), lr=.01)
loss_func = nn.MSELoss()

for epoch_i, epoch in enumerate(epochs):
    for X, y in train_loader:
        predictions = classifier(X)
        loss = loss_func(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_X, test_y = next(iter(test_loader))
    pred = classifier(test_X)
    test_loss = loss_func(pred, test_y)
    results.append(torch.tensor(test_loss).detach())

test_X, test_y = next(iter(test_loader))
pred = classifier(test_X)
test_corr = scipy.stats.pearsonr(np.reshape(test_y, (-1)), np.reshape(pred.detach().numpy(), (-1))).statistic

plt.title("Test Result")
plt.plot(test_y, 'bo', label='Real')
plt.plot(pred.detach().numpy(), 'rs', label='predicted')
plt.xlabel("Correlation Coefficient = "+str(test_corr))
plt.ylabel("Residual Sugar Value")
plt.legend()
plt.show()

plt.plot(epochs, results)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
