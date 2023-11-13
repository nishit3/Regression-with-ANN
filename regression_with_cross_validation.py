import sys
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

N = 1000
x = torch.randn(N, 1)
y = (x + torch.randn(N, 1)) / 2

train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=.8)

training_data = TensorDataset(train_x, train_y)
testing_data = TensorDataset(test_x, test_y)

training_data_loader = DataLoader(training_data, batch_size=5, shuffle=True)
testing_data_loader = DataLoader(testing_data, batch_size=len(testing_data.tensors[0]))

ANN_regressor = nn.Sequential(
    nn.Linear(1, 1),
    nn.ReLU(),
    nn.Linear(1, 1)
)

epochs = 2000
l_rate = .01
lossFun = nn.MSELoss()
optimizer = torch.optim.SGD(ANN_regressor.parameters(), lr=l_rate)

for epoch in range(epochs):
    for batch_x, batch_y in training_data_loader:
        prediction = ANN_regressor(batch_x)
        loss = lossFun(prediction, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

prediction_on_train_set = ANN_regressor(training_data.tensors[0])
loss_on_train_set = lossFun(prediction_on_train_set, training_data.tensors[1])
training_accuracy = (1.00 - loss_on_train_set.item())*100
sys.stdout.write("\nTraining Accuracy = ")
sys.stdout.write(str(training_accuracy))
sys.stdout.write("%\n")

prediction_on_test_set = ANN_regressor(testing_data.tensors[0])
loss_on_test_set = lossFun(prediction_on_test_set, testing_data.tensors[1])
testing_accuracy = (1.00 - loss_on_test_set.item())*100
sys.stdout.write("Testing Accuracy = ")
sys.stdout.write(str(testing_accuracy))
sys.stdout.write("%\n")
