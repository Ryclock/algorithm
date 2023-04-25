import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

import torch
from torch.utils import data as Data
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.tanh(z1)
        z2 = self.fc2(a1)
        a2 = self.tanh(z2)
        z3 = self.fc3(a2)
        a3 = self.softmax(z3)
        return a3


if __name__ == "__main__":
    import os
    from pathlib import Path

    parent_path = Path(__file__).parents[1]
    paths = ['data_sets', 'MNIST']
    train_data = pd.read_csv(os.path.join(parent_path, *paths, 'train.csv'))
    test_data = pd.read_csv(os.path.join(parent_path, *paths, 'test.csv'))
    X_train, X_valid, y_train, y_valid = map(np.array, train_test_split(
        train_data.drop(columns='label'), train_data['label'], test_size=0.2))
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    onehot = LabelBinarizer()
    Y_train = onehot.fit_transform(y_train)

    batch_size = 1
    train_data = Data.TensorDataset(torch.from_numpy(X_train).to(torch.float32),
                                    torch.from_numpy(Y_train).to(torch.float32))
    valid_data = Data.TensorDataset(torch.from_numpy(X_valid).to(torch.float32),
                                    torch.from_numpy(y_valid).to(torch.float32))
    train_loader = Data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = Data.DataLoader(
        dataset=valid_data, batch_size=batch_size, shuffle=True)

    n_epochs = 10
    learning_rate = 0.001
    model = Net(input_size=784, hidden_size1=256,
                hidden_size2=64, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    for epoch in range(n_epochs):
        for i, (x, y) in enumerate(train_loader):
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch +
                                                                         1, n_epochs, i+1, total_step, loss.item()))

    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in valid_loader:
            outputs = model(x)
            _, y_pred = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (y_pred == y).sum().item()

        print('Test Accuracy of the model on the {} valid images: {} %'.format(
            total, 100 * correct / total))
