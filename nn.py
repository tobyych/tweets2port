import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import os
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV

BATCH_SIZE = 16
N_EPOCHS = 10
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5


class Tweet2Returns(torch.nn.Module):
    def __init__(self):
        super(Tweet2Returns, self).__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(20 * 50, 500),
            torch.nn.Sigmoid(),
            # torch.nn.BatchNorm1d(1024),
            # torch.nn.Dropout(0.1),
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(500, 200),
            torch.nn.Sigmoid(),
            # torch.nn.BatchNorm1d(256),
            # torch.nn.Dropout(0.05),
        )
        self.fc3 = torch.nn.Linear(200, 1)

    def forward(self, x):
        x = x.view(-1, 20 * 50)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def train_nn_by_stock(stock_name):
    x = pd.read_pickle("temp/padded_embeddings/pickle/" + stock_name + ".pickle")
    y = pd.read_pickle("temp/prices/pickle/" + stock_name + ".pickle")

    x = torch.Tensor(np.stack(x.values, axis=0))
    y = torch.Tensor(np.stack(y.values, axis=0))

    torch_dataset = data.TensorDataset(x, y)

    # train, validation and test splits
    VALIDATION_SIZE = 0.2
    TEST_SIZE = 0.1
    dataset_size = len(torch_dataset)
    indices = list(range(dataset_size))
    n_validation = int(dataset_size * VALIDATION_SIZE)
    n_test = int(dataset_size * TEST_SIZE)
    n_train = dataset_size - n_test - n_validation
    train_indices, validation_indices, test_indices = (
        indices[:n_train],
        indices[n_train : (n_train + n_validation)],
        indices[(n_train + n_validation) :],
    )

    train_set = data.Subset(torch_dataset, train_indices)
    validation_set = data.Subset(torch_dataset, validation_indices)
    test_set = data.Subset(torch_dataset, test_indices)

    train_loader = data.DataLoader(
        dataset=train_set, batch_size=BATCH_SIZE, num_workers=4
    )
    validation_loader = data.DataLoader(
        dataset=validation_set, batch_size=BATCH_SIZE, num_workers=4
    )
    test_loader = data.DataLoader(
        dataset=test_set, batch_size=BATCH_SIZE, num_workers=4
    )

    model = Tweet2Returns()  # need to add parameters for the size of input layer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    training_losses = []
    validation_losses = []
    for epoch in range(N_EPOCHS):
        # training
        for batch_x, batch_y in train_loader:
            # forward step
            pred = model(batch_x)
            batch_y = batch_y.view(-1, 1)
            loss = criterion(pred, batch_y)
            # backward step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_losses.append(loss.item())

        # # evaluation
        # with torch.set_grad_enabled(False):
        #     for batch_x, batch_y in validation_loader:
        #         pred = model(batch_x)
        #         batch_y = batch_y.view(-1, 1)
        #         validation_losses.append(criterion(pred, batch_y))

        print(
            f"(epoch {epoch}) training loss: {training_losses[epoch]}"  # ", validation loss: {validation_losses[epoch]}"
        )

    print("...training has been completed")
    if not os.path.exists("temp/nn"):
        os.makedirs("temp/nn")
    torch.save(model.state_dict(), "temp/nn/" + stock_name + ".pth")

