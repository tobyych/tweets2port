import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import os, sys

BATCH_SIZE = 16
N_EPOCHS = 5
LEARNING_RATE = 1e-8
WEIGHT_DECAY = 1e-8
CLIPPING_THRESHOLD = 1
INIT_BIAS = 1e-8


class Tweet2Returns(torch.nn.Module):
    def __init__(self, embedding_dim1, embedding_dim2):
        super(Tweet2Returns, self).__init__()
        self.embedding_dim1 = embedding_dim1
        self.embedding_dim2 = embedding_dim2
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim1 * embedding_dim2, 100),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(1024),
            # torch.nn.Dropout(0.1),
        )
        self.fc1.apply(init_weights)
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.Sigmoid(),
            # torch.nn.BatchNorm1d(256),
            # torch.nn.Dropout(0.05),
        )
        self.fc2.apply(init_weights)
        self.fc3 = torch.nn.Sequential(torch.nn.Linear(50, 1))
        self.fc3.apply(init_weights)

    def forward(self, x):
        x = x.view(-1, self.embedding_dim1 * self.embedding_dim2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(INIT_BIAS)


def normalise(x):
    return (x - x.mean()) / x.std()


def train_nn_by_stock(stock_name):
    x = pd.read_pickle("temp/padded_embeddings/pickle/" + stock_name + ".pickle")
    y = pd.read_pickle("temp/returns/pickle/" + stock_name + ".pickle")

    x = torch.Tensor(np.stack(x.values, axis=0))
    y = normalise(torch.Tensor(np.stack(y.values, axis=0)))

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
        dataset=train_set, batch_size=BATCH_SIZE, drop_last=True
    )
    validation_loader = data.DataLoader(
        dataset=validation_set, batch_size=BATCH_SIZE, drop_last=True
    )
    test_loader = data.DataLoader(
        dataset=test_set, batch_size=BATCH_SIZE, drop_last=True
    )

    model = Tweet2Returns(
        x.shape[1], x.shape[2]
    )  # need to add parameters for the size of input layer
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    training_losses = []
    validation_losses = []
    for epoch in range(N_EPOCHS):
        # training
        running_train_loss = 0
        running_valid_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            # forward step
            pred = model(batch_x)
            # if torch.isnan(pred).all():
            #     # print(f"batch_x: {batch_x}")
            #     # print([x for x in model.parameters()])
            #     sys.exit(1)
            batch_y = batch_y.view(-1, 1)
            loss = criterion(pred, batch_y)
            # backward step
            # loss.register_hook(lambda grad: print(grad))
            loss.backward()
            # print(
            #     model.fc1[0].weight.grad,
            #     model.fc2[0].weight.grad,
            #     model.fc3[0].weight.grad,
            # )
            torch.nn.utils.clip_grad_value_(model.parameters(), CLIPPING_THRESHOLD)
            optimizer.step()
            running_train_loss += loss.item()
        training_losses.append(running_train_loss)

        # evaluation
        with torch.set_grad_enabled(False):
            for batch_x, batch_y in validation_loader:
                pred = model(batch_x)
                batch_y = batch_y.view(-1, 1)
                valid_loss = criterion(pred, batch_y)
                running_valid_loss += valid_loss.item()
            validation_losses.append(running_valid_loss)

        print(
            f"(epoch {epoch}) training loss: {training_losses[epoch]}, validation loss: {validation_losses[epoch]}"
        )

    print("...training has been completed")
    if not os.path.exists("temp/nn"):
        os.makedirs("temp/nn")
    torch.save(model.state_dict(), "temp/nn/" + stock_name + ".pth")

