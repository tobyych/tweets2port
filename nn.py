import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import os, sys

HYPERPARAM = {
    "BATCH_SIZE": 16,
    "N_EPOCHS": 100,
    "LEARNING_RATE": 1e-5,
    "WEIGHT_DECAY": 1e-6,
    "CLIPPING_THRESHOLD": 5,
    "VALIDATION_SIZE": 0.2,
    "TEST_SIZE": 0.1,
    "DROP_LAST": True
    # "INIT_BIAS": 1e-5,
}


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
        # m.bias.data.fill_(hyperparam.INIT_BIAS)


def normalise(x):
    return (x - x.mean()) / x.std()


def get_tensor_dataset(x, y):
    """
    input: pandas dataframe
    output: tensor datasets (training, validation, testing)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(np.stack(x.values, axis=0), device=device)
    y = normalise(torch.tensor(np.stack(y.values, axis=0), device=device))
    torch_dataset = data.TensorDataset(x, y)
    return torch_dataset


def train_test_split(torch_dataset, test_size):
    dataset_size = len(torch_dataset)
    indices = list(range(dataset_size))
    n_test = int(dataset_size * test_size)
    n_train = dataset_size - n_test
    train_indices, test_indices = (indices[:n_train], indices[n_train:])

    train_set = data.Subset(torch_dataset, train_indices)
    test_set = data.Subset(torch_dataset, test_indices)

    return train_set, test_set


def train_nn(train_set, test_set, hyperparam=HYPERPARAM):
    train_loader = data.DataLoader(
        dataset=train_set,
        batch_size=hyperparam["BATCH_SIZE"],
        drop_last=hyperparam["DROP_LAST"],
    )
    test_loader = data.DataLoader(
        dataset=test_set,
        batch_size=hyperparam["BATCH_SIZE"],
        drop_last=hyperparam["DROP_LAST"],
    )
    tensor_x, _ = next(iter(train_loader))
    embedding_dim1 = tensor_x.shape[1]
    embedding_dim2 = tensor_x.shape[2]

    model = Tweet2Returns(embedding_dim1, embedding_dim2)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=hyperparam["LEARNING_RATE"])

    training_losses = []
    validation_losses = []
    for epoch in range(hyperparam["N_EPOCHS"]):
        # training
        running_train_loss = 0
        running_valid_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            # forward step
            pred = model(batch_x)
            batch_y = batch_y.view(-1, 1)
            loss = criterion(pred, batch_y)
            # backward step
            loss.backward()
            torch.nn.utils.clip_grad_value_(
                model.parameters(), hyperparam["CLIPPING_THRESHOLD"]
            )
            optimizer.step()
            running_train_loss += loss.item()
        training_losses.append(running_train_loss)

        # evaluation
        with torch.set_grad_enabled(False):
            for batch_x, batch_y in test_loader:
                pred = model(batch_x)
                batch_y = batch_y.view(-1, 1)
                valid_loss = criterion(pred, batch_y)
                running_valid_loss += valid_loss.item()
            validation_losses.append(running_valid_loss)

        print(
            f"(epoch {epoch}) training loss: {training_losses[epoch]}, validation loss: {validation_losses[epoch]}"
        )
    print("...training has been completed")
    return model, training_losses, validation_losses


def predict_nn(test_set, path_to_model):
    test_loader = data.DataLoader(dataset=test_set)
    tensor_x, _ = next(iter(test_loader))
    embedding_dim1 = tensor_x.shape[1]
    embedding_dim2 = tensor_x.shape[2]
    model = Tweet2Returns(embedding_dim1, embedding_dim2)
    model.load_state_dict(torch.load(path_to_model))
    model.eval()
    criterion = torch.nn.MSELoss()
    results = []
    with torch.set_grad_enabled(False):
        for x, y in test_loader:
            pred = model(x)
            y = y.view(-1, 1)
            loss = criterion(pred, y)
            results.append((y.item(), pred.item(), loss.item()))
    return results

