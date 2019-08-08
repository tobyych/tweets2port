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


class Encoder(torch.nn.Module):
    def __init__(self, embedding_dim1, embedding_dim2):
        super(Encoder, self).__init__()
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


class AttentionDecoder(torch.nn.Module):
    def __init__(self, embedding_dim1, embedding_dim2):
        super(AttentionDecoder, self).__init__()
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

