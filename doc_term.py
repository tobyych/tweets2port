from sklearn.feature_extraction.text import CountVectorizer
import data as d
import pandas as pd
import numpy as np
from preprocessing import stock_universe
import os
import torch
from nn import train_test_split


class Tweet2Returns(torch.nn.Module):
    def __init__(self, vocab_size):
        super(Tweet2Returns, self).__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(vocab_size, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1),
        )

    def forward(self, x):
        return self.sequential(x)


def train(stock):
    tweets = pd.read_pickle(os.path.join("temp/tweets/pickle", stock + ".pickle"))
    temp_tweets = [
        " ".join([item for sublist in x for item in sublist]) for x in tweets.iloc[:, 0]
    ]
    temp_returns = d.get_return_by_stock(stock, prices)
    cnt_vec = CountVectorizer()
    dtm = cnt_vec.fit_transform(temp_tweets)
    dtm = dtm.toarray()
    dtm = torch.tensor(np.stack(dtm, axis=0), dtype=torch.float, device=device)
    returns = torch.tensor(np.stack(temp_returns.values, axis=0), device=device)
    # print(dtm.shape, returns.shape)
    dataset = torch.utils.data.TensorDataset(dtm, returns)
    train_set, test_set = train_test_split(dataset, test_size=0.1)
    train_loader = torch.utils.data.DataLoader(dataset=train_set)
    test_loader = torch.utils.data.DataLoader(dataset=test_set)
    nn = Tweet2Returns(dtm.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(nn.parameters(), lr=1e-5)

    training_losses = []
    validation_losses = []
    for epoch in range(100):
        # training
        running_train_loss = 0
        running_valid_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            # forward step
            pred = nn(batch_x)
            batch_y = batch_y.view(-1, 1)
            loss = criterion(pred, batch_y)
            # backward step
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        training_losses.append(running_train_loss)

    # evaluation
    results = []
    with torch.set_grad_enabled(False):
        for batch_x, batch_y in test_loader:
            pred = nn(batch_x)
            batch_y = batch_y.view(-1, 1)
            valid_loss = criterion(pred, batch_y)
            running_valid_loss += valid_loss.item()
            results.append((batch_y, pred, valid_loss))
        validation_losses.append(running_valid_loss)
        print(
            f"(epoch {epoch}) training loss: {training_losses[epoch]}, validation loss: {validation_losses[epoch]}"
        )
    print("...training has been completed")
    return nn, training_losses, validation_losses, results


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prices = d.load_prices()

for stock in stock_universe:
    _, _, _, results = train(stock)
    results_df = pd.DataFrame(results)
    results_df.columns = ["y", "pred", "loss"]
    if not os.path.exists("./output/dtm"):
        os.makedirs("./output/dtm")
    results_df.to_csv("./output/dtm/" + stock + ".csv")
