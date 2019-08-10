import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import data as d
import pandas as pd
import os, sys
import pickle


def get_vectorised_seq(path_to_tweets="temp/tweets.pickle", to_pickle=True):
    if os.path.exists(path_to_tweets):
        tweets = pd.read_pickle(path_to_tweets)
    else:
        tweets = d.load_tweets()
    list_all_tweets = d.get_all_tweets(tweets)
    list_all_tweets = [tweet for tweet in list_all_tweets if len(tweet) != 0]
    list_all_words = d.get_all_words(tweets)
    vocab = ["<pad>"] + list_all_words
    vectorised_seq = [
        [[vocab.index(word) for word in tweet] for tweet in daily_tweets]
        for daily_tweets in tweets
    ]
    if to_pickle:
        with open("temp/vectorised_seq.pickle", "wb") as f:
            pickle.dump(vectorised_seq, f)
    return vectorised_seq, vocab


def get_vectorised_seq_by_stock(
    stock,
    path_to_tweets_folder="temp/tweets/pickle/",
    path_to_vectorised_seq_folder="temp/vectorised_seq/pickle/",
    to_pickle=True,
):
    if os.path.exists(path_to_tweets_folder + stock + ".pickle"):
        tweets = pd.read_pickle(path_to_tweets_folder + stock + ".pickle")
    else:
        tweets = d.load_tweets_by_stock(stock)
    list_all_tweets = d.get_all_tweets(tweets)
    list_all_tweets = [tweet for tweet in list_all_tweets if len(tweet) != 0]
    list_all_words = d.get_all_words(tweets)
    vocab = ["<pad>"] + list_all_words
    tweets = tweets[0].tolist()
    vectorised_seq = [
        [[vocab.index(word) for word in tweet] for tweet in daily_tweets]
        for daily_tweets in tweets
    ]
    if to_pickle:
        if not os.path.exists(path_to_vectorised_seq_folder):
            os.makedirs(path_to_vectorised_seq_folder)
        with open("temp/vectorised_seq/pickle/" + stock + ".pickle", "wb") as f:
            pickle.dump(vectorised_seq, f)
    return vectorised_seq, vocab


class EncoderRNN(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=False
    ):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.lstm = torch.nn.LSTM(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

    def forward(self, seq_tensor, seq_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(seq_tensor)
        packed = pack_padded_sequence(embedded, seq_lengths, batch_first=True)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        if self.bidirectional:
            outputs = (
                outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]
            )  # Sum bidirectional outputs
        return outputs, hidden


class FeedForward(torch.nn.Module):
    def __init__(self, input_size):
        super(FeedForward, self).__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(input_size, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.sequential(x)


def get_seq_input(vectorised_seq):
    seq_lengths = torch.LongTensor(list(map(len, vectorised_seq)))
    seq_tensor = torch.zeros((len(vectorised_seq), seq_lengths.max())).long()
    for idx, (seq, seqlen) in enumerate(zip(vectorised_seq, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    # sorting
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    return seq_tensor, seq_lengths, perm_idx


def get_seq_input_one_sample(vectorised_seq):
    seq_lengths = torch.LongTensor(list(map(len, vectorised_seq)))
    seq_tensor = torch.zeros((len(vectorised_seq), seq_lengths.max())).long()
    for idx, (seq, seqlen) in enumerate(zip(vectorised_seq, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    # sorting
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    return seq_tensor, seq_lengths, perm_idx


def train_rnn(vectorised_seq, prices, input_size, hyperparam):
    encoder = EncoderRNN(input_size=input_size, hidden_size=hyperparam["HIDDEN_SIZE"])
    feedforward = FeedForward(input_size=hyperparam["HIDDEN_SIZE"])
    criterion = torch.nn.MSELoss()
    optimizer_encoder = torch.optim.Adam(
        encoder.parameters(), lr=hyperparam["LEARNING_RATE"]
    )
    optimizer_ff = torch.optim.Adadelta(
        encoder.parameters(), lr=hyperparam["LEARNING_RATE"]
    )
    training_losses = []
    for epoch in range(hyperparam["N_EPOCHS"]):
        running_training_loss = 0
        for seq, price in zip(vectorised_seq, prices):
            # if no tweet on a day, then directly pass a zero tensor to feedforward
            if len(seq[0]) == 0:
                # pass zero tensor to feedforward
                avg_tweet_rep = torch.zeros(hyperparam["HIDDEN_SIZE"])
            else:
                list_daily_rep = []
                for s in seq:
                    seq_tensor, seq_lengths, _ = get_seq_input([s])
                    optimizer_encoder.zero_grad()
                    _, (ht, _) = encoder(seq_tensor, seq_lengths)
                    list_daily_rep.append(ht[-1])
                avg_tweet_rep = torch.mean(
                    torch.stack(list_daily_rep), dim=1, keepdim=True
                )
            pred = feedforward(avg_tweet_rep)
            loss = criterion(pred, price)
            loss.backward()
            torch.nn.utils.clip_grad_value_(
                encoder.parameters(), hyperparam["CLIPPING_THRESHOLD"]
            )
            torch.nn.utils.clip_grad_value_(
                feedforward.parameters(), hyperparam["CLIPPING_THRESHOLD"]
            )
            optimizer_encoder.step()
            optimizer_ff.step()
            running_training_loss += loss.item()
        training_losses.append(running_training_loss)
        print(f"(epoch {epoch}) training loss: {training_losses[epoch]}")
    print("...training has been completed")
    return encoder, feedforward, training_losses


def a():
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=hyperparam["BATCH_SIZE"],
        drop_last=hyperparam["DROP_LAST"],
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=hyperparam["BATCH_SIZE"],
        drop_last=hyperparam["DROP_LAST"],
    )

    encoder = EncoderRNN(input_size=input_size, hidden_size=hyperparam["HIDDEN_SIZE"])
    feedforward = FeedForward(input_size=hyperparam["HIDDEN_SIZE"])
    criterion = torch.nn.MSELoss()
    optimizer_encoder = torch.optim.Adam(
        encoder.parameters(), lr=hyperparam["LEARNING_RATE"]
    )
    optimizer_ff = torch.optim.Adadelta(
        feedforward.parameters(), lr=hyperparam["LEARNING_RATE"]
    )

    training_losses = []
    validation_losses = []
    for epoch in range(hyperparam["N_EPOCHS"]):
        # training
        running_train_loss = 0
        running_valid_loss = 0
        for batch_seq_tensor, batch_seq_lengths, batch_y in train_loader:
            optimizer_encoder.zero_grad()
            optimizer_ff.zero_grad()
            # forward step
            _, (ht, _) = encoder(batch_seq_tensor, batch_seq_lengths)
            pred = feedforward(ht[-1])
            batch_y = batch_y.view(-1, 1)
            loss_ff = criterion(pred, batch_y)
            # backward step
            loss_ff.backward()
            torch.nn.utils.clip_grad_value_(
                encoder.parameters(), hyperparam["CLIPPING_THRESHOLD"]
            )
            torch.nn.utils.clip_grad_value_(
                feedforward.parameters(), hyperparam["CLIPPING_THRESHOLD"]
            )
            optimizer_encoder.step()
            optimizer_ff.step()
            running_train_loss += loss_ff.item()
        training_losses.append(running_train_loss)

        # evaluation
        with torch.set_grad_enabled(False):
            for batch_seq_tensor, batch_seq_lengths, batch_y in test_loader:
                _, (ht, _) = encoder(batch_seq_tensor, batch_seq_lengths)
                pred = feedforward(ht[-1])
                batch_y = batch_y.view(-1, 1)
                loss_ff = criterion_ff(pred, batch_y)
                running_valid_loss += loss_ff.item()
            validation_losses.append(running_valid_loss)

        print(
            f"(epoch {epoch}) training loss: {training_losses[epoch]}, validation loss: {validation_losses[epoch]}"
        )
    print("...training has been completed")
    return encoder, feedforward, training_losses, validation_losses


# # REMEMBER: Your outputs are sorted. If you want the original ordering
# # back (to compare to some gt labels) unsort them
# _, unperm_idx = perm_idx.sort(0)
# output = output[unperm_idx]
# print(output)

