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
        x = self.sequential(x)
        return x


def get_seq_input(vectorised_seq, device):
    seq_lengths = torch.LongTensor(list(map(len, vectorised_seq))).to(device)
    seq_tensor = torch.zeros((len(vectorised_seq), seq_lengths.max())).long().to(device)
    for idx, (seq, seqlen) in enumerate(zip(vectorised_seq, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq).to(device)
    # sorting
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    return seq_tensor, seq_lengths, perm_idx


def train_rnn(vectorised_seq, returns, input_size, hyperparam):
    valid_idx = len(vectorised_seq) - int(len(vectorised_seq) * hyperparam['VALIDATION_SIZE'])
    train_vectorised_seq, test_vectorised_seq = vectorised_seq[:valid_idx], vectorised_seq[valid_idx:] 
    train_returns, test_returns = returns[:valid_idx], returns[valid_idx:] 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = EncoderRNN(input_size=input_size, hidden_size=hyperparam["HIDDEN_SIZE"]).to(device)
    feedforward = FeedForward(input_size=hyperparam["HIDDEN_SIZE"]).to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer_encoder = torch.optim.Adam(
        encoder.parameters(), lr=hyperparam["LEARNING_RATE"]
    )
    optimizer_ff = torch.optim.Adadelta(
        encoder.parameters(), lr=hyperparam["LEARNING_RATE"]
    )
    training_losses = []
    for epoch in range(hyperparam["N_EPOCHS"]):
        running_training_loss = 0
        for seq, ret in zip(train_vectorised_seq, train_returns):
            # if no tweet on a day, then directly pass a zero tensor to feedforward
            if len(seq[0]) == 0:
                avg_tweet_rep = torch.zeros(hyperparam["HIDDEN_SIZE"]).to(device)
            else:
                list_daily_rep = []
                for s in seq:
                    seq_tensor, seq_lengths, _ = get_seq_input([s], device=device)
                    optimizer_encoder.zero_grad()
                    _, (ht, _) = encoder(seq_tensor, seq_lengths)
                    list_daily_rep.append(ht[-1])
                avg_tweet_rep = torch.mean(
                    torch.stack(list_daily_rep), dim=0
                ).to(device)
            optimizer_ff.zero_grad()
            pred = feedforward(avg_tweet_rep.view(-1, ))
            loss = criterion(pred, ret)
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
    # eval mode
    with torch.set_grad_enabled(False):
        results = []
        for seq, ret in zip(test_vectorised_seq, test_returns):
            # if no tweet on a day, then directly pass a zero tensor to feedforward
            if len(seq[0]) == 0:
                avg_tweet_rep = torch.zeros(hyperparam["HIDDEN_SIZE"]).to(device)
            else:
                list_daily_rep = []
                for s in seq:
                    seq_tensor, seq_lengths, _ = get_seq_input([s], device=device)
                    _, (ht, _) = encoder(seq_tensor, seq_lengths)
                    list_daily_rep.append(ht[-1])
                avg_tweet_rep = torch.mean(
                    torch.stack(list_daily_rep), dim=0
                ).to(device)
            pred = feedforward(avg_tweet_rep.view(-1,))
            loss = criterion(pred, ret)
        results.append((ret.item(), pred.item(), loss.item()))
    print("...training has been completed")
    return encoder, feedforward, training_losses, results

# # REMEMBER: Your outputs are sorted. If you want the original ordering
# # back (to compare to some gt labels) unsort them
# _, unperm_idx = perm_idx.sort(0)
# output = output[unperm_idx]
# print(output)

