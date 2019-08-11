import data as d
import word2vec as w2v
import nn
from preprocessing import stock_universe
import os, sys
import pandas as pd
import numpy as np
import argparse
import operator
from utils import get_hyperparam_list, NN_HYPERPARAM_DICT, RNN_HYPERPARAM_DICT
import pickle
import torch
import rnn_seq as rnn

PATH_TO_PRICES = "temp/prices/pickle"
PATH_TO_EMBEDDINGS = "temp/padded_embeddings/pickle"
PATH_TO_WORD2VEC = "temp/word2vec.model"
EXT_PICKLE = ".pickle"


def main(passed_args=None):
    parser = argparse.ArgumentParser(
        description="train a neural network on tweets against prices"
    )
    parser.add_argument(
        "--word2vec",
        "-w",
        dest="word2vec",
        action="store_true",
        default=False,
        help="toggle this option if you are obtaining dataset using word2vec",
    )
    parser.add_argument(
        "--tune",
        "-t",
        dest="tuning",
        action="store_true",
        default=False,
        help="toogle this option if you are tuning hyperparameters",
    )
    parser.add_argument(
        "--rnn",
        "-r",
        dest="train_rnn",
        action="store_true",
        default=False,
        help="toogle this option to train rnn",
    )
    parser.add_argument(
        "--predict",
        "-d",
        dest="predict",
        action="store_true",
        default=False,
        help="toogle this option if you are making predictions",
    )
    parser.add_argument(
        "--markowtiz",
        "-m",
        dest="markowitz",
        action="store_true",
        default=False,
        help="toogle this option if you are doing Markowitz portfolio optimisation",
    )
    parser.add_argument(
        "--glove",
        "-g",
        dest="glove",
        action="store_true",
        default=False,
        help="toogle this option if you are obtaining dataset using glove",
    )
    args = parser.parse_args(passed_args)
    if args.word2vec:
        # prepare Word2Vec model
        if not os.path.exists(PATH_TO_WORD2VEC):
            w2v.train_word2vec()

        # prepare all data required
        prices = d.load_prices()
        w2v_model = w2v.load_word2vec()
        for stock in stock_universe:
            d.get_return_by_stock(stock, prices)
            d.load_tweets_by_stock(stock)
            w2v.get_padded_embeddings(stock, w2v_model)
        sys.exit()

    if args.glove:
        # prepare all data required
        prices = d.load_prices()
        w2v_model = w2v.load_glove_model(path_to_glove="./temp/glove_wv.txt")
        for stock in stock_universe:
            d.get_return_by_stock(stock, prices)
            d.load_tweets_by_stock(stock)
            w2v.get_padded_embeddings(
                stock, w2v_model, path_to_output="./temp/padded_embeddings/glove"
            )

    if args.tuning:
        hyperparam_list = get_hyperparam_list(NN_HYPERPARAM_DICT)
        best_hyperparam_list = []
        for stock in stock_universe:
            print(stock)
            x = pd.read_pickle("temp/padded_embeddings/pickle/" + stock + ".pickle")
            y = pd.read_pickle("temp/returns/pickle/" + stock + ".pickle")
            torch_dataset = nn.get_tensor_dataset(x, y)
            for hyperparam in hyperparam_list:
                train_set, _ = nn.train_test_split(
                    torch_dataset, hyperparam["TEST_SIZE"]
                )
                train_set, validation_set = nn.train_test_split(
                    train_set, hyperparam["VALIDATION_SIZE"]
                )
                tuning_list = []
                _, _, validation_losses = nn.train_nn(
                    train_set, validation_set, hyperparam
                )
                tuning_list.append((hyperparam, validation_losses[-1]))
            tuning_list.sort(key=operator.itemgetter(1))
            best_hyperparam = tuning_list[0][0]
            best_hyperparam_list.append((stock, best_hyperparam))
        with open("best-hyperparam.txt", "wb") as f:
            pickle.dump(best_hyperparam_list, f)
        print(best_hyperparam_list)
        sys.exit()

    if args.predict:
        if os.path.exists("./best-hyperparam.txt"):
            with open("best-hyperparam.txt", "rb") as f:
                best_hyperparam_list = pickle.load(f)
                best_hyperparam_dict = dict(best_hyperparam_list)
        for stock in stock_universe:
            hyperparam = best_hyperparam_dict[stock]
            x = pd.read_pickle("temp/padded_embeddings/pickle/" + stock + ".pickle")
            y = pd.read_pickle("temp/returns/pickle/" + stock + ".pickle")
            torch_dataset = nn.get_tensor_dataset(x, y)
            _, test_set = nn.train_test_split(torch_dataset, hyperparam["TEST_SIZE"])
            results = nn.predict_nn(test_set, "temp/nn/" + stock + ".pth")
            results_df = pd.DataFrame(results)
            results_df.columns = ["y", "pred", "loss"]
            if not os.path.exists("./output/"):
                os.makedirs("./output/")
            results_df.to_csv("./output/" + stock + ".csv")
        sys.exit()

    if args.train_rnn:
        hyperparam_list = get_hyperparam_list(RNN_HYPERPARAM_DICT)
        for hyperparam in hyperparam_list:
            for stock in stock_universe:
                print(stock)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                returns = pd.read_pickle("temp/returns/pickle/" + stock + ".pickle")
                returns = nn.normalise(torch.tensor(np.stack(returns.values, axis=0), device=device))
                vectorised_seq, vocab = rnn.get_vectorised_seq_by_stock(stock)
                input_size = len(vocab)

                encoder, feedforward, _, results = rnn.train_rnn(vectorised_seq, returns, input_size, hyperparam)
                if not os.path.exists("temp/rnn"):
                    os.makedirs("temp/rnn/encoder")
                    os.makedirs("temp/rnn/feedforward")
                torch.save(encoder.state_dict(), "temp/rnn/encoder/" + stock + ".pth")
                torch.save(feedforward.state_dict(), "temp/rnn/feedforward/" + stock + ".pth")
                results_df = pd.DataFrame(results)
                results_df.columns = ["returns", "pred", "loss"]
                if not os.path.exists("./output/rnn"):
                    os.makedirs("./output/rnn")
                results_df.to_csv("./output/rnn/" + stock + ".csv")
        sys.exit()

    if args.markowitz:
        pass

    if os.path.exists("./best-hyperparam.txt"):
        with open("best-hyperparam.txt", "rb") as f:
            best_hyperparam_list = pickle.load(f)
            best_hyperparam_dict = dict(best_hyperparam_list)

    for stock in stock_universe:
        print(stock)
        hyperparam = best_hyperparam_dict[stock]
        x = pd.read_pickle("temp/padded_embeddings/pickle/" + stock + ".pickle")
        y = pd.read_pickle("temp/returns/pickle/" + stock + ".pickle")
        torch_dataset = nn.get_tensor_dataset(x, y)
        train_set, test_set = nn.train_test_split(
            torch_dataset, hyperparam["TEST_SIZE"]
        )
        model, _, _ = nn.train_nn(train_set, test_set, hyperparam)
        if not os.path.exists("temp/nn"):
            os.makedirs("temp/nn")
        torch.save(model.state_dict(), "temp/nn/" + stock + ".pth")
    sys.exit()


if __name__ == "__main__":
    main()
