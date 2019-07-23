import data as d
import word2vec as w2v
import nn
from preprocessing import stock_universe
import os
import pandas as pd
import argparse

PATH_TO_PRICES = "temp/prices/pickle"
PATH_TO_EMBEDDINGS = "temp/padded_embeddings/pickle"
PATH_TO_WORD2VEC = "temp/word2vec.model"
EXT_PICKLE = ".pickle"


def check_price_files(path_to_folder=PATH_TO_PRICES):
    for stock in stock_universe:
        if not os.path.exists(os.path.join(path_to_folder, stock + ".pickle")):
            return False
    return True


def check_embeddings_files(path_to_folder=PATH_TO_EMBEDDINGS):
    for stock in stock_universe:
        if not os.path.exists(os.path.join(path_to_folder, stock + ".pickle")):
            return False
    return True


def main(passed_args=None):
    parser = argparse.ArgumentParser(
        description="train a neural network on tweets against prices"
    )
    parser.add_argument(
        "--prep",
        "-p",
        dest="need_prep",
        action="store_true",
        default=False,
        help="toggle this option if you are running this program the first time",
    )
    args = parser.parse_args(passed_args)
    if args.need_prep:
        # prepare Word2Vec model
        if not os.path.exists(PATH_TO_WORD2VEC):
            w2v.train_word2vec()

        # prepare all data required
        prices = d.load_prices()
        w2v_model = w2v.load_word2vec()
        for stock in stock_universe:
            d.get_price_by_stock(stock, prices)
            d.load_tweets_by_stock(stock)
            w2v.get_padded_embeddings(stock, w2v_model)
    # training and evaluation
    for stock in stock_universe:
        nn.train_nn_by_stock(stock)


if __name__ == "__main__":
    main()

