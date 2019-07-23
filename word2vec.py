from gensim.models import Word2Vec
import data as d
import pandas as pd
import numpy as np
import os, multiprocessing, warnings, logging

logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
warnings.filterwarnings("ignore")


def train_word2vec(use_pickle=True):
    if use_pickle:
        df = pd.read_pickle("temp/tweets.pickle")
    else:
        df = d.load_tweets()
    sentences = d.get_all_tweets(df)
    n_cpu = multiprocessing.cpu_count()
    model = Word2Vec(size=50, min_count=1, workers=n_cpu - 1)
    model.build_vocab(sentences)
    model.train(
        sentences=sentences, total_examples=model.corpus_count, epochs=model.iter
    )
    model.save("temp/word2vec.model")
    return model


def load_word2vec(path_to_model="temp/word2vec.model"):
    return Word2Vec.load(path_to_model)


def get_embeddings(tweets_df, model):
    outer_list = []
    for i in range(tweets_df.shape[0]):
        inner_list = []
        for item in tweets_df.iloc[i].to_list()[0]:
            averaged_wv = np.zeros(model.vector_size, dtype="float32")
            for word in item:
                averaged_wv += model.wv[word] / len(item)
            inner_list.append(averaged_wv)
        outer_list.append(inner_list)
    embeddings_df = pd.Series(outer_list)
    return embeddings_df


def add_paddings(embeddings_df, wv_size):
    max_len = 0
    for item in embeddings_df:
        max_len = len(item) if len(item) > max_len else max_len
    for item in embeddings_df:
        for _ in range(max_len - len(item)):
            item.append(np.zeros(wv_size, dtype="float32"))
    return embeddings_df


def get_padded_embeddings(
    stock_name,
    w2v_model,
    path_to_tweets="temp/tweets/pickle",
    to_csv=False,
    to_pickle=True,
):
    tweets = pd.read_pickle(os.path.join(path_to_tweets, stock_name + ".pickle"))
    embeddings = get_embeddings(tweets, w2v_model)
    padded_embeddings = add_paddings(embeddings, w2v_model.vector_size)
    if to_csv:
        if not os.path.exists("temp/padded_embeddings/csv"):
            os.makedirs("temp/padded_embeddings/csv")
        padded_embeddings.to_csv("temp/padded_embeddings/csv/" + stock_name + ".csv")
    if to_pickle:
        if not os.path.exists("temp/padded_embeddings/pickle"):
            os.makedirs("temp/padded_embeddings/pickle")
        padded_embeddings.to_pickle(
            "temp/padded_embeddings/pickle/" + stock_name + ".pickle"
        )
    return padded_embeddings

