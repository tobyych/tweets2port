from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
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


def train_word2vec():
    if not os.path.exists("temp/tweets.pickle"):
        df = d.load_tweets()
    else:
        df = pd.read_pickle("temp/tweets.pickle")
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


def padding(array, target_len, wv_size):
    if target_len > len(array):
        for _ in range(target_len - len(array)):
            array.append(np.zeros(wv_size, dtype="float32"))
        return array
    else:
        for _ in range(target_len - len(array)):
            array.pop()
        return array


def add_paddings(embeddings_df, wv_size, use_max=True):
    if use_max:
        padded_len = np.max(embeddings_df.apply(lambda x: len(x)))
    else:
        padded_len = int(np.mean(embeddings_df.apply(lambda x: len(x))))
    padded_list = []
    for item in embeddings_df.values:
        if len(item) < padded_len:
            for _ in range(padded_len - len(item)):
                item.append(np.zeros(wv_size, dtype="float32"))
            padded_list.append(item)
        else:
            padded_list.append(item[:padded_len])
    new_embeddings_df = pd.Series(padded_list)
    assert embeddings_df.shape == new_embeddings_df.shape
    return new_embeddings_df


def add_randomness(padded_embeddings, wv_size):
    outer_list = []
    for item in padded_embeddings.values:
        inner_list = []
        for wv in item:
            if all(elem == 0 for elem in wv):
                inner_list.append(np.random.rand(wv_size))
            else:
                inner_list.append(wv)
        outer_list.append(inner_list)
    new_padded_embeddings = pd.Series(outer_list)
    assert padded_embeddings.shape == new_padded_embeddings.shape
    return new_padded_embeddings


def get_padded_embeddings(
    stock_name,
    w2v_model,
    path_to_tweets="temp/tweets/pickle",
    path_to_output="temp/padded_embeddings",
    to_csv=False,
    to_pickle=True,
):
    tweets = pd.read_pickle(os.path.join(path_to_tweets, stock_name + ".pickle"))
    embeddings = get_embeddings(tweets, w2v_model)
    padded_embeddings = add_paddings(embeddings, w2v_model.vector_size, use_max=True)
    # padded_embeddings = add_randomness(padded_embeddings, w2v_model.vector_size)
    if to_csv:
        if not os.path.exists(os.path.join(path_to_output, "csv")):
            os.makedirs(os.path.join(path_to_output, "csv"))
        padded_embeddings.to_csv(
            os.path.join(path_to_output, "csv", stock_name + ".csv")
        )
    if to_pickle:
        if not os.path.exists(os.path.join(path_to_output, "pickle")):
            os.makedirs(os.path.join(path_to_output, "pickle"))
        padded_embeddings.to_pickle(
            os.path.join(path_to_output, "pickle", stock_name + ".pickle")
        )
    return padded_embeddings


def load_glove_model(
    path_to_glove="./temp/glove_wv.txt", path_to_output="./temp/glove_wv_w2vformat.txt"
):
    glove2word2vec(path_to_glove, path_to_output)
    model = KeyedVectors.load_word2vec_format(path_to_output)
    return model
