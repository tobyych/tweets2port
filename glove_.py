from glove import Corpus, Glove
import glove
import data as d
import pandas as pd
import numpy as np
import os, multiprocessing, warnings, logging

logging.basicConfig(``
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
warnings.filterwarnings("ignore")


def train_glove():
    if not os.path.exists("temp/tweets.pickle"):
        df = d.load_tweets()
    else:
        df = pd.read_pickle("temp/tweets.pickle")
    sentences = d.get_all_tweets(df)
    n_cpu = multiprocessing.cpu_count()
    corpus = Corpus(sentences)
    model = Glove(no_components=50, learning_rate=0.05)
    model.fit(corpus.matrix, epochs=30, no_threads=n_cpu, verbose=True)
    model.add_dictionary(corpus.dictionary)
    model.save("temp/glove.model")
    return model


# def load_glove(path_to_model="temp/glove.model"):
#     return glove.load(path_to_model)


train_glove()

