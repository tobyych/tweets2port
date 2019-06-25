from gensim.models import Word2Vec
import multiprocessing
import data
import pandas as pd
import warnings
import logging

logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
warnings.filterwarnings("ignore")


def train_word2vec():
    df = data.load_data()
    sentences = data.get_sentences(df)
    n_cpu = multiprocessing.cpu_count()
    model = Word2Vec(size=50, workers=n_cpu - 1)
    model.build_vocab(sentences)
    model.train(
        sentences=sentences, total_examples=model.corpus_count, epochs=model.iter
    )
    model.save("temp/word2vec.model")

