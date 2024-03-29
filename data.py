import os, json, csv, pickle
import pandas as pd
import numpy as np
from preprocessing import stock_universe
import yfinance as yf
from datetime import datetime


def load_tweets(
    datapath="../data/stocknet-dataset/tweet/new_processed",
    to_csv=False,
    to_pickle=True,
):
    temp = []
    folderpath = [x[0] for x in os.walk(datapath)][1:]
    stock_list = [os.path.basename(os.path.normpath(x)) for x in folderpath]
    for folder in folderpath:
        print(f"current folder: {folder}")
        filelist = os.listdir(folder)
        temp_data = list()
        for filename in filelist:
            with open(os.path.join(folder, filename), "r") as f:
                line = f.readline()
            line = json.loads(line)
            line = list(line.values())
            temp_data.append(line)
        temp.append(temp_data)
    df = pd.DataFrame(temp)
    df.index = stock_list
    df.columns = pd.date_range(start="1/1/2014", end="3/31/2016")
    df = df.T.asfreq(pd.tseries.offsets.BDay()).T
    if to_csv:
        df.to_csv("temp/tweets.csv")
    if to_pickle:
        df.to_pickle("temp/tweets.pickle")
    return df


def load_tweets_by_stock(
    stock_name,
    path_to_data="../data/stocknet-dataset/tweet/new_processed",
    to_csv=False,
    to_pickle=True,
):
    folder_path = os.path.join(path_to_data, stock_name)
    file_list = os.listdir(folder_path)
    all_tweets = list()
    for file_name in file_list:
        with open(os.path.join(folder_path, file_name), "r") as f:
            daily_tweets = f.readline()
        daily_tweets = json.loads(daily_tweets)  # read json string
        daily_tweets = list(daily_tweets.values())
        all_tweets.append([daily_tweets])
    tweets_df = pd.DataFrame(all_tweets)
    tweets_df.index = pd.date_range(start="1/1/2014", end="3/31/2016")
    tweets_df = tweets_df.asfreq(pd.tseries.offsets.BDay())
    if to_csv:
        if not os.path.exists("temp/tweets/csv/"):
            os.makedirs("temp/tweets/csv/")
        tweets_df.to_csv("temp/tweets/csv/" + stock_name + ".csv")
    if to_pickle:
        if not os.path.exists("temp/tweets/pickle/"):
            os.makedirs("temp/tweets/pickle/")
        tweets_df.to_pickle("temp/tweets/pickle/" + stock_name + ".pickle")
    return tweets_df


def concat_data(df, to_csv=False, to_pickle=True):
    """
    concatenate tweets across all stocks for each day
    """
    data_list = []
    for j in range(df.shape[1]):
        temp = []
        for i in range(df.shape[0]):
            if df.iloc[i, j] != [[]] and df.iloc[i, j] != None:
                temp.extend(df.iloc[i, j])
        data_list.append(temp)
    data_list = [[x] for x in data_list]
    df_new = pd.DataFrame(data_list)
    if to_csv:
        df_new.to_csv("temp/tweets_concat.csv")
    if to_pickle:
        df_new.to_pickle("temp/tweets_concat.pickle")
    return df_new


def get_all_tweets(tweets_all_stocks):
    list_all_tweets = []
    for j in range(tweets_all_stocks.shape[1]):
        for i in range(tweets_all_stocks.shape[0]):
            if (
                tweets_all_stocks.iloc[i, j] != [[]]
                and tweets_all_stocks.iloc[i, j] != None
            ):
                list_all_tweets.extend(tweets_all_stocks.iloc[i, j])
    return list_all_tweets


def get_all_words(tweets_all_stocks):
    list_all_words = []
    for j in range(tweets_all_stocks.shape[1]):
        for i in range(tweets_all_stocks.shape[0]):
            if (
                tweets_all_stocks.iloc[i, j] != [[]]
                and tweets_all_stocks.iloc[i, j] != None
            ):
                for item in tweets_all_stocks.iloc[i, j]:
                    list_all_words += item
    with open("temp/corpus.txt", "w") as f:
        wr = csv.writer(f, delimiter=" ")
        wr.writerow(list_all_words)
    list_all_words = list(set(list_all_words))
    with open("temp/vocab.txt", "w") as f:
        wr = csv.writer(f, delimiter=" ")
        wr.writerow(list_all_words)
    return list_all_words


def load_prices(
    datapath="../data/stocknet-dataset/price/raw", to_csv=False, to_pickle=True
):
    df_list = []
    keys_list = []
    filelist = os.listdir(datapath)
    for filename in filelist:
        with open(os.path.join(datapath, filename), "r") as f:
            csv_reader = csv.reader(f, delimiter=",")
            temp_data = []
            for line in csv_reader:
                temp_data.append(line)
        df = pd.DataFrame(temp_data[1:])
        df.columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjusted close",
            "volume",
        ]
        df.index = df.date
        df.drop(columns=["date"], inplace=True)
        df_list.append(df)
        keys_list.append(filename[:-4])
    concat_df = pd.concat(df_list, keys=keys_list)
    if to_csv:
        concat_df.to_csv("temp/prices.csv")
    if to_pickle:
        concat_df.to_pickle("temp/prices.pickle")
    return concat_df


def get_price_df(df, to_csv=False, to_pickle=True):
    price_list = []
    stocks = set(df.index.get_level_values(0).to_list())
    for stock in stocks:
        price_list.append(df.xs(stock)["adjusted close"])
    price_df = pd.DataFrame(price_list, dtype="float32")
    price_df.index = stocks
    price_df.sort_index(axis=1, inplace=True)
    price_df = price_df.loc[:, "2013-12-31":"2016-03-31"]
    price_df = price_df.T
    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.reindex(pd.date_range(start="31/12/2013", end="3/31/2016"))
    price_df = price_df.asfreq(pd.tseries.offsets.BDay())
    price_df.fillna(method="pad", inplace=True)
    price_df = price_df.iloc[1:, :]
    price_df = price_df[stock_universe]
    if to_csv:
        price_df.to_csv("temp/close_prices.csv")
    if to_pickle:
        price_df.to_pickle("temp/close_prices.pickle")
    return price_df


def get_return_by_stock(stock_name, df, to_csv=False, to_pickle=True):
    df = df.xs(stock_name)
    df = df.astype("float32")
    df.sort_index(axis=0, inplace=True)
    df = df.loc["2013-12-31":"2016-03-31", "adjusted close"]
    df.index = pd.to_datetime(df.index)
    df = df.reindex(pd.date_range(start="31/12/2013", end="3/31/2016"))
    df = df.asfreq(pd.tseries.offsets.BDay())
    df.fillna(method="pad", inplace=True)
    df = df.pct_change(1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method="pad", inplace=True)
    # note: current method doesn't account for holidays
    df = df.iloc[1:]
    if to_csv:
        if not os.path.exists("temp/returns/csv/"):
            os.makedirs("temp/returns/csv/")
        df.to_csv("temp/returns/csv/" + stock_name + ".csv")
    if to_pickle:
        if not os.path.exists("temp/returns/pickle/"):
            os.makedirs("temp/returns/pickle/")
        df.to_pickle("temp/returns/pickle/" + stock_name + ".pickle")
    return df


def load_predictions(path_to_pred, path_to_output, to_csv=False, to_pickle=True):
    filelist = os.listdir(path_to_pred)
    stock_list = []
    df_list = []
    for filename in filelist:
        with open(os.path.join(path_to_pred, filename), "r") as f:
            csv_reader = csv.reader(f, delimiter=",")
            temp_data = []
            for line in csv_reader:
                temp_data.append(line)
            columns = temp_data.pop(0)
            temp_df = pd.DataFrame(temp_data)
            temp_df = temp_df.apply(pd.to_numeric)
            temp_df.columns = columns
            df_list.append(temp_df["pred"].values)
            stock_list.append(
                os.path.basename(os.path.normpath(filename))[:-4]
            )  # taking away '.csv' 4 chars
    df = pd.DataFrame(df_list)
    df.index = stock_list
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)
    if to_csv:
        df.to_csv(os.path.join(path_to_output, "pred.csv"))
    if to_pickle:
        df.to_pickle(os.path.join(path_to_output, "pred.pickle"))
    return df


def get_etf_mean_var(
    tickers=["SPY", "EEM", "GDX", "XLF", "QQQ", "USO", "TVIX", "EWZ", "EFA"],
    start=datetime(2014, 1, 1),
    end=datetime(2016, 3, 31),
):
    mean_var = {}
    for ticker in tickers:
        data = yf.download(ticker, start, end)
        data = data.asfreq(pd.tseries.offsets.BDay())
        data.fillna(method="pad", inplace=True)
        data = data.pct_change(1)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(method="pad", inplace=True)
        data = data.loc[:, "Adj Close"]
        data = data.values[-int(len(data) * 0.1) :]
        mean_var[ticker] = (np.mean(data), np.sqrt(np.cov(data).item()))
    return mean_var

