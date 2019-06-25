import os, json
import pandas as pd
import numpy as np


def load_data(datapath="../data/stocknet-dataset/tweet/new_processed"):
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
    return df


def concat_data(df, to_csv=False):
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
        df_new.to_csv("temp/data_concat.csv")
    return df_new


def get_sentences(df):
    data_list = []
    for j in range(df.shape[1]):
        temp = []
        for i in range(df.shape[0]):
            if df.iloc[i, j] != [[]] and df.iloc[i, j] != None:
                temp.extend(df.iloc[i, j])
    return temp
