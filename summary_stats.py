import os
import json
import pandas as pd


def get_summary_stats(
    datapath="../data/stocknet-dataset/tweet/new_processed", to_csv=True
):
    folderpath = [x[0] for x in os.walk(datapath)][1:]
    df_list = list()
    for folder in folderpath:
        print(f"current folder: {folder}")
        filelist = os.listdir(folder)
        filelist.sort()
        n_files = 0
        n_tweets = 0
        for filename in filelist:
            with open(os.path.join(folder, filename), "r") as f:
                line = f.readline()
                line = json.loads(line)
                n_tweets += int(list(line.keys())[-1])
                n_files += 1
        df_list.append((os.path.basename(os.path.normpath(folder)), n_tweets, n_files))
    df = pd.DataFrame(df_list)
    df.columns = ["tic", "n_tweets", "n_days"]
    if to_csv:
        df.to_csv("summary_stats.csv")
    return df


if not os.path.exists("./summary_stats.csv"):
    df = get_summary_stats()
df = pd.read_csv("summary_stats.csv", index_col=0)
df["avg_tweets"] = df["n_tweets"] / df["n_days"]
df = df[df["avg_tweets"] >= 1]
print(df.describe())
