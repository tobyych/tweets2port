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


def get_ndays_above_avg(
    summary_stats, datapath="../data/stocknet-dataset/tweet/new_processed", to_csv=True
):
    summary_stats["avg_tweets"] = summary_stats["n_tweets"] / summary_stats["n_days"]
    summary_stats = summary_stats[summary_stats["avg_tweets"] >= 1]
    summary_stats.index = summary_stats["tic"]
    summary_stats.drop(columns=["tic"], inplace=True)
    folderpath = [x[0] for x in os.walk(datapath)][1:]
    folderpath.sort()
    list_ndays_above_avg = []
    for folder in folderpath:
        print(f"current folder: {folder}")
        filelist = os.listdir(folder)
        filelist.sort()
        ndays_above_avg = 0
        for filename in filelist:
            with open(os.path.join(folder, filename), "r") as f:
                line = f.readline()
                line = json.loads(line)
                daily_n_tweets = int(list(line.keys())[-1])
                if (
                    daily_n_tweets
                    >= summary_stats.loc[
                        os.path.basename(os.path.normpath(folder)), "avg_tweets"
                    ]
                ):
                    ndays_above_avg += 1
        list_ndays_above_avg.append(ndays_above_avg)
    series_ndays_above_avg = pd.Series(list_ndays_above_avg, index=summary_stats.index)
    summary_stats.loc[:, "ndays_above_avg"] = series_ndays_above_avg
    if to_csv:
        summary_stats.to_csv("new_summary_stats.csv")
    return summary_stats


if not os.path.exists("./summary_stats.csv"):
    df = get_summary_stats()
df = pd.read_csv("summary_stats.csv", index_col=0)
new_summary_stats = get_ndays_above_avg(df)
# df["avg_tweets"] = df["n_tweets"] / df["n_days"]
# df = df[df["avg_tweets"] >= 1]
