import pandas as pd

df_list = []
filename = "../data/stocknet-dataset/StockTable"
with open(filename, "r") as f:
    for line in f:
        df_list.append(tuple(line.strip("\n").split(sep="\t")))
df = pd.DataFrame(df_list)

df.columns = df.iloc[0, :]
df = df.iloc[1:, :]
df.to_csv("temp/stock_table.csv")


stock_list = []
filename = "../data/stocknet-dataset/StockTable"
with open(filename, "r") as f:
    count = 0
    for line in f:
        if count == 0:
            count += 1
            continue
        stock_list.append(tuple(line.strip("\n").split(sep="\t")[1:]))
stock_dict = dict(stock_list)

