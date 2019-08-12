from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from preprocessing import stock_universe
import numpy as np
import pandas as pd
import os, csv

def get_metric(y_pred, y_true, metric):
	return metric(y_pred, y_true)


def get_metrics_summary(model):
	path_to_pred = os.path.join('./output', model)
	result_list = []
	for stock in stock_universe:
	    with open(os.path.join(path_to_pred, stock + '.csv'), "r") as f:
	        csv_reader = csv.reader(f, delimiter=",")
	        temp_data = []
	        for line in csv_reader:
	            temp_data.append(line)
	        columns = temp_data.pop(0)
	        temp_df = pd.DataFrame(temp_data)
	        temp_df = temp_df.apply(pd.to_numeric)
	        temp_df.columns = columns
	        r2 = r2_score(temp_df.loc[:, 'returns'], temp_df.loc[:, 'pred'])
	        mse = mean_squared_error(temp_df.loc[:, 'returns'], temp_df.loc[:, 'pred'])
	        mae = median_absolute_error(temp_df.loc[:, 'returns'], temp_df.loc[:, 'pred'])
	        result_list.append((stock, r2, mse, mae))
	result_df = pd.DataFrame(result_list)
	result_df.columns = ['stock', 'r2', 'mse', 'mae']
	result_summary = result_df.describe()
	result_df.to_csv('output/return_pred_summary_' + model + '.csv')

	        