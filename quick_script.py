import re, os, csv
from preprocessing import stock_universe
import pandas as pd

pattern1 = re.compile(r'\[\[(.*?)\]\]')
pattern2 = re.compile(r'\((.*?)\,')
test_str = "tensor(3.5053e-05, device='cuda:0')"

path_to_output = './output/tfidf'
for stock in stock_universe:
	filepath = os.path.join(path_to_output, stock + '.csv')
	temp = []
	with open(filepath, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		for line in reader:
			temp.append(line)
	columns = temp.pop(0)
	df = pd.DataFrame(temp)
	df.columns = columns
	df.drop(df.columns[0], axis=1, inplace=True)
	try:
		df['y'] = df['y'].apply(lambda x: re.search(pattern1, str(x)).groups()[0])
		df['pred'] = df['pred'].apply(lambda x: re.search(pattern1, str(x)).groups()[0])
		df['loss'] = df['loss'].apply(lambda x: re.search(pattern2, str(x)).groups()[0])
		df = df.astype(float)
		df.to_csv(os.path.join(path_to_output, stock + '.csv'))
	except AttributeError:
		continue