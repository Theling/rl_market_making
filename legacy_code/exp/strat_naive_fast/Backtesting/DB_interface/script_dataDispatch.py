import pandas as pd

#data = pd.read_csv('./data/Daily2.csv', header=[0,1], index_col = 0)
data = pd.read_csv('./data/DJI.csv', header=[0,1], index_col = 0)
#features = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
features = ['Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close']

for feature in features:
    tmp = data[feature]
    tmp.columns = [i.split(' ')[0] for i in tmp.columns]
    tmp.to_csv('data/'+feature+'.csv')

