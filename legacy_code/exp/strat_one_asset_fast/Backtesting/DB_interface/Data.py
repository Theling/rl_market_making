import pandas as pd
import os
# pd.read_csv('data/tmp.csv', index_col=0)
cwd = os.getcwd()
FIELD = ['State2', 'Close2']
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
States = pd.read_csv(f'data/{FIELD[0]}.csv', index_col=0)
Close = pd.read_csv(f'data/{FIELD[1]}.csv', index_col=0)

# Calendar = pd.read_csv('data/tmp.csv', index_col=0)
os.chdir(cwd)

