import pandas as pd
import os
# pd.read_csv('data/tmp.csv', index_col=0)
cwd = os.getcwd()

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
States = pd.read_csv('data/State.csv', index_col=0)
Close = pd.read_csv('data/Close.csv', index_col = 0)

# Calendar = pd.read_csv('data/tmp.csv', index_col=0)
os.chdir(cwd)

