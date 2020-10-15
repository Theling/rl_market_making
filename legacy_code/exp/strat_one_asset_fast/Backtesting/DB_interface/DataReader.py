import pandas as pd
from . import Data as dt
import os


class dataReader:
    def __init__(self):
        self._close = dt.Close
        self.date_ls = list(self._close.index)
        self.today = self.date_ls[0]
        self.date_index = 0
        self.universe = list(self._close.columns)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.re_dir = os.path.join(dir_path,'./data')
        self.data = {}

        self.field = dt.FIELD

        self.init_data()

    def init_data(self):
        self.data = {}
        for f in self.field:
            self.data[f] = pd.read_csv(os.path.join(self.re_dir, f+'.csv'), index_col=0)

    def get(self, key, interval = 21):
        assert self.date_index-interval+1 >= 0, f'data reference is beyond limit (head)'
        tmp = self.data[key].iloc[(self.date_index-interval+1):(self.date_index+1),:]
        return tmp.copy()

    def set_date(self, date):
        self.today = date
        self.date_index = self.date_ls.index(self.today)

    def next(self):
        self.date_index += 1
        self.today = self.date_ls[self.date_index]

if __name__=='__main__':
    a = dataReader()
    a.set_date(50)
    print(a.get('Close', 11))
    a.next()
    print(a.get('Close', 11))
    b = a.get('Close', 11)
    print(b[['AAPL']])
    c = b[['AAPL']]
    print(c.pct_change()[1:].values)
    d = a.get('Close', 11)
    d = d['AAPL']
    print(d.pct_change()[1:].values)
    print(d.shape)