from DB_interface.DataReader import dataReader
from Engine.engine import Market
import numpy as np
import pandas as pd



def scale(data):
    data = data.dropna()
    ret = data - np.mean(data)
    ret = 2*ret/np.sum(np.abs(ret))
    return ret


mkt = Market(start_date='2018-10-17',
             end_date='2018-11-15',
             cash=1000000)


dr = dataReader()
dr.set_date('2018-10-17')
while True:
    close = dr.get('Close', 5)
    print(mkt.today)
    print(mkt.today_price)
    print(close.iloc[-1,:])
    # print(mkt.cash_account, mkt.portfolio_account, mkt.market_account)
    # print(close)
    # tmp = 1000000*0.95*scale(close.iloc[-1,:])/close.iloc[-1,:]
    # tmp = pd.DataFrame(tmp)
    # mkt.adjustPosition(tmp)
    ret = mkt.next()
    dr.next()
    if ret == 0:
            break

