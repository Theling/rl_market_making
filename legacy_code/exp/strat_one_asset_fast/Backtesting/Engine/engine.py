
if __name__ == '__main__':
    import os, sys
    lib_path = os.path.abspath(os.path.join('..'))
    sys.path.append(lib_path)
    from  DB_interface import Data as dt

    from statistics import Stat
else:
    from ..DB_interface import Data as dt
    from .statistics import Stat
from copy import deepcopy

from collections import OrderedDict
import pandas as pd
import numpy as np


def alpha():
    pass

class Market:
    def __init__(self, start_date, end_date, cash, isClose = True, commision = 0.00, rf = 0.0, universe=None):
        self.universe = universe


        self.init_cash = cash
        self.start_date = start_date
        self.end_date = end_date
        self.commision = commision
        self.rf = rf

        self.portfolio = None
        self.market_account = {'long':0.0,
                               'short':0.0}
        self.cash_account = {'cash':self.init_cash,
                             'short_proceed':None}
        self.portfolio_account = {}
        self.today = None

        self.price = None
        self.today_price = None
        self.today_available_stock = None
        self.date_index = None
        self._done_flag = None

        if isClose:
            self.price = dt.Close
        else:
            self.price = dt.Open

        if not self.universe:
            self.universe = list(self.price.columns)

        self.n_stock = len(self.universe)
        self.stats = None
        self.reset()



    def reset(self):
        self.portfolio = pd.DataFrame([0] * self.n_stock, index=self.universe, columns=['qty'])
        self.market_account['long'] = 0.0
        self.market_account['short'] = 0.0
        self.cash_account['cash'] = self.init_cash
        self.cash_account['short_proceed'] = 0.0
        self._update_portfolio_account()

        self.today = self.start_date
        tmp = list(self.price.index)
        # print(tmp)
        assert self.start_date in tmp, f'start date: {self.start_date} is not in trading calendar.'
        assert self.end_date in tmp, f'end date: {self.end_date} is not in trading calendar.'
        self.date_index = tmp.index(self.today)
        self.today_price = pd.DataFrame(self.price.iloc[self.date_index, :])
        self.today_available_stock = list(self.today_price.index)
        assert self.today_price.columns[0] == self.today, f'date does not match {self.today_price.columns[0] }, {self.today}'
        self._done_flag = False
        self.portfolio_account['total_commision'] = 0.0
        self.stats = Stat(features=['portfolio_value'],
                          assets=self.universe,
                          int_rate=self.rf)
        self.stats.dump([self.portfolio_account['portfolio_value']],
                        (self.portfolio['qty'] * self.today_price.iloc[:, 0]).fillna(0).tolist(),
                        self.today)
        print('Simulation reset!')


    def _update_portfolio_account(self):
        self.portfolio_account['portfolio_value'] = self.market_account['long']+self.cash_account['cash']+self.cash_account['short_proceed']-self.market_account['short']
        self.portfolio_account['buying_power'] = self.portfolio_account['portfolio_value']+self.cash_account['cash'] - self.cash_account['short_proceed']

    def _update_market_account(self):
        self.market_account['long'] = ((self.portfolio.loc[self.portfolio['qty']>0]['qty']*self.today_price.iloc[:,0]).dropna()).values.sum()
        self.market_account['short'] = -((self.portfolio.loc[self.portfolio['qty'] < 0]['qty'] * self.today_price.iloc[:, 0]).dropna()).values.sum()

    def get_date(self):
        return deepcopy(self.today)

    def next(self):
        if not self._done_flag:
            self.date_index += 1
            self._update_today_price()
            self.today_available_stock = list(self.today_price.index)
            self._update_market_account()
            self._update_portfolio_account()
            self.cash_account['cash'] *= np.exp(self.rf/250)
            self.stats.dump([self.portfolio_account['portfolio_value']],
                            (self.portfolio['qty']*self.today_price.iloc[:,0]).fillna(0).tolist(),
                            self.today)
            if self.today == self.end_date:
                self._done_flag = True
        else:
            print('Simulation finished.')
            return 0


    # def placeOneOrder(self, symbol, quantity):
    #     self.portfolio[symbol] += quantity


    def adjustPosition(self, new_position):
        '''
        :param new_position: int, target position.
        :return:
        '''
        if not set(new_position.index).issubset(self.today_available_stock):
            for stock in new_position.index:
                if stock not in self.today_available_stock:
                    raise ValueError(f'"{stock}" is not available today.')
        market_value = np.abs(new_position.iloc[:,0]*self.today_price.iloc[:,0]).sum()
        total_buying_power = self.portfolio_account['portfolio_value']+self.cash_account['cash']+self.market_account['long']-self.market_account['short']+self.cash_account['short_proceed']
        assert market_value < total_buying_power, f'Buying power low: {total_buying_power}, {market_value} is required.'
        positionChange = (self.portfolio['qty']-new_position.iloc[:,0]).dropna()
        commision = np.abs(positionChange.dropna()*self.commision).sum()
        self.portfolio['qty'] = (self.portfolio['qty']*0 + new_position.iloc[:, 0]).fillna(0)
        self._update_market_account()
        # self.portfolio_account['buying_power'] = 2*self.portfolio_account['portfolio_value']- self.market_account['long'] - self.market_account['short']
        self.cash_account['cash'] = (self.portfolio_account['portfolio_value'] - self.market_account['long'])-commision
        self.portfolio_account['total_commision'] += commision
        assert self.cash_account['cash']>0, f'cash is negative: {self.cash_account["cash"]}'
        self.cash_account['short_proceed'] = self.market_account['short']
        self._update_portfolio_account()

    def summaryToday(self):
        tmp = {}
        tmp.update(self.cash_account)
        tmp.update(self.portfolio_account)
        tmp.update(self.market_account)
        return pd.Series(tmp)



    def _update_today_price(self):
        self.today_price = pd.DataFrame(self.price.iloc[self.date_index, :])
        self.today = self.today_price.columns[0]
        self.today_available_stock = list(self.today_price.index)


if __name__=='__main__':
    mkt = Market(start_date=110,
                 end_date=11000,
                 cash=1000000,
                 rf = 0)
    print(mkt.cash_account, mkt.portfolio_account, mkt.market_account)
    position = pd.DataFrame([1], index=['AAPL'])
    mkt.adjustPosition(position)
    print(mkt.cash_account, mkt.portfolio_account, mkt.market_account)
    mkt.next()
    print(mkt.cash_account, mkt.portfolio_account, mkt.market_account)
    print(mkt.get_date())
    while True:
        ret = mkt.next()
        print(mkt.get_date())
        print(mkt.cash_account, mkt.portfolio_account, mkt.market_account)

        if ret == 0:
            break

    a, b = mkt.stats.compute(time_range=10, isPlot=True)
    print(b)