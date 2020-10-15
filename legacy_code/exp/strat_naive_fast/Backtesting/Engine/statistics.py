import matplotlib.pyplot as plt
# import scipy.stats as sp
import pandas as pd
import numpy as np
from collections import OrderedDict
import dateutil.parser


class Stat:
    def __init__(self, features, assets, **kwargs):
        self._market_info = kwargs
        self._date_ls = []
        self._insertIdx = 0
        self._lsFeature = features.copy()
        self._numFeature = len(self._lsFeature)
        self._lsAsset = assets.copy()
        self._numAsset = len(self._lsAsset)
        self._market_value = pd.DataFrame(columns=assets)
        self._data = pd.DataFrame(columns=features)


    def _dumpData(self, ls):
        assert self._numFeature == len(ls), f'Invalid input: length of list must be {self._numFeature}\n Or call set_feature() firstly.'
        self._data.loc[self._insertIdx] = ls

    def _dumpMarketValue(self, ls):
        assert self._numAsset == len(ls)
        self._market_value.loc[self._insertIdx] = ls


    def dump(self, account_data, market_value, date):
        self._dumpData(account_data)
        self._dumpMarketValue(market_value)
        self._date_ls.append(date)
        self._insertIdx += 1

    def get_features(self):
        assert self._lsFeature, f"Feature list doesn't exist."
        return self._lsFeature.copy()

    def add_market_info(self, dic):
        self._market_info.update(dic)

    def trading_log(self):
        ret1 = pd.DataFrame(self._data.values,
                            columns=self._lsFeature,
                            index=self._date_ls)
        ret2 = pd.DataFrame(self._market_value.values,
                            columns=self._lsAsset,
                            index=self._date_ls)
        return ret1, ret2

    def compute(self, time_range=10, isPlot = False):
        '''
        all, monthly, weekly, yearly

        :param time_range:
        :return:
        '''
        #assert time_range in ['all', 'monthly', 'weekly', 'yearly']

        date_size = self._data['portfolio_value'].index.size
        total_result = self._compute_one_period(self._data.index,
                                        date_size/250/24/3600,
                                        rf = np.exp(self._market_info['int_rate']/250/24/3600)-1)

        ret_table = pd.DataFrame(columns = list(total_result.keys()))
        # print(ret_table)
        index = []

        df = pd.Series(self._date_ls)
        # df = df.apply(lambda x: dateutil.parser.parse(x))
        # print(df)
        # df.index = pd.Series(df.index).apply(lambda x: dateutil.parser.parse(x))

        # if time_range == 'monthly':
        #     per = df.dt.to_period("M")
        # elif time_range == 'weekly':
        #     per = df.dt.to_period("W")
        # elif time_range == 'yearly':
        #     per = df.dt.to_period("Y")
        # else:
        #     print(total_result)
        #     if isPlot:
        #         self._plot_PnL(self._data['portfolio_value'])
        #     return total_result


        #df = df.reset_index(drop=True)
        agg = df.groupby(df.index // time_range)
        for i in range(len(agg)):
            tmp = agg.get_group(i)
            # print(tmp)
            tmp = self._compute_one_period(tmp.index,
                                     tmp.index.size / 250 / 24 / 3600,
                                     rf=np.exp(self._market_info['int_rate'] / 250 / 24 / 3600) - 1)
            index.append(i)
            ret_table.loc[len(index)] = list(tmp.values())
            ret_table.index = index


        print(ret_table)
        print(total_result)
        if isPlot:
            self._plot_PnL(self._data['portfolio_value'])
        return total_result, ret_table



    def _compute_one_period(self, data_index, tau, rf):

        '''
        TODO:
        self._data should be a pd.DataFrame, one columns is Portfortlio value,
        and row indicts DATE, you can compute the return by the change of this column.

        tau = data time range (in year), eg: 1 month: date_size = 1/12
        Data here is only a piece of data (1 month, 1 year etc)
        You can compute Sharpe ratio, max drawdown. Please write functions for this.
        I will arrange them once you finished it.

        :return:
        '''
        data = self._data.iloc[data_index,:]['portfolio_value']
        mkt_v = self._market_value.iloc[data_index,:]


        self._market_info['ret'] = data.pct_change()[1:]
        ret = OrderedDict()
        tmp = data.tolist()
        ret['return'] = np.log(tmp[-1]/tmp[0])/tau
        ret['Sharpe'] = self._sharpe_ratio(self._market_info['ret'], rf)
        ret['max_drawdown'] = self._max_drawdown(data)
        ret['turnover'] = self._turnover(mkt_v, data)
        ret['sterling_ratio'] = self._sterling_ratio(data, self._market_info['ret'], rf)

        return ret

    def _sharpe_ratio(self, ret, rf):
        """
        rf: the risk free rate, if not entered just use the default value of 0.005
        """
        # print(ret.mean(), rf, ret.std())
        return (ret.mean() - rf) / ret.std()

    def _drawdown(self, data):
        val = data.values
        maxs = np.maximum.accumulate(val)
        dd = 1 - val / maxs
        return dd


    def _max_drawdown(self, data):
        maxs = np.max(self._drawdown(data))
        return maxs


    def _inform_ratio(self, ret, bench):
        """
        bench: the return of benchmark, should be a pd.DataFrame
        """
        ret = ret.values
        bench = bench.values
        diff = ret - bench
        ir = diff.mean() / diff.std()
        return ir


    def _sterling_ratio(self, data, ret, rf):
        ave_dd = self._drawdown(data).mean()
        return (ret.mean() - rf) / ave_dd


    def _turnover(self, mktval, port_value):
        """
        mktval: the market value of the stocks, should be a pd.DataFrame with 2 columns
        """
        # port_value = self._data['portfolio_value']
        diff = np.abs(np.diff(mktval.values, axis=0)).sum(axis=1)
        to = diff / 2 / port_value.values[1:]
        # to = pd.DataFrame(to, columns=['turnover'], index=port_value.index[1:])
        return to.mean()


    def _plot_PnL(self, portfolio_value):
        tmp = portfolio_value
        date_tmp = pd.Series(self._date_ls)
        assert len(date_tmp) == len(tmp)
        plt.plot(range(len(tmp)), tmp)
        num_index = int(len(tmp)/10)
        ls = range(0, len(tmp), num_index)
        plt.xticks(ls, list(date_tmp.iloc[ls]), rotation=45)
        plt.show()



if __name__=='__main__':
    class A:
        pass
    mkt = A()

    portfolio_account = {'portfolio_value': 600}
    market_value = np.array([100, 200, 300])
    st = Stat(features = ['portfolio_value'],
              assets = ['AAPL', 'GOOG', 'AMZN'],
              int_rate=0.0001)

    for i in range(100):
        market_value = -1* market_value
        portfolio_account['portfolio_value'] += 0.01* np.random.normal()
        st.dump([portfolio_account['portfolio_value']], market_value, '2018-01-01')


    print(st.trading_log())
    print(st.compute())
    # print(st._market_info['ret'])


