#TODO clean init func.
import pandas as pd
import numpy as np

class Market(object):
    def __init__(self, 
                 date, 
                 instrument, 
                 trade_fee_ratio, 
                 threshold, 
                 reverse_position, 
                 MAX_BID):
        self.date = date
        self.instrument = instrument
        self.trade_fee_ratio = trade_fee_ratio
        self.threshold = threshold
        self.reverse_position = reverse_position
        self.MAX_BID = MAX_BID
        self.signal_list = []
        self.MAX_POSITION = 12
        self.cash = 0
        self.cash_ideal = 0
        self.position = 0
        self.trade_volume = 0
        self.fee = 0
        self.num_trade = 0
        self.num_trade_volume = 0
        self.mkt_data = None
        self._load_data()
        self.__is_broker_linked = False
        
        
    def link(self, broker):
        self.broker = broker
        self.__is_broker_linked = True

    def _load_data(self):
        tmp = pd.read_csv(f'{self.date}.csv')
        self.mkt_data = tmp
        # self.mkt_data = self.mkt_data.set_index('local_time')
        self.max_idx = len(self.mkt_data.index)-1
        self.idx = 0 
    
    def _next(self):
        self.idx += 1

    def next(self):
        if self.__is_broker_linked:
            raise AssertionError(f'calling next() of linked market is prohibited ')
        self._next()
        

    # def trade(self, order):
    #     # TODO
    #     '''
    #     Should check the feasibility of incoming orders, 
    #     and return total value of instrument and fee (commission & rebate).
    #     '''

    def get_data(self, length = 1):
        idxs = list(range(max(0,self.idx-length), self.idx+1))
        return self.mkt_data.iloc[idxs, :].copy()
    
    @property
    def latest_tick(self):
        return self.mkt_data.iloc[self.idx, :]
    
    def get_close_price(self, qty):
        if qty > 0:
            dat = self.latest_orderbook['ask']
        elif qty < 0:
            dat = self.latest_orderbook['bid']
        else:
            return 0, qty
        n = len(dat.index)
        executed_qty = np.zeros(n)
        executed_price = np.zeros(n)
        remained_qty = qty
        for i in dat.index:
            executed_qty[i] = min(dat.loc[i, 'qty'], remained_qty)
            executed_price[i] = dat.loc[i, 'p']
            remained_qty -= executed_qty[i]
            if remained_qty <= 0:
                break
        if remained_qty > 0: raise ValueError(f'Total size exceeds the capacity')
        tot_qty = executed_qty.sum()
        tot_val = (executed_price*executed_qty).sum()
        effective_price = tot_val/tot_qty
        # print(f'-------DEBUG-------')
        # print(executed_price)
        # print(executed_qty)
        # print(f'-------FINISH-------')
        return tot_val/tot_qty, tot_qty
    # @property
    # def local_time(self):
    #     return self.latest_tick['local_time']

    @property
    def exch_time(self):
        return self.latest_tick['real_time']

    # @property
    # def quantity(self):
    #     return self.latest_tick['quantity']

    @property
    def best_ask_bid(self):
        a, b = self.latest_tick['ask_price_1'], self.latest_tick['bid_price_1']
        aq, bq = self.latest_tick['ask_size_1'], self.latest_tick['bid_size_1']
        return a, aq ,b, bq

    @property
    def mid_price(self):
        a,_,b,_ = self.best_ask_bid
        return a - b
    @property
    def latest_orderbook(self):
        return self._reconstruct_orderbook(self.latest_tick)

    @staticmethod
    def _reconstruct_orderbook(tick):
        ret = {}
        # ret['local_time'] = self.local_time
        ret['real_time'] = tick['real_time']
        bid_price = tick.loc[[f'bid_price_{i+1}' for i in range(5)]]#.to_list()
        bid_quantity = tick.loc[[f'bid_size_{i+1}' for i in range(5)]].to_list()
        ask_price = tick.loc[[f'ask_price_{i+1}' for i in range(5)]].to_list()
        ask_quantity = tick.loc[[f'ask_size_{i+1}' for i in range(5)]].to_list()
        bid = pd.DataFrame({'level': range(5), 'p': bid_price, 'qty': bid_quantity})
        ask = pd.DataFrame({'level': range(5), 'p': ask_price, 'qty': ask_quantity})
        ret['orderbook'] = {'ask': ask, 'bid': bid}
        return ret['orderbook']
    # def step(self, action):
    #     tick = self.mkt_data[self.idx]
    #     obs = self.features_comp.compute(tick)
        
    #     pass
    def reset(self, verbose = False):
        if self.__is_broker_linked:
            raise AssertionError(f'calling reset() of linked market is prohibited ')
        self._reset(verbose)


    def _reset(self, verbose = False):
        if verbose:
            print(f'market reset.')
        self.idx = 0

if __name__ == "__main__":
    import os
    # os.chdir('dev/Envs')
    print(os.getcwd())

    mkt = Market('mkt_data/TRV20200220',
                 'sc',
                 0.00025,
                 threshold=0,
                 reverse_position=0,
                 MAX_BID=0)
    # for _ in range(5):
    #     print(mkt.exch_time)
    #     # print(mkt.latest_tick)
    #     # print(mkt.latest_orderbook['ask'])
    #     # print(mkt.latest_orderbook['bid'])
    #     mkt.next()
    # print(mkt.get_data(5))
    # mkt.reset()
    for _ in range(5):
        print(mkt.exch_time)
        print(mkt.latest_orderbook['ask'])
        print(mkt.get_close_price(10))
        print(mkt.best_ask_bid)
        mkt.next()
