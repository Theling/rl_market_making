import pandas as pd
import numpy as np
# if __name__ == '__main__':
#     from features import Features
#     from misc import Recorder
#     from market import Market
#     from brokerage import Brokerage, Market_Order
# else:
from .features import Features
from .misc import Recorder
from .market import Market
from .brokerage import Brokerage, Market_Order

class RLBroker:
    def __init__(self, brokerage):
        self.brokerage = brokerage
        self.market = brokerage.market
        self.features_comp = Features()
        self.name = 'RL_trading'
    
    @property
    def act_dim(self):
        return 1
    
    @property
    def obs_dim(self):
        return len(self.features_comp.feature_list)

    @property
    def act_space(self):
        return np.arange(-5, 6, 1)

    @property
    def act_num(self):
        return len(self.act_space)

    @property
    def data(self):
        return self.brokerage.recorder.table

    # def _signal_to_action(self, signal):
    #     '''
    #     We consider single instrument currently.
    #     To consider expanding the ability of this class to multiple instrument,
    #     action should be a list of dictionaries.
    #     order   = {'instrument': str ,
    #               'time': local_time,
    #               'trade_direction': 1 or -1
    #               'quantity': int,
    #               'price': float}
    #     action = list of orders
    #     '''
    #     pass #TODO

    def _get_obs(self):
        data = self.market.get_data()
        # return self.features_comp.comp(data)
        #TODO
        return 'temp obs placeholder'

    def step(self, qty):
        order = Market_Order('sc', np.sign(qty), abs(qty))
        self.brokerage.submit_order(order)

        self.brokerage.next()

        next_obs = self._get_obs()
        # TODO: terminal condition
        terminal = False
        # TODO: reward function
        reward = self.data['pnl'][len(self.data)-1]
        return next_obs, terminal, reward, {}


    def reset(self):
        self.brokerage.reset()
        return self._get_obs()





if __name__=='__main__':
    import os
    os.chdir('Envs')
    print(os.getcwd())

    mkt = Market('mkt_data/TRV20200220',
                 'sc',
                 0.00025,
                 threshold=0,
                 reverse_position=0,
                 MAX_BID=0)

    broker = RLBroker(Brokerage(mkt))

    for _ in range(10):
        action = np.random.randint(-5,5)
        print()
        print(mkt.exch_time)
        print(broker.brokerage.account)
        print(broker.step(action))
        print(broker.brokerage.account)
    print(broker.data)
    broker.reset()
    for _ in range(10):
        action = np.random.randint(-5,5)
        print()
        print(mkt.exch_time)
        print(broker.brokerage.account)
        print(broker.step(action))
        print(broker.brokerage.account)
    print(broker.data)