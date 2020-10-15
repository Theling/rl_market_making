import pandas as pd
import numpy as np
from .features import Features
from .misc import Recorder
from .market import Market

class Order(dict):
    '''
    {'instrument': str ,
                #   'time': exch_time,
                  'trade_direction': 1 or -1
                  'quantity': int,
                  'price': float}
    '''
    def __init__(self, 
                 instrument, 
                #  time, 
                 trade_direction, 
                 quantity,
                 price):
         
        assert quantity >= 0, f'{quantity}'
        if price is not None: assert price > 0, f'{price}'
        assert int(trade_direction) in [1,-1,0], f'{trade_direction}'
        if price is not None:
            type_ = 'limit'
        else:
            type_ = 'market'
        super().__init__(
                      instrument = instrument,
                      time = None,
                      type = type_,
                      trade_direction = trade_direction, 
                      quantity = quantity,
                      price = price)
    


class Market_Order(Order):
    def __init__(self, instrument, trade_direction, quantity):
        super().__init__(instrument, trade_direction, quantity, None)

class Brokerage:
    def __init__(self, market):
        # self.time_stamp = 0
        # self.pnl = []
        # self.pnl_ideal = []
        # self.position = []
        # self.position_cash_list = []
        # self.action_list = []
        # self.cash_list = []
        # self.cash_ideal_list = []
        # self.fee_list = []
        # self.volume_list = []
        # self.signal_list = []


        self.recorder = Recorder(max_size=None,
                                 fields = ['timestamp', 'total_val','pnl', 
                                #  'pnl_ideal', 
                                 'position', 'position_val',\
                                        #    'actions', 
                                           'cash', 
                                        #    'cash_ideal', 
                                           'fees', 'trade_num'
                                        #    'volumns', 'signals',
                                           ])
        self.market = market
        self.market.link(self)

        self._broker_reset(False)
        
    @property
    def _instrument_val(self):
        '''
        The value if close position immediately.
        '''
        eff_price, eff_qty = self.market.get_close_price(-self._instrument_qty)
        return -eff_price*eff_qty

    @property
    def time_stamp(self):
        return self.market.exch_time

    @property
    def _total_val(self):
        return self._cash_val + self._instrument_val

    @property
    def account(self):
        return {'cash_val': self._cash_val,
                        'asset_val': self._instrument_val, 
                        'asset_qty': self._instrument_qty,
                        'total_val': self._total_val,
                        'trade_times': len(self.order_flow),
                        'fees': self.fees}
    

    def submit_order(self, order, verbose=False):
        self.order_flow.append(order)
        self._check_limit_order(order)
        side, quantity = order['trade_direction'], order['quantity']
        effective_price, effective_qty = self.market.get_close_price(quantity)
        tot_val = effective_price*effective_qty
        fee = self.market.trade_fee_ratio * tot_val

        order['time'] = self.time_stamp
        pre_account = self.account

        self._cash_val -= side*tot_val
        self._cash_val -= fee
        self._instrument_qty += side*effective_qty
        self.trade_one_step += 1
        self.fees += fee
        if verbose:
            print(f'pre_account: {pre_account}')
            print(f'cur_account: {self.account}')
            print(f'fee: {fee}')
            
        
        return effective_price, effective_qty, fee

    def next(self):
        self.market._next()
        #------- trading info --------
        pnl = self.account['total_val'] - self.pre_account['total_val']
        trade_times = self.trade_one_step
        fees = self.account['fees'] - self.pre_account['fees']
        #------- 
        self.recorder.record(timestamp = self.time_stamp,
                             total_val = self.account['total_val'],
                             pnl = pnl,
                             position = self.account['asset_qty'], 
                             position_val = self.account['asset_val'],
                             cash = self.account['cash_val'],
                             fees = fees,
                             trade_num = trade_times)

        self.pre_account = self.account
        self.trade_one_step = 0

    def _broker_reset(self, verbose):
        if verbose:
            print('broker reset.')
        self.trade_times = 0
        self.fees = 0
        self.trade_one_step = 0
        self.order_flow = []
        self._cash_val = 0
        self._instrument_qty = 0
        self.pre_account = self.account

    def dump_data(self, file_, *args, **kwargs):
        self.recorder.dump(file_, *args, **kwargs)

    def reset(self, verbose = False):
        self.market._reset(verbose)
        self._broker_reset(verbose)
        self.recorder.reset()

    def _check_limit_order(self, order):
        if order['price'] is not None:
            side, price = order['trade_direction'], order['price']
            a, _, b, _ = self.market.best_ask_bid
            if side == 1:
                assert price >= a
            elif side == -1:
                assert price <= b
            else:
                raise ValueError






if __name__=='__main__':
    import os
    os.chdir('dev/Env')
    print(os.getcwd())

    mkt = Market('mkt_data/20190701',
                 'sc',
                 0.00025,
                 threshold=0,
                 reverse_position=0,
                 MAX_BID=0)

    broker = Brokerage(mkt)
    order = Market_Order('sc', 1, 10)
    print(broker.submit_order(order, verbose= True))
    for _ in range(5):
        print()
        print(mkt.exch_time)
        # print(mkt.latest_orderbook['bid'])
        print(mkt.get_close_price(-10))
        # print(mkt.best_ask_bid)
        # print(broker.account)
        broker.next()
    # print(broker.recorder.table)
    # broker.reset(verbose=True)
    
    order = Market_Order('sc', -1, 20)
    print(broker.submit_order(order, verbose= True))
    for _ in range(5):
        print()
        print(mkt.exch_time)
        # print(mkt.latest_orderbook['bid'])
        print(mkt.get_close_price(10))
        # print(mkt.best_ask_bid)
        print(broker.account)
        broker.next()
    print(broker.recorder.table)
    print(broker.order_flow)
    
    