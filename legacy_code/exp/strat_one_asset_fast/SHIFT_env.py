import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)

import shift
from threading import Thread, Lock
import pandas as pd
import numpy as np
import time
import queue
import datetime
import os
from portfolio_value import portfolioValue

class CirList:
    def __init__(self, length):
        self.size = length
        self._table = [None]*length
        self.idx = 0
        self._counter = 0

    def insertData(self, data):
        self._counter += 1
        self._table[self.idx] = data
        self.idx = (self.idx+1) % self.size

    def getData(self):
        tail = self._table[0:self.idx]
        head = self._table[self.idx:]
        ret = head+tail
        return ret.copy()

    def isFull(self):
        return self._counter >= self.size

    def __repr__(self):
        return str(self.getData())


class OrderBook:
    def __init__(self, AskOrder, BidOrder, last_price):
        self.last_price = last_price
        idx = 0
        tmp = pd.DataFrame(columns=['price', 'size', 'type', 'last_price'])
        ls = AskOrder

        for order in ls[::-1]:
            tmp.loc[idx] = [order.price, order.size, 'Ask', last_price]
            idx += 1

        self.n_ask = idx
        ls = BidOrder

        for order in ls:
            tmp.loc[idx] = [order.price, order.size, 'Bid', last_price]
            idx += 1

        self.n_bid = idx - self.n_ask

        self.df = tmp

    def __repr__(self):
        return str(self.df)



class SHIFT_env:
    def __init__(self,
                 trader,
                 scanner_wait_time,
                 decision_gap,
                 nTimeStep,
                 ODBK_range,
                 symbol,
                 executioner,
                 shares_factor = 2,
                 execution_times = 5,
                 max_step = 2000,
                 commission = 0.003,
                 rebate = 0.002,
                 save_data = True):

        #         trader = shift.Trader("test001")
        #         assert trader.connect("initiator.cfg", "password"), f'Connection error'
        #         assert trader.subAllOrderBook(), f'subscription fail'

        self.timeInterval = scanner_wait_time
        self.symbol = symbol
        self.nTimeStep = nTimeStep
        self.ODBK_range = ODBK_range
        self.trader = trader
        self.commission = commission
        self.rebate = rebate
        self.mutex = Lock()
        self.executioner = executioner
        self.execution_time = execution_times
        self.max_step = max_step
        self.decision_gap = decision_gap
        self.shares_factor = shares_factor

        self.dataThread = Thread(target=self._link)
        # features = ['symbol', 'orderPrice', 'orderSize', 'OrderTime']
        self.table = CirList(nTimeStep)
        self._cols = ['BA_spead', 'last_traded_price', 'Smart_price', 'Liquidity_imb', 'market_cost',
                'reward(log_r)','port_v','last_savings', 'delta_shares',  'done']
        self.df = pd.DataFrame(columns=self._cols)
        self.df_idx = 0

        print('Waiting for connection', end='')
        for _ in range(5):
            time.sleep(1)
            print('.', end='')
        print()

        self.thread_alive = True
        self.dataThread.start()

        self.remained_share = None
        self.total_share = None
        self.currentPos = None
        self.objPos = None
        self.isBuy = None
        self.tmp_obs = [None]*5
        self.name = 'strat_one_asset'
        self.isSave = save_data
        self.counter = 0

        self.rewardPipe = queue.Queue()
        self.execThread = None

    @staticmethod
    def action_space():
        return 1

    @staticmethod
    def obs_space():
        return 5

    def _link(self):
        while self.trader.isConnected() and self.thread_alive:

            last_price = self.trader.getLastPrice(self.symbol)

            Ask_ls = self.trader.getOrderBook(self.symbol, shift.OrderBookType.GLOBAL_ASK, self.ODBK_range)
            # assert Ask_ls, f'getOrderBook: return empty list: {self.symbol}-ASK-{self.ODBK_range}'

            Bid_ls = self.trader.getOrderBook(self.symbol, shift.OrderBookType.GLOBAL_BID, self.ODBK_range)
            # assert Bid_ls, f'getOrderBook: return empty list: {self.symbol}-BID-{self.ODBK_range}'

            orders = OrderBook(Ask_ls, Bid_ls, last_price)

            self.mutex.acquire()
            # print(88)
            self.table.insertData(orders)
            # print(tmp)
            self.mutex.release()

            time.sleep(self.timeInterval)
        print('Data Thread stopped.')

    @staticmethod
    def log_return(pv, cv):
        ret = np.log(pv / cv)
        return ret

    def _execute(self, share, exec_time):
        ret = self.executioner.execute(share, exec_time)
        self.rewardPipe.put(ret)

    def step(self, action):
        self.counter += 1
        self.currentPos = self._getCurrentPosition()
        real_action = 1 if action[0]>0 else -1
        delta_shares = real_action*self.shares_factor - self.currentPos
        if not delta_shares == 0:
            if self.execThread is not None:
                print('Last execution threading is still alive.')
                self.execThread.join(timeout=60)
                print('Last execution threading ends.')
            print(f'exec instruction: {int(delta_shares)}')
            self.execThread = Thread(target=self._execute, args=(int(delta_shares), self.execution_time))
            self.execThread.start()
        try:
            savings = self.rewardPipe.get(block=False)
        except queue.Empty:
            savings = ''
        done = False

        if self.counter > self.max_step:
            done = True

        time.sleep(self.decision_gap)

        next_obs = self._get_obs(delta_shares)
        cv = portfolioValue(self.trader)
        print(f'portfolio_value: {cv}')
        rwd = self.log_return(self.pv, cv)

        if self.isSave:
            tmp = next_obs[-1].tolist()+[rwd, cv, savings, delta_shares, done]
            # print('-------------', self.tmp_obs)
            self.df.loc[self.df_idx] = tmp
            self.df_idx += 1

        self.pv = cv

        return next_obs, rwd, done, dict()

    def _get_orderID(self):
        ID = []
        for i in self.trader.getSubmittedOrders():
            ID.append(i.id)
        return ID[-1]

    def _cancelAllOrder(self, order):
        if order.type == shift.Order.LIMIT_BUY or order.type == shift.Order.MARKET_BUY:
            order.type = shift.Order.CANCEL_BID
        elif order.type == shift.Order.LIMIT_SELL or order.type == shift.Order.MARKET_SELL:
            order.type = shift.Order.CANCEL_ASK
        else:
            raise TypeError

        tmp_con = 0
        self.trader.submitOrder(order)
        print('Canceling order:', end='')
        status = shift.Order.CANCELLED
        if status == False:
            tmp_con += 1
            time.sleep(0.05)
            if tmp_con > 1000:
                print(f'\n current order info: %6s\t%21s\t%7.2f\t\t%4d\t%36s\t%26s' %
                      (order.symbol, order.type, order.price, order.size, order.id, order.timestamp))
                print("Symbol\t\t\t\t\t Type\t  Price\t\tSize\tID\t\t\t\t\t\t\t\t\t\tTimestamp")
                for od in self.trader.getWaitingList():
                    print("%6s\t%21s\t%7.2f\t\t%4d\t%36s\t%26s" %
                          (od.symbol, od.type, od.price, od.size, od.id, od.timestamp))
                raise TimeoutError(f'Waited for canceling order for {tmp_con * 0.05} seconds.')
        else:
            print(' done.')

    def _get_obs(self, delta):
        self.tmp_obs = self.compute(delta)
        return self.tmp_obs.copy()
    def _getClosePrice(self, share):
        return self.trader.getClosePrice(self.symbol, self.isBuy, abs(share))

    def reset(self):
        print(f'Holding shares: {self.trader.getPortfolioItem(self.symbol).getShares()}')
        print(f'Buying Power: {self.trader.getPortfolioSummary().getTotalBP()}')
        self.pv = portfolioValue(self.trader)
        self.counter = 0
        return self._get_obs(0)

    def save_to_csv(self, epoch):
        try:
            self.df.to_csv(f'./iteration_info/itr_{epoch}.csv')
            self.df = pd.DataFrame(columns=self._cols)
        except FileNotFoundError:
            os.makedirs(f'./iteration_info/', exist_ok= True)
            self.df.to_csv(f'./iteration_info/itr_{epoch}.csv')
            self.df = pd.DataFrame(columns=self._cols)


    def kill_thread(self):
        self.thread_alive = False

    def compute(self, delta):
        tab = self.table
        feature = []
        if tab.isFull():
            for ele in tab.getData():
                #feat_step = pd.DataFrame(columns=['ba_spread', 'price', 'smart_price', 'liquid_imbal', 'market_cost'])
                #n_ask = ele.type.value_counts()['Ask']
                n_ask = ele.n_ask
                assert n_ask > 0, f'The number of Ask order is 0'
                #n_bid = len(ele) - n_ask
                n_bid = ele.n_bid
                assert n_bid > 0, f'The number of Bid order is 0'
                act_direction = 'Buy' if self.isBuy else 'Sell'
                bas = self._ba_spread(ele.df, n_ask)
                p = self._price(ele.df)
                sp = self._smart_price(ele.df, n_ask)
                li = self._liquid_imbal(ele.df, n_ask, n_bid, act_direction)
                if act_direction:
                    mc, _ = self._market_cost(ele.df,
                                              n_ask,
                                              n_bid,
                                              act_direction,
                                              delta,
                                              self.commission)
                    feature.append([bas, p, sp, li, mc])
                else:
                    feature.append([bas, p, sp, li, np.nan])
        return np.array(feature)

    @staticmethod
    def _ba_spread(df, n_ask):
        spread = df.price[n_ask - 1] - df.price[n_ask]
        return spread

    @staticmethod
    def _price(df):
        return df.last_price[0]/1000

    @staticmethod
    def _smart_price(df, n_ask):
        price = (df['size'][n_ask] * df.price[n_ask - 1] + df['size'][n_ask - 1] * df.price[n_ask]) \
                / (df['size'][n_ask] + df['size'][n_ask - 1])
        return price/1000

    @staticmethod
    def _liquid_imbal(df, n_ask, n_bid, act_direction):
        if n_ask > n_bid:
            imbal = df['size'][n_ask:].sum() - df['size'][(n_ask - n_bid):n_ask].sum()
        else:
            imbal = df['size'][n_ask:(2 * n_ask)].sum() - df['size'][0:n_ask].sum()
        if act_direction == 'Sell':
            imbal = -imbal
        return imbal/1000

    @staticmethod
    def _market_cost(df, n_ask, n_bid, act_direction, shares, commission):
        if act_direction == 'Buy':
            counter = df['size'][n_ask-1]
            n_cross = 1
            while counter < shares and n_ask-1 >= n_cross:
                counter += df['size'][n_ask-1-n_cross]
                n_cross += 1
            if n_cross > 1:
                sub_size = np.array(df['size'][(n_ask-n_cross):n_ask])
                sub_price = np.array(df.price[(n_ask-n_cross):n_ask])
                sub_size[0] = shares - sum(sub_size) + sub_size[0]
                market_price = sub_size.dot(sub_price)/shares
                cost = shares*(market_price - df.price[n_ask] + df.price[n_ask-1]*commission)
            else:
                market_price = df.price[n_ask-1]
                cost = shares*(market_price*(1+commission)-df.price[n_ask])
        else:
            counter = df['size'][n_ask]
            n_cross = 1
            while counter < shares and n_cross <= n_bid-1:
                counter += df['size'][n_ask+n_cross]
                n_cross += 1
            if n_cross > 1:
                sub_size = np.array(df['size'][n_ask:(n_ask+n_cross)])
                sub_price = np.array(df.price[n_ask:(n_ask+n_cross)])
                sub_size[-1] = shares - sum(sub_size) + sub_size[-1]
                market_price = sub_size.dot(sub_price)/shares
                cost = shares*(market_price - df.price[n_ask-1] + df.price[n_ask]*commission)
            else:
                market_price = df.price[n_ask]
                cost = shares*(market_price*(1+commission) - df.price[n_ask-1])
        return cost/1000, market_price

    def close_all(self):
        share = self.trader.getPortfolioItem(self.symbol).getShares()
        BP = self.trader.getPortfolioSummary().getTotalBP()
        waitingStep = 0
        small_order = 1
        while share != 0:
            position = int(share / 100)
            orderType = shift.Order.MARKET_BUY if position < 0 else shift.Order.MARKET_SELL

            if share < 0 and BP < abs(share) * self.trader.getClosePrice(self.symbol, True, abs(position)):
                order = shift.Order(orderType, self.symbol, small_order)
                self.trader.submitOrder(order)
                small_order *= 2
            else:
                order = shift.Order(orderType, self.symbol, abs(position))
                self.trader.submitOrder(order)

            time.sleep(0.5)
            #print(trader.getPortfolioItem(symbol).getShares())
            #print(trader.getPortfolioSummary().getTotalBP())
            share = self.trader.getPortfolioItem(self.symbol).getShares()
            waitingStep += 1
            assert  waitingStep < 40



    def _getCurrentPosition(self):
        return int(self.trader.getPortfolioItem(self.symbol).getShares() / 100)

    def __del__(self):
        self.kill_thread()

#         self.trader.disconnect()

class MarketOrder_exec:
    def __init__(self,
                 trader,
                 symbol):
        self.trader = trader
        self.symbol = symbol

    def execute(self, share, _):
        if share > 0:
            orderType = shift.Order.MARKET_BUY
        else:
            orderType = shift.Order.MARKET_SELL

        order = shift.Order(orderType, self.symbol,size = abs(share))
        self.trader.submitOrder(order)
        order_ = self.trader.getOrder(order.id)
        while not order_.status == shift.Order.FILLED:
            order_ = self.trader.getOrder(order.id)
            time.sleep(0.1)
        return 0.0

if __name__=='__main__':
    trader = shift.Trader("test002")
    trader.disconnect()
    trader.connect("initiator.cfg", "password")
    trader.subAllOrderBook()
    exec = MarketOrder_exec(trader, 'AAPL')
    symbol = 'AAPL'
    # if trader.isConnected() and trader.subAllOrderBook():
    #     print(trader.isConnected())
    #     order = trader.getOrderBook("AAPL", shift.OrderBookType.GLOBAL_ASK, 1)[0]
    #     print([order.price, order.size, 'Ask'])

    # time.sleep(5)
    # for i in range(10000):
    #     order = trader.getOrderBook("AAPL", shift.OrderBookType.GLOBAL_ASK, 1)[0]
    #     print([order.price, order.size, 'Ask'])
    #     del order
    #     time.sleep(1)

    env = SHIFT_env(trader,
                    scanner_wait_time = 1,
                    decision_gap = 10,
                    nTimeStep = 10,
                    ODBK_range = 10,
                    symbol = 'AAPL',
                    executioner = exec)


    time.sleep(15)



    obs = env.reset()
    pv = pd.DataFrame([[datetime.datetime.now(), 1000000, 0]],index=['row'],columns=['TimeStep', 'Time', 'PortfolioValue', 'CurrentShares'])
    df = pd.DataFrame(index=['row'],
                      columns=['Time', 'BA_spread', 'Price', 'Smart_price', 'Liquid_imb', 'MKT_cost'])
    df_save_name = 'Daily data'
    pv_save_name = 'pv Time Series.csv'
    pv.to_csv(pv_save_name)
    df.to_csv(df_save_name)
    print(os.getcwd())
    total_reward = 0
    print(f'The first obs: {obs}')
    for i in range(100):
        action = np.random.choice([-1, 1], size=1)
        print(f'action: {action}')
        tmp_pv = portfolioValue(trader)
        port_shares = trader.getPortfolioItem(symbol).getShares()/100
        pv.loc['row'] = [datetime.datetime.now(),tmp_pv,port_shares]
        pv.to_csv(pv_save_name, mode='a',header=False)
        obs, reward, done, _ = env.step(action)
        for j in range(len(obs)):
            df.loc['row'] = [datetime.datetime.now(),obs[j,0], obs[j,1], obs[j,2], obs[j,3], obs[j,4]]
            df.to_csv(df_save_name, mode='a', header=False)
            #print(f'BA_spread:{df.loc["row"][0]}, Price: {df.loc["row"][1]}, Smart_price: {df.loc["row"][2]}, Liquid_imb: {df.loc["row"][3], }, MKT_cost: {df.loc["row"][4]})')
        print(f'holding shares: {trader.getPortfolioItem("AAPL").getShares()}')
        print(obs)
        print(reward)
        total_reward += reward
        print(done)

        if done:
            print(f'finished')
            break
    env.save_to_csv(0)
    print(f'Total reward is {total_reward}')
    print()

    print(trader.getPortfolioItem('AAPL').getShares())

    for order in trader.getWaitingList():
        print(1)
        if order.type == shift.Order.LIMIT_BUY:
            order.type = shift.Order.CANCEL_BID
        elif order.type == shift.Order.LIMIT_SELL:
            order.type = shift.Order.CANCEL_ASK
        trader.submitOrder(order)

    print(trader.getPortfolioSummary().getTotalBP())

    env.dataThread.is_alive()
    #
    env.kill_thread()
    #
    env.dataThread.is_alive()

    trader.disconnect()

