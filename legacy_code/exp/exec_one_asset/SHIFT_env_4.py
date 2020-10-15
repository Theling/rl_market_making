import shift
from threading import Thread, Lock
import pandas as pd
import numpy as np
import time
import os
import copy

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
        return copy.deepcopy(ret)

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
                 t, # backend data thread refresh frequency
                 nTimeStep, # Look back steps
                 ODBK_range,
                 symbol,
                 commission = -0.003,
                 rebate = 0.002,
                 save_data = True):

        self.timeInterval = t
        self.symbol = symbol
        self.nTimeStep = nTimeStep
        self.ODBK_range = ODBK_range
        self.trader = trader
        self.commission = commission
        self.rebate = rebate
        self.mutex = Lock()

        self.dataThread = Thread(target=self._link)
        # features = ['symbol', 'orderPrice', 'orderSize', 'OrderTime']
        self.table = CirList(self.nTimeStep)

        print('Waiting for connection', end='')
        for _ in range(5):
            time.sleep(1)
            print('.', end='')
        print()

        self.thread_alive = True
        self.dataThread.start()

        time.sleep(self.timeInterval * self.nTimeStep)
        self.remained_share = None
        self.total_share = None
        self.currentPos = None
        self.basePrice = None
        self.objPos = None
        self.isBuy = None
        self.remained_time = None
        self.tmp_obs = [None]*7
        self.name = 'exec_one_asset'

        # ----- dataframe config -------
        self.isSave = save_data
        self._cols = []
        for i in range(self.nTimeStep):
            self._cols += [f'BA_spead_{i}', f'last_traded_price_{i}', f'Smart_price_{i}', f'Liquidity_imb_{i}',
                           f'market_cost_{i}']
        self._cols += \
            ['signBuy', 'delta_base_price', 'remained_shares', 'remained_time',
             'reward', 'order_type', 'is_buy', 'premium',
             'obj_price', 'base_price', 'executed', 'done']

        self.df = pd.DataFrame(columns=self._cols)
        self.df_idx = 0

    def set_objective(self, share, remained_time):
        self.isBuy = True if share > 0 else False
        self.remained_share = abs(share)
        self.total_share = self.remained_share
        self.currentPos = self._getCurrentPosition()
        self.basePrice = self._getClosePrice(share)
        self.objPos = self.currentPos + share
        self.remained_time = remained_time
        print(f'set obj: {share}, time: {remained_time}, base_price: {self.basePrice}')

    @staticmethod
    def action_space():
        return 1

    def obs_space(self):
        return 4 + self.nTimeStep * 5

    def _link(self):
        while self.trader.is_connected() and self.thread_alive:

            last_price = self.trader.get_last_price(self.symbol)

            Ask_ls = self.trader.get_order_book(self.symbol, shift.OrderBookType.GLOBAL_ASK, self.ODBK_range)
            # assert Ask_ls, f'getOrderBook: return empty list: {self.symbol}-ASK-{self.ODBK_range}'

            Bid_ls = self.trader.get_order_book(self.symbol, shift.OrderBookType.GLOBAL_BID, self.ODBK_range)
            # assert Bid_ls, f'getOrderBook: return empty list: {self.symbol}-BID-{self.ODBK_range}'

            orders = OrderBook(Ask_ls, Bid_ls, last_price)

            self.mutex.acquire()
            # print(88)
            self.table.insertData(orders)
            # print(tmp)
            self.mutex.release()

            time.sleep(self.timeInterval)
        print('Data Thread stopped.')


    def step(self, action):
        premium = np.round(action[0], 2)
        print(f'premium: {premium}')
        signBuy = 1 if self.isBuy else -1
        base_price = self._getClosePrice(self.remained_share)
        obj_price = base_price - signBuy * premium

        print(f'base price: {base_price}, obj price: {obj_price}')

        if self.remained_time > 0:
            orderType = shift.Order.LIMIT_BUY if self.isBuy else shift.Order.LIMIT_SELL
        else:
            orderType = shift.Order.MARKET_BUY if self.isBuy else shift.Order.MARKET_SELL

        order = shift.Order(orderType, self.symbol, self.remained_share, obj_price)
        self.trader.submit_order(order)
        print(f'submited: {order.symbol}, {order.type}, {order.price}, base: {base_price},  {order.size}, {order.id}, {order.timestamp}')
        if self.remained_time > 0:
            time.sleep(self.timeInterval)
        else:
            time.sleep(1)
        print(f'waiting list size : {len(self.trader.get_waiting_list())}')
        if self.trader.get_waiting_list_size() > 0:
            self._cancelAllOrder(order)

        exec_price, exec_share, cancel_size, ComOrReb = self._executedOrder(order.id, True)

        assert self.remained_share == exec_share + cancel_size, f'{self.remained_share} != {exec_share} + {cancel_size}'
        self.remained_share -= exec_share

        print(f'remain: {self.remained_share}, {cancel_size}, executed: {exec_share}, current: {self._getCurrentPosition()}')
        done = False
        reward = exec_share * signBuy * (self.basePrice - exec_price) * 100 + ComOrReb
        if self.remained_time == 0 or self._getCurrentPosition() - self.objPos == 0: done = True

        self.remained_time -= 1

        if self.isSave:
            tmp = self.tmp_obs.tolist()+[reward, orderType, self.isBuy,
                                        premium, obj_price,base_price, exec_share, done]
            # print('-------------', self.tmp_obs)
            self.df.loc[self.df_idx] = tmp
            self.df_idx += 1

        next_obs = self._get_obs()
        return next_obs, reward, done, dict()

    def _cancelAllOrder(self, order):
        if order.type == shift.Order.LIMIT_BUY or order.type == shift.Order.MARKET_BUY:
            order.type = shift.Order.CANCEL_BID
        elif order.type == shift.Order.LIMIT_SELL or order.type == shift.Order.MARKET_SELL:
            order.type = shift.Order.CANCEL_ASK
        else:
            raise TypeError

        tmp_con = 0
        self.trader.submit_order(order)
        print('Canceling order:', end='')

        while True:
            exec_size = 0
            cancel_size = 0
            for suborder in self.trader.get_executed_orders(order.id):
                if suborder.type in [shift.Order.LIMIT_SELL, shift.Order.LIMIT_BUY,
                                     shift.Order.MARKET_SELL, shift.Order.MARKET_BUY]:
                    exec_size += suborder.executed_size
                elif suborder.type in [shift.Order.CANCEL_ASK, shift.Order.CANCEL_BID]:
                    cancel_size += suborder.executed_size
            if cancel_size + exec_size == self.remained_share: break

            tmp_con += 1
            time.sleep(0.05)
            print(self.trader.get_waiting_list_size(), end='')
            if tmp_con > 1000:
                print(f'\n current order info: %6s\t%21s\t%7.2f\t\t%4d\t%36s\t%26s' %
                      (order.symbol, order.type, order.price, order.size, order.id, order.timestamp))
                print("Symbol\t\t\t\t\t Type\t  Price\t\tSize\tID\t\t\t\t\t\t\t\t\t\tTimestamp")
                for od in self.trader.get_waiting_list():
                    print("%6s\t%21s\t%7.2f\t\t%4d\t%36s\t%26s" %
                          (od.symbol, od.type, od.price, od.size, od.id, od.timestamp))
                self._executedOrder(order.id, verbose = True)
                raise TimeoutError(f'Waited for canceling order for {tmp_con * 0.05} seconds.')
        print(' done.')

    def _executedOrder(self, orderID, verbose = False):
        limit_size = 0
        market_size = 0
        cancel_size = 0
        tmp_sum_price = 0
        for suborder in self.trader.get_executed_orders(orderID):
            if suborder.type in [shift.Order.LIMIT_SELL, shift.Order.LIMIT_BUY]:
                limit_size += suborder.executed_size
                tmp_sum_price += suborder.executed_price * suborder.executed_size
            elif suborder.type in [shift.Order.MARKET_SELL, shift.Order.MARKET_BUY]:
                market_size += suborder.executed_size
                tmp_sum_price += suborder.executed_price * suborder.executed_size
            elif suborder.type in [shift.Order.CANCEL_ASK, shift.Order.CANCEL_BID]:
                cancel_size += suborder.executed_size
        exec_size = limit_size + market_size
        if exec_size == 0:
            avg_price = 0
        else:
            avg_price = tmp_sum_price/exec_size
        com = market_size * 100 * self.commission
        rebate = limit_size * 100 * self.rebate

        if verbose:
            print("Symbol\t\t\t\tType\t  Price\t\tSize\tExecuted\t\t Status\t\tTimestamp")
            for order in self.trader.get_executed_orders(orderID):
                print("%6s\t%16s\t%7.2f\t\t%4d\t\t%4d\t\t%23s\t\t%26s" %
                      (order.symbol, order.type, order.executed_price, order.size,
                       order.executed_size, order.status, order.timestamp))
            print(f'avg_price = {avg_price}, exec_size = {exec_size}, cancel_size = {cancel_size}')
        return avg_price, exec_size, cancel_size, rebate+com


    def _get_obs(self):
        # add a new feature: delta_base_price: isBuy * (currentClosePrice - originalClosePrice)
        signBuy = 1 if self.isBuy else -1
        tmp = (self._getClosePrice(self.remained_share) - self.basePrice) * signBuy
        self.tmp_obs = np.concatenate((self.compute(), np.array([signBuy, tmp, self.remained_share, self.remained_time])))
        return self.tmp_obs.copy()

    def _getClosePrice(self, share):
        return self.trader.get_close_price(self.symbol, self.isBuy, abs(share))

    def reset(self):
        print(f'Holding shares: {self.trader.get_portfolio_item(self.symbol).get_shares()}')
        print(f'Buying Power: {self.trader.get_portfolio_summary().get_total_bp()}')
        return self._get_obs()

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

    def compute(self):
        self.mutex.acquire()
        tabData = self.table.getData()
        self.mutex.release()
        feature = []
        if self.table.isFull():
            for ele in tabData:
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
                                              self.remained_share,
                                              self.commission)
                    feature += [bas, p, sp, li, mc]
                else:
                    feature += [bas, p, sp, li, np.nan]
        else:
            raise ValueError('Data table is not full.')
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
        share = self.trader.get_portfolio_item(self.symbol).get_shares()
        BP = self.trader.get_portfolio_summary().get_total_bp()
        waitingStep = 0
        small_order = 1
        while share != 0:
            position = int(share / 100)
            orderType = shift.Order.MARKET_BUY if position < 0 else shift.Order.MARKET_SELL

            if share < 0 and BP < abs(share) * self.trader.get_close_price(self.symbol, True, abs(position)):
                order = shift.Order(orderType, self.symbol, small_order)
                self.trader.submit_order(order)
                small_order *= 2
            else:
                order = shift.Order(orderType, self.symbol, abs(position))
                self.trader.submit_order(order)

            time.sleep(0.5)
            #print(trader.getPortfolioItem(symbol).getShares())
            #print(trader.getPortfolioSummary().getTotalBP())
            share = self.trader.get_portfolio_item(self.symbol).get_shares()
            waitingStep += 1
            assert  waitingStep < 40



    def _getCurrentPosition(self):
        return int(self.trader.get_portfolio_item(self.symbol).get_shares() / 100)

    def __del__(self):
        self.kill_thread()

#         self.trader.disconnect()


if __name__=='__main__':
    trader = shift.Trader("test004")
    trader.disconnect()
    trader.connect("initiator.cfg", "password")
    trader.subAllOrderBook()
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

    env = SHIFT_env(trader, 5, 5, 5, 'AAPL')
    # env.close_all()

    obj_share = -6
    env.set_objective(share= obj_share, remained_time= 1)
    # for  i in range(1000):
    #     print(f'time step {i}')
    #     for ele in env.table.getData():
    #         print(ele)
    #     print(env.compute())
    #     time.sleep(1)
    #     print()

    # for i in range(10000):
    #     print(env.trader.getBestPrice("AAPL").getAskPrice(), env._getClosePrice(5))
    #     time.sleep(0.5)
    time.sleep(10)
    for j in range(3,1000):
        total_reward = 0
        print(f'Set goal: {obj_share} shares, current: {trader.getPortfolioItem("AAPL").getShares() / 100}')
        env.set_objective(obj_share, 1)
        # obj_share = (obj_share % 6) + 1
        env.reset()

        for i in range(10):
            # action = np.random.uniform(0, 0.05, size=1)
            bp = trader.getBestPrice("AAPL")
            action = [bp.getAskPrice() - bp.getBidPrice()]
            obs, reward, done, _ = env.step(action)
            print(f'obs shape: {obs.shape}')
            print(reward)
            total_reward += reward
            print(done)

            if done:
                print(i)
                step = 0
                if obj_share < 0:
                    pass
                else:
                    obj_share = (obj_share % 6) + 1
                obj_share *= -1
                break

        env.save_to_csv(j)
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

