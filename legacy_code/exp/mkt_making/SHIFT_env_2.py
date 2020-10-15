import shift
from threading import Thread, Lock
import pandas as pd
import numpy as np
import time
import os

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
                 t,
                 nTimeStep,
                 ODBK_range,
                 symbol,
                 commission = 0.003,
                 rebate = 0.002,
                 save_data = True):

        #         trader = shift.Trader("test001")
        #         assert trader.connect("initiator.cfg", "password"), f'Connection error'
        #         assert trader.subAllOrderBook(), f'subscription fail'

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
        self.table = CirList(nTimeStep)
        self._cols = ['BA_spead', 'last_traded_price', 'Smart_price', 'Liquidity_imb', 'market_cost',
                'remained_shares', 'remained_time', 'reward', 'order_type','is_buy', 'premium',
                'obj_price', 'base_price', 'executed', 'done']
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
        self.remained_time = None
        self.tmp_obs = [None]*7
        self.name = 'exec_one_asset'
        self.isSave = save_data

    def set_objective(self, share, remained_time):
        self.remained_share = abs(share)
        self.total_share = self.remained_share
        self.currentPos = self._getCurrentPosition()
        self.objPos = self.currentPos + share
        self.remained_time = remained_time
        self.isBuy = True if share> 0 else False

    @staticmethod
    def action_space():
        return 1

    @staticmethod
    def obs_space():
        return 7

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

    def _get_orderID(self):
        ID = []
        for i in self.trader.getSubmittedOrders():
            ID.append(i.id)
        return ID[-1]

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
        self.trader.submitOrder(order)
        print(f'submited: {order.symbol}, {order.type}, {order.price}, base: {base_price},  {order.size}, {order.id}, {order.timestamp}')
        time.sleep(self.timeInterval)

        print(f'waiting list size : {len(self.trader.getWaitingList())}')
        if self.trader.getWaitingListSize() > 0:
            self._cancelAllOrder(order)

        tmp_share = self.remained_share
        self.remained_share = self.total_share - abs(self._getCurrentPosition()-self.currentPos)
        exec_share = tmp_share - self.remained_share
        exec_price = self.trader.getOrder(self._get_orderID()).executed_price
        premium_act = abs(exec_price - self.init_MidPrice)
        print(f'remain: {self.remained_share}, executed: {exec_share}, current: {self._getCurrentPosition()}')
        done = False
        if self.remained_time > 0:
            if premium > 0:
                reward = exec_share * premium_act * 100 + exec_share * self.rebate * 100
            else:
                reward = exec_share* premium_act * 100 - exec_share * self.commission * 100
        else:
            reward = exec_share*0 - exec_share * 0.3
            done = True

        if self._getCurrentPosition() - self.objPos == 0:
            done = True
        self.remained_time -= 1


        if self.isSave:
            tmp = self.tmp_obs.tolist()+[reward, orderType, self.isBuy,
                                        premium_act, obj_price,base_price, exec_share, done]
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
        self.trader.submitOrder(order)
        print('Canceling order:', end='')
        while self.trader.getWaitingListSize() > 0:
            tmp_con += 1
            time.sleep(0.05)
            print(self.trader.getWaitingListSize(), end='')
            if tmp_con > 1000:
                print(f'\n current order info: %6s\t%21s\t%7.2f\t\t%4d\t%36s\t%26s' %
                      (order.symbol, order.type, order.price, order.size, order.id, order.timestamp))
                print("Symbol\t\t\t\t\t Type\t  Price\t\tSize\tID\t\t\t\t\t\t\t\t\t\tTimestamp")
                for od in self.trader.getWaitingList():
                    print("%6s\t%21s\t%7.2f\t\t%4d\t%36s\t%26s" %
                          (od.symbol, od.type, od.price, od.size, od.id, od.timestamp))
                raise TimeoutError(f'Waited for canceling order for {tmp_con * 0.05} seconds.')
        print(' done.')

    def _get_obs(self):
        self.tmp_obs = np.concatenate((self.compute(), np.array([self.remained_share, self.remained_time])))
        return self.tmp_obs.copy()
    def _getClosePrice(self, share):
        return self.trader.getClosePrice(self.symbol, self.isBuy, abs(share))

    def reset(self):
        print(f'Holding shares: {self.trader.getPortfolioItem(self.symbol).getShares()}')
        print(f'Buying Power: {self.trader.getPortfolioSummary().getTotalBP()}')
        ask_price = self.trader.getOrderBook(self.symbol, shift.OrderBookType.GLOBAL_ASK, 1)[0].price
        bid_price = self.trader.getOrderBook(self.symbol, shift.OrderBookType.GLOBAL_BID, 1)[0].price
        self.init_MidPrice = (ask_price + bid_price) /2
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
                                              self.remained_share,
                                              self.commission)
                    feature += [bas, p, sp, li, mc]
                else:
                    feature += [bas, p, sp, li, np.nan]
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


if __name__=='__main__':
    trader = shift.Trader("test007")
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

    env = SHIFT_env(trader, 2, 1, 5, 'AAPL', 0.003)
    # env.close_all()

    obj_share = -1
    env.set_objective(share= obj_share, remained_time= 5)
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
    time.sleep(2)
    for j in range(1000):
        total_reward = 0
        print(f'Set goal: {obj_share} shares, current: {trader.getPortfolioItem("AAPL").getShares() / 100}')
        env.set_objective(obj_share, 5)

        if obj_share < 0:
            obj_share *= -1
        else:
            obj_share = (obj_share % 9) + 1
            obj_share *= -1



        env.reset()

        for i in range(100):
            action = np.random.uniform(0, 0.05, size=1)

            obs, reward, done, _ = env.step(action)
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

