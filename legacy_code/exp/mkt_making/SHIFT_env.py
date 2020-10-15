import shift
import pandas as pd
import numpy as np
import time
import os
from threading import Thread, Lock
from queue import Queue
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
                'remained_shares_buy', 'remained_shares_sell','remained_time', 'reward', 'order_type1', 'order_type2',
                      'market_order_type', 'market_price','premium1',
                'premium2', 'obj_price1', 'obj_price2', 'base_price1', 'base_price2', 'executed1', 'executed2', 'net_mkt_val', 'done',
                      'pv']
        self.df = pd.DataFrame(columns=self._cols)
        self.df_idx = 0

        print('Waiting for connection', end='')
        for _ in range(5):
            time.sleep(1)
            print('.', end='')
        print()

        self.thread_alive = True
        self.dataThread.start()

        self.cum_rebate = 0
        self.remained_share = None
        self.total_share = None
        self.currentPos = None
        self.objPos = None
        self.isBuy = True
        self.remained_time = None
        self.tmp_obs = [None]*8
        self.name = 'mkt_making'
        self.isSave = save_data
        self.t_mutex = Lock()
        self.pos_at_begining = None

    def set_objective(self, share, remained_time):
        self.remained_share1 = abs(share)
        self.remained_share2 = abs(share)
        self.total_share = self.remained_share
        self.currentPos = self._getCurrentPosition()
        self.objPos = self.currentPos + share
        self.remained_time = remained_time
        self.isBuy = True if share > 0 else False

    @staticmethod
    def action_space():
        return 2

    @staticmethod
    def obs_space():
        return 8

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

    def _get_mid_price(self, trader, symbol):
        ask_price = trader.getOrderBook(symbol, shift.OrderBookType.GLOBAL_ASK, 1)[0].price
        bid_price = trader.getOrderBook(symbol, shift.OrderBookType.GLOBAL_BID, 1)[0].price
        mid_price = (ask_price + bid_price) / 2
        return mid_price

    def _submitOrder(self, premium, remained_share, flag, orderType):

        signBuy = 1 if flag else -1
        base_price = self._get_mid_price(self.trader, self.symbol)
        if orderType in [shift.Order.LIMIT_SELL, shift.Order.LIMIT_BUY]:
            obj_price = base_price - signBuy*premium
        else:
            obj_price = base_price
        if remained_share <= 0:
            print('Pass order with 0 shares')
            return 0, base_price, obj_price
        print(f'base price: {base_price}, obj price: {obj_price}')
        order = shift.Order(orderType, self.symbol, remained_share, obj_price)

        self.trader.submitOrder(order)
        print(f'submitted: {order.symbol}, {order.type}, {order.price}, base: {base_price},  {order.size}, {order.id}, {order.timestamp}')

        order_id = order.id
        return order_id, base_price, order.price


    def _stepOneOrder(self, premium, isBuy, rShares, rTime, q):
        if rShares <= 0:
            print(f'[side: {isBuy}] remaining share: {rShares}, pass')
            q.put((None, None, 0, 0, 0))
            return

        orderType = shift.Order.LIMIT_BUY if isBuy else shift.Order.LIMIT_SELL

        print(f'[side: {isBuy}] premium: {premium}')

        order_id, base_price, obj_price = self._submitOrder(premium, rShares, isBuy, orderType)

        time.sleep(self.timeInterval)

        o1 = self.trader.getOrder(order_id)
        count = 0
        while o1.status == shift.Order.PENDING_NEW:
            time.sleep(0.1)
            o1 = self.trader.getOrder(order_id)
            count += 1
            if count > 600:
                raise TimeoutError(f'order is in pending-new: \n{o1.type}, {o1.size}, {o1.status};')
        print(f'[side: {isBuy}] status: {o1.status}')
        print(f'waiting list size : {self.trader.getWaitingListSize()}')
        num1, exec_share = self._cancelOrder(order_id)
        print(f'[side: {isBuy}] cancel result {num1}')
        q.put((orderType, order_id, base_price, obj_price, exec_share))
        print(f'one side step finished: {orderType}')

    def step(self, action):

        origin_pos = self._getCurrentPosition()
        premium1 = np.round(action[0], 2)
        premium2 = np.round(action[1], 2)

        q_Buy = Queue()
        q_Sell = Queue()

        t_Buy = Thread(target=self._stepOneOrder, args= (premium1,
                                                         True,
                                                         self.remained_share1,
                                                         self.remained_time,
                                                         q_Buy))
        t_Sell = Thread(target=self._stepOneOrder, args= (premium2,
                                                          False,
                                                          self.remained_share2,
                                                          self.remained_time,
                                                          q_Sell))


        t_Buy.start()
        t_Sell.start()

        t_Buy.join()
        t_Sell.join()


        orderType1, order_id1, base_price1, obj_price1, exec_share1 = q_Buy.get()
        orderType2, order_id2, base_price2, obj_price2, exec_share2 = q_Sell.get()
        cur_pos = self._getCurrentPosition()
        if origin_pos + exec_share1-exec_share2 == cur_pos:
            pass
        else:
            print(f'position not match (adjust remaining shares): {origin_pos}+{exec_share1}-{exec_share2} != {cur_pos}')
            if cur_pos - origin_pos > 0:
                exec_share1 = cur_pos - origin_pos
                exec_share2 = 0
            else:
                exec_share1 = 0
                exec_share2 = origin_pos - cur_pos

        net_mkt_value = -(exec_share1 * obj_price1 - exec_share2 * obj_price2) * 100
        print(f'Buy: {exec_share1}, {obj_price1} = {exec_share1 * obj_price1}')
        print(f'Sell: {exec_share2}, {obj_price2} = {exec_share2 * obj_price2}')
        self.remained_share1 = self.remained_share1 - exec_share1
        self.remained_share2 = self.remained_share2 - exec_share2

        print(f'BUY: remain: {self.remained_share1}, executed: {exec_share1}, current: {self._getCurrentPosition()}')
        print(f'SELL: remain: {self.remained_share2}, executed: {exec_share2}, current: {self._getCurrentPosition()}')
        done = False
        if self.remained_time > 0:
            if premium1 > 0:
                reward1 = exec_share1 * premium1 * 100 + exec_share1 * self.rebate * 100
            else:
                reward1 = exec_share1 * premium1 * 100 - exec_share1 * self.commission * 100
            if premium2 > 0:
                reward2 = exec_share2 * premium2 * 100 + exec_share2 * self.rebate * 100
            else:
                reward2 = exec_share2 * premium2 * 100 - exec_share2 * self.commission * 100
        else:
            reward1 = exec_share1*0 - exec_share1 * self.commission * 100
            reward2 = exec_share2*0 - exec_share2 * self.commission * 100
            done = True
        reward = reward1 + reward2

        if self.remained_share1 == 0 and self.remained_share2 == 0:
            done = True
        self.remained_time -= 1

        if self.remained_time == 0:
            done = True

        tmp_type = 0
        mkt_price = 0
        if done:
            try:
                assert self._getCurrentPosition() == self.pos_at_begining
            except AssertionError:
                deltashares = self.pos_at_begining - self._getCurrentPosition()
                tmp_type = shift.Order.MARKET_BUY if deltashares > 0 else shift.Order.MARKET_SELL
                fo = shift.Order(tmp_type, self.symbol, abs(deltashares))
                self.trader.submitOrder(fo)
                mkt_price = self.trader.getOrder(fo.id).executed_price
                self.cum_rebate -= abs(deltashares)*self.commission*100
                while mkt_price == 0:
                    time.sleep(1)
                    mkt_price = self.trader.getOrder(fo.id).executed_price
                print(f'submitted: {fo.symbol}, {fo.type}, {fo.size},  {mkt_price}, {fo.id}, {fo.timestamp}')
                net_mkt_value += mkt_price * fo.size * 100 * (1 if deltashares > 0 else -1) * -1


        if self.remained_time != 0:
            self.cum_rebate += (exec_share1 + exec_share2) * self.rebate * 100
        pv = portfolioValue(self.trader) + self.cum_rebate
        print(f'pv: {pv}')

        if self.isSave:
            tmp = self.tmp_obs.tolist()+[reward, orderType1, orderType2, tmp_type, mkt_price,
                                        premium1, premium2, obj_price1,
                                        obj_price2, base_price1, base_price2,
                                        exec_share1, exec_share2, net_mkt_value,  done, pv]
            # print('-------------', self.tmp_obs)
            self.df.loc[self.df_idx] = tmp
            self.df_idx += 1

        next_obs = self._get_obs()

        return next_obs, reward, done, {'net_mkt_val': net_mkt_value}

    def _cancelOrder(self, order_id):
        submited_order = self.trader.getOrder(order_id)
        if submited_order.status == shift.Order.FILLED:
            return 2, submited_order.size
        if submited_order.status in [shift.Order.NEW, shift.Order.PARTIALLY_FILLED]:
            s = submited_order.executed_size
            self.trader.submitCancellation(submited_order)
            print(
                f'Submited Cancellation: {submited_order.symbol}, {submited_order.type}, {submited_order.price}, {submited_order.id}')
            count = 0
            while count < 3000:
                if submited_order.status in [shift.Order.CANCELED, shift.Order.FILLED]:
                    print(f' {submited_order.executed_size}, {self.trader.getWaitingListSize()}')
                    if submited_order.status == shift.Order.CANCELED:
                        return 1, s
                    else:
                        return 2, submited_order.size
                else:
                    s = submited_order.executed_size
                time.sleep(0.01)
                submited_order = self.trader.getOrder(order_id)
                if self.trader.getWaitingListSize() == 0:
                    if submited_order.status in [shift.Order.PENDING_CANCEL, shift.Order.REJECTED]:
                        return 4, s
                count += 1
                print(self.trader.getWaitingListSize(), end='')
                if self.trader.getWaitingListSize()==0:
                    print(submited_order.status)

            print(submited_order.status)
            return 3, s

        else:
            print(f'abnormal order: {submited_order.type}, {submited_order.status}, {submited_order.size}')
            return 0, 0


    def _get_obs(self):
        self.tmp_obs = np.concatenate((self.compute(), np.array([self.remained_share1, self.remained_share2, self.remained_time])))
        return self.tmp_obs.copy()
    def _getClosePrice(self, share, ordertype):
        return self.trader.getClosePrice(self.symbol, ordertype, abs(share))

    def reset(self):
        print(f'Holding shares: {self.trader.getPortfolioItem(self.symbol).getShares()}')
        print(f'Buying Power: {self.trader.getPortfolioSummary().getTotalBP()}')
        self.pos_at_begining = self._getCurrentPosition()
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
                                              self.remained_share1,
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
    trader = shift.Trader("test002")
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

    env = SHIFT_env(trader, 10, 1, 5, 'CSCO', 0.003)
    # env.close_all()

    obj_share = 1
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
    time.sleep(5)
    for j in range(1000):
        total_reward = 0
        total_val = 0
        print(f'Set goal: {obj_share} shares, current: {trader.getPortfolioItem("TRV").getShares() / 100}')
        env.set_objective(obj_share, 1)

        # if obj_share < 0:
        #     obj_share *= -1
        # else:
        #     obj_share = (obj_share % 9) + 1
        #     #obj_share *= -1

        env.reset()

        for i in range(100):
            # action = np.random.uniform(0.01, 0.05, size=2)
            action = np.array([0.01,0.01])
            obs, reward, done, di = env.step(action)
            print(obs)
            print(reward)
            total_reward += reward
            total_val += di['net_mkt_val']
            print(done)

            if done:
                print(f'finished')
                break
        env.save_to_csv(0)
        print(f'Total reward is {total_reward}')
        print(f'Total val is {total_val}')
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

