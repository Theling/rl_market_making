import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)

import time
import shift
import numpy as np
import tensorflow as tf
from ppo import PPO
from SHIFT_env import SHIFT_env as Env
from mlp_stoch_policy import Policy
from threading import Thread
import queue
from Portfolio_value import *
import datetime


# import pandas as pd
# pd.core.common.is_list_like = pd.api.types.is_list_like
# import pandas_datareader.data as web
#
# stocks = ['FB']
# def get_stock(stock, start, end):
#      return web.DataReader(stock, 'yahoo', start, end)['Adj Close']
# px = pd.DataFrame({n: get_stock(n, '1/1/2016', '12/31/2016') for n in stocks})
# px
# px['26_ema'] = px['FB'].ewm(span=26).mean()
# px['12_ema'] = px['FB'].ewm(span=12).mean()
# px

def get_mid_price(trader, symbol):
    ask_price = trader.getOrderBook(symbol, shift.OrderBookType.GLOBAL_ASK, 1)[0].price
    bid_price = trader.getOrderBook(symbol, shift.OrderBookType.GLOBAL_BID, 1)[0].price
    mid_price = (ask_price + bid_price) / 2
    return mid_price


def MACD(trader, symbol, init_st_ema, init_lt_ema, short_period, long_period):
    st_ita = 2 / (short_period+1)
    lt_ita = 2 / (long_period+1)
    mid_price = get_mid_price(trader, symbol)
    st_ema = mid_price * st_ita + init_st_ema * (1 - st_ita)
    lt_ema = mid_price * lt_ita + init_lt_ema * (1 - lt_ita)
    macd = st_ema - lt_ema
    return macd, st_ema, lt_ema

def signal(macd_old, macd_new):
    if macd_old < 0 and macd_new > 0:
        sign = True
    else:
        sign = False
    return sign

def dspg_try(shares, symbol, time, target_shares, q):
    ret = 0
    try:
        ret = dspg.execute(shares, time)
    except TimeoutError as info:
        trader.cancelAllPendingOrders()
        print(info)
    delta_port_shares = trader.getPortfolioItem(symbol).getShares() / 100 - target_shares
    orderType = shift.Order.MARKET_BUY if shares > 0 else shift.Order.MARKET_SELL
    remaining_share = int(np.abs(delta_port_shares))
    if remaining_share != 0:
        order = shift.Order(orderType, symbol, remaining_share)
        trader.submitOrder(order)
    q.put(ret)



action_dim = 1
state_dim = 7
seed = 13
symbol = 'TRV'
trader = shift.Trader('test004')
trader.connect("initiator.cfg", "password")
trader.subAllOrderBook()

env = Env(trader = trader,
          t = 2,
          nTimeStep=1,
          ODBK_range=5,
          symbol='TRV')

sess = tf.Session()
tf.set_random_seed(seed)
policy = Policy(sess, state_dim, action_dim, 'ppo', hidden_units=(64, 64))

old_policy = Policy(sess, state_dim, action_dim, 'oldppo', hidden_units=(64, 64))

dspg = PPO(env=env,
                policy=policy,
                old_policy=old_policy,
                session=sess,
            restore_fd='2019-03-07_22-49-34',
            policy_learning_rate = 0.001,
            epoch_length = 500,
           c1 = 1,
           c2 = 0.5,
           lam = 0.95,
           gamma = 0.99,
           max_local_step = 50,
            )
dspg.load(itr=61)


mid_price = get_mid_price(trader, symbol)
macd_new, init_st_ema, init_lt_ema = MACD(trader,
                                          symbol,
                                          mid_price,
                                          mid_price,
                                          short_period=2.5*60,
                                          long_period=5*60)
time.sleep(1)

count = 1

q = queue.Queue()
while trader.isConnected():
    port_shares = trader.getPortfolioItem(symbol).getShares() / 100
    macd, init_st_ema, init_lt_ema = MACD(trader,
                                          symbol,
                                          init_st_ema,
                                          init_lt_ema,
                                          short_period=2.5*60,
                                          long_period=5*60)
    # # Benchmark
    # if count % 10 == 0:
    #     macd_old = macd_new
    #     macd_new = macd
    #     sign = signal(macd_old, macd_new)
    #     target_shares = 5 if sign else -5
    #     shares = int(target_shares - port_shares)
    #     if shares != 0:
    #         orderType = shift.Order.MARKET_BUY if shares > 0 else shift.Order.MARKET_SELL
    #         order = shift.Order(orderType, symbol, shares)
    #         trader.submitOrder(order)
    #     else:
    #         print(f"Already holding the target shares of {target_shares*100}, no need to trade")

    # Execution Agent
    if count % 10 == 0:
        macd_old = macd_new
        macd_new = macd
        sign = signal(macd_old, macd_new)
        target_shares = 5 if sign else -5
        print('----------------------------------------------------------------------------------')
        print(f'target shares for attempt {int(count/10)}: {target_shares * 100}')
        print('----------------------------------------------------------------------------------')
        shares = int(target_shares - port_shares)
        if shares != 0:
            T = Thread(target=dspg_try, args=(shares, symbol, 4, target_shares, q))
            T.start()
        else:
            print(f"Already holding the target shares of {target_shares*100}, no need to trade")


    count += 1
    time.sleep(1)
    aaaaaa.append(q.get(block=False))
    if count > 50:
        T.join()
        print(f'portfolio: {trader.getPortfolioItem(symbol).getShares() / 100}')
        env.close_all()
        print(f'portfolio: {trader.getPortfolioItem(symbol).getShares() / 100}')
        time.sleep(2)
        break

    # if trader.getLastTradeTime().time() > datetime.datetime(2018, 12, 17, 15, 30, 00).time():
    #     env.close_all()
    #     break
print(f'BuyingPower: {trader.getPortfolioSummary().getTotalBP()}')

trader.disconnect()






