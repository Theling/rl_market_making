import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)

import time
import shift
import numpy as np
import pandas as pd
import tensorflow as tf
from ppo import PPO
from SHIFT_env_2 import SHIFT_env as Env
from mlp_stoch_policy_2 import Policy
from threading import Thread
import queue
from portfolio_value import *
import datetime


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
    if macd_old <= 0 and macd_new >= 0:
        sign = 1
    elif macd_old >= 0 and macd_new <= 0:
        sign = -1
    else:
        sign = 0
    # sign = 1 if np.random.normal()>0 else -1
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

isBenchmark = False

action_dim = 1
state_dim = 7
seed = 13
base_shares = 15
symbol = 'AAPL'
trader = shift.Trader('test007')
trader.connect("initiator.cfg", "password")
trader.subAllOrderBook()
np.random.seed(seed)

env = Env(trader = trader,
          t = 2,
          nTimeStep=1,
          ODBK_range=5,
          symbol=symbol)

sess = tf.Session()
tf.set_random_seed(seed)
policy = Policy(sess, state_dim, action_dim, 'ppo', hidden_units=(64, 64))

old_policy = Policy(sess, state_dim, action_dim, 'oldppo', hidden_units=(64, 64))

dspg = PPO(env=env,
                policy=policy,
                old_policy=old_policy,
                session=sess,
            restore_fd='2019-03-13_11-14-05',
            policy_learning_rate = 0.001,
            epoch_length = 500,
           c1 = 1,
           c2 = 0.5,
           lam = 0.95,
           gamma = 0.99,
           max_local_step = 50,
            )
dspg.load(itr=18)

to_start = 0
while to_start == 0:
    time.sleep(1)
    to_start = len(trader.getOrderBook(symbol, shift.OrderBookType.GLOBAL_ASK, 1))

mid_price = get_mid_price(trader, symbol)
macd_new, init_st_ema, init_lt_ema = MACD(trader,
                                          symbol,
                                          mid_price,
                                          mid_price,
                                          short_period=60,
                                          long_period=90)
time.sleep(1)

macd_new = round(macd_new, 2)
count = 1
q = queue.Queue()
rwd = pd.DataFrame([[0, datetime.datetime.now(), 0]], index=['row'], columns=['TimeStep', 'Time', 'Reward'])
port_value = pd.DataFrame([[0, datetime.datetime.now(), 1000000, 0]],
                          index=['row'],
                          columns=['TimeStep', 'Time', 'PortfolioValue', 'CurrentShares'])

pv_save_name = 'Portfolio Value time series.csv' if isBenchmark else 'Portfolio Value time series_Execution.csv'
rwd_save_name = 'Reward time series.csv' if isBenchmark else 'Reward time series_Execution.csv'

port_value.to_csv(pv_save_name)
rwd.to_csv(rwd_save_name)
while trader.isConnected():
    port_shares = trader.getPortfolioItem(symbol).getShares() / 100
    macd, init_st_ema, init_lt_ema = MACD(trader,
                                          symbol,
                                          init_st_ema,
                                          init_lt_ema,
                                          short_period=60,
                                          long_period=90)
    T = 0
    commision = 0
    if count % 12 == 0:
        # Benchmark
        if isBenchmark:
            macd_old = macd_new
            macd_new = macd
            sign = signal(macd_old, macd_new)
            target_shares = base_shares * sign
            print('----------------------------------------------------------------------------------')
            print(f'Current holding shares: {port_shares*100}')
            print(f'target shares for attempt {int(count / 12)}: {target_shares * 100}')
            print('----------------------------------------------------------------------------------')
            print(f'ma_old: {macd_old}')
            print(f'ma_new: {macd_new}')
            print()
            if target_shares != 0:
                shares = int(target_shares - port_shares)
                if shares != 0:
                    orderType = shift.Order.MARKET_BUY if shares > 0 else shift.Order.MARKET_SELL
                    order = shift.Order(orderType, symbol, abs(shares))
                    trader.submitOrder(order)
                    print(
                        f'submited: {order.symbol}, {order.type}, {env._getClosePrice(abs(shares))}, {order.size}, {order.id}, {order.timestamp}')
                else:
                    print(f"Already holding the target shares of {target_shares*100}, no need to trade")
                commision = -abs(shares)*0.3
            else:
                print('No signal, hold the current position')
                commision = 0

        # Execution Agent
        else:
            macd_old = macd_new
            macd_new = macd
            sign = signal(macd_old, macd_new)
            target_shares = base_shares * sign
            print('----------------------------------------------------------------------------------')
            print(f'target shares for attempt {int(count/12)}: {target_shares * 100}')
            print('----------------------------------------------------------------------------------')
            print(f'ma_old: {macd_old}')
            print(f'ma_new: {macd_new}')
            print()
            if target_shares != 0:
                shares = int(target_shares - port_shares)
                if shares != 0:
                    T = Thread(target=dspg_try, args=(shares, symbol, 4, target_shares, q))
                    T.start()
                else:
                    print(f"Already holding the target shares of {target_shares*100}, no need to trade")
            else:
                print("No signal, Hold the current position")

    port_value.loc['row'] = [count, datetime.datetime.now(), portfolioValue(trader), port_shares]
    port_value.to_csv(pv_save_name, mode='a', header=False)
    print(f'Time Step: {count}, Time: {port_value.loc["row"][1].time()}, Portfolio Value: {port_value.loc["row"][2]}')
    try:
        rwd.loc['row'] = [count, datetime.datetime.now(), q.get(block=False)]
        print(f'Reward: {rwd.loc["row"][-1]}')
    except queue.Empty:
        rwd.loc['row'] = [count, datetime.datetime.now(), commision]
    rwd.to_csv(rwd_save_name, mode='a', header=False)


    count += 1
    time.sleep(1)
    print()

    if count > 3600*4:
        if T != 0:
            T.join()
        try:
            env.close_all()
        except AssertionError:
            print('close position failed')
            pass

        print(f'portfolio shares: {trader.getPortfolioItem(symbol).getShares() / 100}')
        time.sleep(2)
        curr_pos = trader.getPortfolioItem(symbol).getShares() / 100
        print(f'portfolio shares: {curr_pos}')
        final_pv = trader.getPortfolioSummary().getTotalBP()
        port_value.loc['row'] = [count, datetime.datetime.now(), final_pv, curr_pos]
        port_value.to_csv(pv_save_name, mode='a', header=False)
        break

    # if trader.getLastTradeTime().time() > datetime.datetime(2018, 12, 17, 15, 30, 00).time():
    #     env.close_all()
    #     break

port_value = pd.DataFrame(port_value, columns=['TimeStep', 'Time', 'PortfolioValue', 'CurrentShares'])
rwd = pd.DataFrame(rwd, columns=['TimeStep', 'Time', 'Reward'])

#date = str(datetime.datetime.today())

if isBenchmark:
    port_value.to_csv('Portfolio Value time series_macd2.csv')
    rwd.to_csv('Reward time series_macd2.csv')
else:
    port_value.to_csv('Portfolio Value time series_Execution_macd2.csv')
    rwd.to_csv('Reward time series_Execution_macd2.csv')
print()
print('----------------------------------------------------------------------------------')
print(f'Final Portfolio Value: {final_pv}')
print('Times Series of Portfolio Value:')
print(port_value)
print('Time Series of the Reward:')
print(rwd)
print('----------------------------------------------------------------------------------')

trader.disconnect()





