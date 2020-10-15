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


def get_mid_price(trader, symbol):
    ask_price = trader.getOrderBook(symbol, shift.OrderBookType.GLOBAL_ASK, 1)[0].price
    bid_price = trader.getOrderBook(symbol, shift.OrderBookType.GLOBAL_BID, 1)[0].price
    mid_price = (ask_price + bid_price) / 2
    return mid_price


def strategy(trader, symbol, nTime, shares=5):
    d = []
    for i in range(nTime):
        mid_price = get_mid_price(trader, symbol)
        d += [mid_price]
        time.sleep(1)
    avg_price = np.average(d)
    next_price = get_mid_price(trader, symbol)
    if next_price > avg_price:
        return -shares
    else:
        return shares

action_dim = 1
state_dim = 7
seed = 13
nTime = 60
symbol = 'AAPL'
trader = shift.Trader('test007')
trader.connect("initiator.cfg", "password")
trader.subAllOrderBook()

env = Env(trader = trader,
          t = 2,
          nTimeStep=1,
          ODBK_range=5,
          symbol='AAPL')

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

count = 1
while trader.isConnected():
    port_shares = trader.getPortfolioItem(symbol).getShares()/100
    shares = strategy(trader, symbol, nTime, shares=5)
    print('----------------------------------------------------------------------------------')
    print(f'target shares for round {count}: {shares*100}')
    print('----------------------------------------------------------------------------------')
    try:
        dspg.execute(int(shares - port_shares), time=4)
    except TimeoutError as info:
        trader.cancelAllPendingOrders()
        print(info)
    delta_port_shares = trader.getPortfolioItem(symbol).getShares()/100 - port_shares
    orderType = shift.Order.MARKET_BUY if shares > 0 else shift.Order.MARKET_SELL
    remaining_share = int(np.abs(delta_port_shares))
    if remaining_share != 0:
        order = shift.Order(orderType, symbol, remaining_share)
        trader.submitOrder(order)

    count += 1
    if count > 5:
        env.close_all()
        trader.disconnect()
        break






