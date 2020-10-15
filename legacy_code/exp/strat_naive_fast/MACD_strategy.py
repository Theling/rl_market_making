import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)

import time
import numpy as np
import pandas as pd
import tensorflow as tf
from ppo import PPO
from one_asset_env import One_Asset_Env as Env
from mlp_stoch_policy2 import Policy
from Backtesting.Engine.engine import Market
from Backtesting.DB_interface.DataReader import dataReader


def MACD(init_st_ema, init_lt_ema, short_period, long_period):
    st_ita = 2 / (short_period+1)
    lt_ita = 2 / (long_period+1)
    price = float(dr.get('State', 1).iloc[:, -5]*100)
    st_ema = price * st_ita + init_st_ema * (1 - st_ita)
    lt_ema = price * lt_ita + init_lt_ema * (1 - lt_ita)
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


action_dim = 1
state_dim = 10
seed = 16
base_shares = 10
symbol = 'AAPL'
start=100
end=2000

env = Env(action_spec = 1,
            obs_spec=state_dim,
            symbol='AAPL',
            nTimeStep=state_dim-1,
            start_date=start,
            end_date=end,
            n_share=1000)

sess = tf.Session()
tf.set_random_seed(seed)
policy = Policy(sess, state_dim, action_dim, 'ppo', hidden_units=(64, 64))

old_policy = Policy(sess, state_dim, action_dim, 'oldppo', hidden_units=(64, 64))

dspg = PPO(env=env,
                policy=policy,
                old_policy=old_policy,
                session=sess,
            restore_fd='2019-04-18_10-39-14',
            policy_learning_rate = 0.001,
            epoch_length = 500,
           c1 = 1,
           c2 = 0.5,
           lam = 0.95,
           gamma = 0.99,
           max_local_step = 2000,
           batch_size=100
            )
r3 = dspg.run(15, max_epoch = 1)
r1, _ = env.mkt.stats.trading_log()

mkt = Market(start_date=start,
            end_date=end,
             cash=1000000,
             rf=0,
            commision=0.003)
dr = dataReader()
dr.set_date(start)
price = float(dr.get('State', 1).iloc[:, -5]*100)
macd_new, init_st_ema, init_lt_ema = MACD(price,
                                          price,
                                          short_period=60,
                                          long_period=90)


macd_new = round(macd_new, 2)
count = 1
port_value = [[0, 1000000, 0]]
tomo = 1
cur_pos = 0
while tomo != 0:
    port_shares = cur_pos
    macd, init_st_ema, init_lt_ema = MACD(init_st_ema,
                                          init_lt_ema,
                                          short_period=60,
                                          long_period=90)

    macd_old = macd_new
    macd_new = macd
    sign = signal(macd_old, macd_new)
    target_shares = base_shares * sign
    print('----------------------------------------------------------------------------------')
    print(f'Current holding shares: {port_shares * 100}')
    print(f'target shares: {target_shares * 100}')
    print('----------------------------------------------------------------------------------')
    print(f'ma_old: {macd_old}')
    print(f'ma_new: {macd_new}')
    print()
    if target_shares != 0:
        shares = int(target_shares - port_shares)
        if shares != 0:
            mkt.adjustPosition(pd.DataFrame([target_shares],index=['AAPL']))
            print(f'Real action: {shares*100}')
            cur_pos = target_shares
        else:
            print(f"Already holding the target shares of {target_shares*100}, no need to trade")
    else:
        print("No signal, Hold the current position")

    tomo = mkt.next()
    dr.next()

    port_value.append([count, mkt.portfolio_account['portfolio_value'], port_shares])
    print(f'Time Step: {port_value[-1][0]}, Portfolio Value: {port_value[-1][1]}')
    # rwd.append([count, reward])
    # print(f'Reward: {rwd[-1][1]}')


    count += 1
    print()

r2, _ = mkt.stats.trading_log()
port_value = pd.DataFrame(port_value, columns=['TimeStep', 'PortfolioValue', 'CurrentShares'])
# rwd = pd.DataFrame(rwd, columns=['TimeStep', 'Reward'])


#port_value.to_csv('Portfolio Value time series_naive_fast.csv')
# rwd.to_csv('Reward time series_Execution.csv')
print()
print('----------------------------------------------------------------------------------')
print(f'Final Portfolio Value: {port_value.iloc[-1, 1]}')
print('Times Series of Portfolio Value:')
print(port_value)
# print('Time Series of the Reward:')
# print(rwd)
print('----------------------------------------------------------------------------------')





