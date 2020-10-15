import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)

import numpy as np
import pandas as pd
import tensorflow as tf
from ppo import PPO
from one_asset_env import One_Asset_Env as Env
from RRL_policy import Policy
from Backtesting.Engine.engine import Market
from Backtesting.DB_interface.DataReader import dataReader
import matplotlib.pyplot as plt


def MACD(init_st_ema, init_lt_ema, short_period, long_period):
    st_ita = 2 / (short_period+1)
    lt_ita = 2 / (long_period+1)
    price = float(dr.get('State6', 1).iloc[:, -5]*100)
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
state_dim = 6
seed = 23
base_shares = 1000
symbol = 'PFE'
start=100
end=1900

env = Env(action_spec = action_dim,
            obs_spec=state_dim,
            symbol=symbol,
            nTimeStep=30,
            start_date=start,
            end_date=end,
            n_share=1000,
          commision = 0.003
          )

sess = tf.Session()
tf.set_random_seed(seed)
policy = Policy(sess, state_dim, action_dim, 'ppo',
                num_units=64,
                time_len=30)

old_policy = Policy(sess, state_dim, action_dim, 'oldppo',
                    num_units=64,
                    time_len=30)

dspg = PPO(env=env,
                policy=policy,
                old_policy=old_policy,
                session=sess,
            restore_fd='2019-04-21_16-01-03',
            policy_learning_rate = 0.001,
            epoch_length = 500,
           c1 = 1,
           c2 = 0.5,
           lam = 0.95,
           gamma = 0.99,
           max_local_step = 2000,
           batch_size=100
            )
r3 = dspg.run(1, max_epoch = 1)
r1, _ = env.mkt.stats.trading_log()
env.mkt.stats.compute(1800)

mkt = Market(start_date=start,
            end_date=end,
             cash=1000000,
             rf=0,
            commision=0.003
             )
dr = dataReader()
dr.set_date(start)
price = float(dr.get('State6', 1).iloc[:, -5]*100)
macd_new, init_st_ema, init_lt_ema = MACD(price,
                                          price,
                                          short_period=15,
                                          long_period=30)


macd_new = round(macd_new, 2)
count = 1
port_value = [[0, 1000000, 0]]
tomo = 1
cur_pos = 0
while tomo != 0:
    port_shares = cur_pos
    macd, init_st_ema, init_lt_ema = MACD(init_st_ema,
                                          init_lt_ema,
                                          short_period=15,
                                          long_period=30)

    macd_old = macd_new
    macd_new = macd
    sign = signal(macd_old, macd_new)
    target_shares = base_shares * sign
    print('----------------------------------------------------------------------------------')
    print(f'Current holding shares: {port_shares}')
    print(f'target shares: {target_shares}')
    print('----------------------------------------------------------------------------------')
    print(f'ma_old: {macd_old}')
    print(f'ma_new: {macd_new}')
    print()
    if target_shares != 0:
        shares = int(target_shares - port_shares)
        if shares != 0:
            mkt.adjustPosition(pd.DataFrame([target_shares],index=[symbol]))
            print(f'Real action: {shares}')
            cur_pos = target_shares
        else:
            print(f"Already holding the target shares of {target_shares}, no need to trade")
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

plt.subplot(2, 1, 1)
plt.plot(range(r1.index.size), r1, label = 'strategy')
plt.plot(range(r2.index.size), r2, label = 'benchmark')
# plt.ylim(999000, 1002000)
# plt.axvline(x=2000, color='r', linestyle="--")
plt.legend()
plt.title('Portfolio Value')

if action_dim == 1:
    a = np.array(r3)
    a = np.sign(a)
    a = np.reshape(a, newshape=(-1,))
    plt.subplot(2, 1, 2)
    plt.plot(range(len(a)), a, linewidth=0.1, color='dodgerblue')
    plt.title('Action')
elif action_dim == 3:
    a = []
    for i in range(0, len(r3)):
        a_max = r3[i].argmax()
        if a_max == 0:
            realAction = -1
        elif a_max == 1:
            realAction = 0
        else:
            realAction = 1
        a.append(realAction)
    plt.subplot(2, 1, 2)
    plt.plot(range(len(a)), a, linewidth=0.1, color='dodgerblue')
    plt.title('Action')



plt.show()



