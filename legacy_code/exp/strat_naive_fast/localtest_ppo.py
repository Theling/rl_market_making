import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)
from Backtesting.Engine.engine import Market
import pandas as pd
import matplotlib.pyplot as plt
#import shift
from ppo import PPO
from one_asset_env import One_Asset_Env
# from SHIFT_env import MarketOrder_exec
from mlp_stoch_policy2 import Policy


import numpy as np
import tensorflow as tf
import time





action_dim = 1
state_dim = 11
symbol = 'AAPL'
start = 100
end = 2000
n_share = 1000
seed = 13

# trader = shift.Trader("test002")
# trader.disconnect()
# trader.connect("initiator.cfg", "password")
# trader.subAllOrderBook()



env = One_Asset_Env(action_spec=action_dim,
                    obs_spec=state_dim,
                    symbol='AAPL',
                    nTimeStep=10,
                    start_date=start,
                    end_date=end,
                    n_share=1000,
                    commision = 0.003)


sess = tf.Session()
tf.set_random_seed(seed)
policy = Policy(sess, state_dim, action_dim, 'ppo', hidden_units=(64, 64))

old_policy = Policy(sess, state_dim, action_dim, 'oldppo', hidden_units=(64, 64))


dspg = PPO(env=env,
                policy=policy,
                old_policy=old_policy,
                session=sess,
            restore_fd='2019-04-18_18-49-47',
            policy_learning_rate = 0.001,
           c1 = 1,
           c2 = 0.5,
           lam = 0.95,
           gamma = 0.99,
           max_local_step = 1000,
           batch_size = 100
            )
r3 = dspg.run(15, max_epoch = 1)
r1, _ = env.mkt.stats.trading_log()

mkt = Market(start_date=start,
                 end_date=end,
                 cash=1000000,
             rf = 0,
             commision=0.003)
position = pd.DataFrame([n_share],index=[symbol])
mkt.adjustPosition(position)
while True:
    ret = mkt.next()
    if ret == 0:
        break

r2, _ = mkt.stats.trading_log()
plt.subplot(2, 1, 1)
plt.plot(range(r1.index.size), r1, label = 'strategy')
plt.plot(range(r2.index.size), r2, label = 'benchmark')
plt.ylim(999000, 1002000)
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

len(a)
t=list(a)
t.pop()
t.pop(0)

stk = pd.read_csv('/mnt/c/Users/liuyu/PycharmProjects/FE800/RL/exp/strat_naive_fast/Backtesting/DB_interface/data/Close.csv')

stkI = stk.iloc[100:2000,:]
stkO = stk.iloc[2000:4000,:]
stkI['Action'] = t
stkO['Action'] = t
stkI['up'] = stkI.Action.diff()>0
stkI['down'] = stkI.Action.diff()<0

stkO['up'] = stkO.Action.diff()>0
stkO['down'] = stkO.Action.diff()<0


fig = plt.figure()
ax=fig.add_subplot(111)
ax.plot(stkI.index, stkI.AAPL)
tmp = stkI[stkI['up']==True]
ax.plot(tmp.index, tmp.AAPL, '^', markersize=4,color='green',label = 'buy')
tmp = stkI[stkI['down']==True]
ax.plot(tmp.index, tmp.AAPL, 'v', markersize=4,color='red',label = 'sell')
ax.set_ylim(146.8, 150, 0.2)
ax.legend()
plt.show()

fig = plt.figure()
ax=fig.add_subplot(111)
ax.plot(stkO.index, stkO.AAPL)
tmp = stkO[stkO['up']==True]
ax.plot(tmp.index, tmp.AAPL, '^', markersize=4,color='green',label = 'buy')
tmp = stkO[stkO['down']==True]
ax.plot(tmp.index, tmp.AAPL, 'v', markersize=4,color='red',label = 'sell')
ax.set_ylim(146.8, 150, 0.2)
ax.legend()
plt.show()



# ax1 = si.add_subplot(111, ylabel = 'Price in $')
# # Plot the buy signals
# ax1.plot(stkI.loc[stkI.Action == 1.0].index,
#          stkI[stkI.Action == 1.0],
#          '^', markersize=0.1, color='green')
#
# # Plot the sell signals
# ax1.plot(stkI.loc[stkI.Action == -1.0].index,
#          stkI[stkI.Action == -1.0],
#          'v', markersize=0.1, color='red')
#
# # Show the plot
# plt.show()
#
# # Plot the buy signals
# ax1.plot(signals_fb.loc[signals_fb.positions == 2.0].index,
#          signals_fb.short_avg[signals_fb.positions == 2.0],
#          '^', markersize=10, color='green')
# ax1.plot(signals_fb.loc[signals_fb.positions == 1.0].index,
#          signals_fb.short_avg[signals_fb.positions == 1.0],
#          '^', markersize=10, color='green')
#
# # Plot the sell signals
# ax1.plot(signals_fb.loc[signals_fb.positions == -2.0].index,
#          signals_fb.short_avg[signals_fb.positions == -2.0],
#          'v', markersize=10, color='red')
# ax1.plot(signals_fb.loc[signals_fb.positions == -1.0].index,
#          signals_fb.short_avg[signals_fb.positions == -1.0],
#          'v', markersize=10, color='red')
#
# # Show the plot
# plt.show()

