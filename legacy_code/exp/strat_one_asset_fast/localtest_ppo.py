import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)

from Backtesting.Engine.engine import Market
import pandas as pd
import matplotlib.pyplot as plt


import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)

from ppo import PPO
from one_asset_env import One_Asset_Env
from RRL_policy import Policy

import tensorflow as tf





action_dim = 1
state_dim = 6
symbol = 'AAPL'
start = 900
end = 1900
nTimeStep = 30
# n_share = 100
seed = 22

# trader = shift.Trader("test002")
# trader.disconnect()
# trader.connect("initiator.cfg", "password")
# trader.subAllOrderBook()



env = One_Asset_Env(action_spec=action_dim,
                    obs_spec=state_dim,
                    symbol=symbol,
                    nTimeStep=nTimeStep,
                    start_date=start,
                    end_date=end,
                    n_share=1000,
                    commision = 0.003
                    )


sess = tf.Session()
tf.set_random_seed(seed)
policy = Policy(sess, state_dim, action_dim, 'ppo',
                num_units = 64,
                 time_len = nTimeStep)

old_policy = Policy(sess, state_dim, action_dim, 'oldppo',
                num_units = 64,
                 time_len = nTimeStep)

dspg = PPO(env=env,
                policy=policy,
                old_policy=old_policy,
                session=sess,
            restore_fd='2019-04-21_15-30-20',
            policy_learning_rate = 0.001,
           c1 = 1,
           c2 = 0.5,
           lam = 0.95,
           gamma = 0.99,
           max_local_step = 1000,
           batch_size = 100
            )
r3 = dspg.run(38, max_epoch=1)
r1, _ = env.mkt.stats.trading_log()
env.mkt.stats.compute(1800)

mkt = Market(start_date=start,
                 end_date=end,
                 cash=1000000,
             rf = 0)
position = pd.DataFrame([1000],index=[symbol])
mkt.adjustPosition(position)
while True:
    ret = mkt.next()
    if ret == 0:
        break

r2, _ = mkt.stats.trading_log()
plt.subplot(2, 1, 1)
plt.plot(range(r1.index.size), r1, label='strategy', linewidth=0.3, color='dodgerblue')
plt.plot(range(r2.index.size), r2, label='benchmark', linewidth=0.3, color='k')
# plt.axvline(x=900, color='k', linestyle="--", linewidth=0.6)
plt.legend()
plt.title('Portfolio Value')

a = []
for i in range(0, len(r3)):
    realaction = 1 if r3[i][0] > 0 else -1
    a.append(realaction)
plt.subplot(2, 1, 2)
plt.plot(range(len(a)), a, linewidth=0.1, color='dodgerblue')
plt.title('Action')
plt.show()

df_a = pd.DataFrame(a, columns=['Action'])
df_a['down'] = df_a.Action.diff()
ct = df_a['down'][df_a.down != 0]
print(ct.count())

len(a)
t=list(a)
t.pop()
t.pop(0)
stk = pd.read_csv('/mnt/c/Users/liuyu/PycharmProjects/FE800/RL/exp/strat_naive_fast/Backtesting/DB_interface/data/Close2.csv')

stkI = stk.iloc[100:900,:]
stkI['Action'] = t
stkI['up'] = stkI.Action.diff()>0
stkI['down'] = stkI.Action.diff()<0

stkO = stk.iloc[900:1900,:]
stkO['Action'] = t
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
