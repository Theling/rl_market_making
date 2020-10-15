import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)

from ppo import PPO
import shift
#from env.one_asset_env import one_asset_env as Env
#from Backtesting.Engine.engine import Market
import pandas as pd
import matplotlib.pyplot  as plt

#from policy.mlp_stoch_policy import Policy

import numpy as np
import tensorflow as tf
from RRL_policy import Policy
from SHIFT_env import SHIFT_env, MarketOrder_exec
import time
from portfolio_value import *
import datetime


# action_dim = 1
# state_dim = 10
# symbol = 'IBM'
# start = '2017-01-03'
# end = '2018-11-01'
# n_share = 100
#
# env = Env(action_dim, state_dim, symbol, start, end, n_share)
# sess = tf.Session()
# policy = Policy(sess, state_dim, action_dim, 'ppo', hidden_units=(64, 64))
#
# old_policy = Policy(sess, state_dim, action_dim, 'oldppo', hidden_units=(64, 64))

# dspg = PPO(env=env,
#                 policy=policy,
#                 old_policy=old_policy,
#                 session=sess,
#             # restore_fd='2018-08-21_15-00-13',
#             policy_learning_rate = 0.001,
#             epoch_length = 100,
#            c1 = 1,
#            c2 = 0.5,
#            lam = 0.95,
#            gamma = 0.99,
#            max_local_step = 1000,
#             )
# dspg.train()

action_dim = 1
state_dim = 6
symbol = 'AAPL'
start = 100
end = 1900
steps = 30
# n_share = 100
seed = 22

trader = shift.Trader('test007')
trader.connect("initiator.cfg", "password")
trader.subAllOrderBook()
exec_ = MarketOrder_exec(trader, 'AAPL')
env = SHIFT_env(trader,
          scanner_wait_time=1,
          decision_gap=10,
          nTimeStep=steps+1,
          ODBK_range=10,
          symbol='AAPL',
          shares_factor = 1,
          executioner=exec_)

time.sleep(steps+3)
sess = tf.Session()
tf.set_random_seed(seed)
policy = Policy(sess, state_dim, action_dim, 'ppo',
                num_units = 64,
                 time_len = steps)

old_policy = Policy(sess, state_dim, action_dim, 'oldppo',
                num_units = 64,
                 time_len = steps)



dspg = PPO(env=env,
                policy=policy,
                old_policy=old_policy,
                session=sess,
            restore_fd='2019-04-21_15-30-20',
            policy_learning_rate = 0.001,
            epoch_length = 100,
           c1 = 1,
           c2 = 0.5,
           lam = 0.95,
           gamma = 0.99,
           max_local_step = 6,
            )
dspg.run(37)



# r1, _ = env.mkt.stats.trading_log()

# mkt = Market(start_date=start,
#                  end_date=end,
#                  cash=1000000,
#              rf = 0)
# position = pd.DataFrame([n_share],index=[symbol])
# mkt.adjustPosition(position)
# while True:
#     ret = mkt.next()
#     if ret == 0:
#         break
#
# r2, _ = mkt.stats.trading_log()
# plt.plot(range(r1.index.size), r1)
# plt.plot(range(r2.index.size), r2)
# plt.axvline(x=900, color='r', linestyle="--")
#
# plt.show()
