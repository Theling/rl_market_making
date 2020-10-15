import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)
import shift
from ppo import PPO
from SHIFT_env import SHIFT_env as Env, MarketOrder_exec
from RRL_policy import Policy


import numpy as np
import tensorflow as tf
import time





action_dim = 1
state_dim = 5
# symbol = 'GS'
# start = '2015-02-01'
# end = '2017-01-03'
# n_share = 100
seed = 13

trader = shift.Trader("test007")
trader.disconnect()
trader.connect("initiator.cfg", "password")
trader.subAllOrderBook()

executioner = MarketOrder_exec(trader, 'AAPL')

env = Env(trader,
          scanner_wait_time = 1,
          decision_gap = 10,
          nTimeStep = 50,
          ODBK_range = 10,
          symbol = 'AAPL',
          executioner = executioner)
time.sleep(60)


sess = tf.Session()
tf.set_random_seed(seed)
policy = Policy(sess, state_dim, action_dim, 'ppo',
                num_units = 64,
                 time_len = 50)

old_policy = Policy(sess, state_dim, action_dim, 'oldppo',
                num_units = 64,
                 time_len = 50)

dspg = PPO(env=env,
                policy=policy,
                old_policy=old_policy,
                session=sess,
            # restore_fd='2019-03-06_00-21-33',
            policy_learning_rate = 0.001,
           c1 = 1,
           c2 = 0.5,
           lam = 0.95,
           gamma = 0.99,
           max_local_step = 50,
           batch_size = 20
            )
dspg.train()




