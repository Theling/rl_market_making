import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)
import shift
from ppo import PPO
from SHIFT_env import SHIFT_env as Env
from mlp_stoch_policy import Policy

import numpy as np
import tensorflow as tf
import time





action_dim = 2
state_dim = 8
# symbol = 'GS'
# start = '2015-02-01'
# end = '2017-01-03'
# n_share = 100
seed = 13

trader = shift.Trader("test002")
trader.disconnect()
trader.connect("initiator.cfg", "password")
trader.subAllOrderBook()

to_start = 0
while to_start == 0:
    time.sleep(1)
    to_start = len(trader.getOrderBook('CSCO', shift.OrderBookType.GLOBAL_ASK, 1))

env = Env(trader = trader,
          t = 5,
          nTimeStep=1,
          ODBK_range=5,
          symbol='CSCO')

sess = tf.Session()
tf.set_random_seed(seed)
policy = Policy(sess, state_dim, action_dim, 'ppo', hidden_units=(64, 64))

old_policy = Policy(sess, state_dim, action_dim, 'oldppo', hidden_units=(64, 64))

dspg = PPO(env=env,
                policy=policy,
                old_policy=old_policy,
                session=sess,
            # restore_fd='2019-03-13_21-52-23',
            policy_learning_rate = 0.001,
            epoch_length = 500,
           c1 = 1,
           c2 = 0.5,
           lam = 0.95,
           gamma = 0.99,
           max_local_step = 50,
            )
dspg.train()




