import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)

from algo.ppo import PPO
from env.one_asset_env import one_asset_env as Env
from Backtesting.Engine.engine import Market
import pandas as pd
import matplotlib.pyplot  as plt

from policy.mlp_stoch_policy import Policy

import numpy as np
import tensorflow as tf





action_dim = 1
state_dim = 10
symbol = 'GS'
start = '2015-02-01'
end = '2017-01-03'
n_share = 100
seed = 13

env = Env(action_dim,state_dim, symbol, start , end, n_share)
sess = tf.Session()
tf.set_random_seed(seed)
policy = Policy(sess, state_dim, action_dim, 'ppo', hidden_units=(64, 64))

old_policy = Policy(sess, state_dim, action_dim, 'oldppo', hidden_units=(64, 64))

dspg = PPO(env=env,
                policy=policy,
                old_policy=old_policy,
                session=sess,
            # restore_fd='2018-08-21_15-00-13',
            policy_learning_rate = 0.001,
            epoch_length = 100,
           c1 = 1,
           c2 = 0.5,
           lam = 0.95,
           gamma = 0.99,
           max_local_step = 1000,
            )
dspg.train()




