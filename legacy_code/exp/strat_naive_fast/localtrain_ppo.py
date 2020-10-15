import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)
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
# symbol = 'GS'
start = 100
end = 2000
# n_share = 100
seed = 13

# trader = shift.Trader("test002")
# trader.disconnect()
# trader.connect("initiator.cfg", "password")
# trader.subAllOrderBook()



env = One_Asset_Env(action_spec=action_dim,
                    obs_spec=state_dim,
                    symbol='AAPL',
                    nTimeStep=state_dim-1,
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
            # restore_fd='2019-04-18_18-39-06',
            policy_learning_rate = 0.001,
           c1 = 1,
           c2 = 0.5,
           lam = 0.95,
           gamma = 0.99,
           max_local_step = 2000,
           batch_size = 100
            )
dspg.train()




