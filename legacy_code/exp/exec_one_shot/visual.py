import os, sys
os.chdir('RL/exp/exec_one_asset')
lib_path = os.path.abspath(os.path.join('.'))
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
from itertools import product


action_dim = 1
state_dim = 7
seed = 13
nTime = 60
# symbol = 'AAPL'
# trader = shift.Trader('test007')
# trader.connect("initiator.cfg", "password")
# trader.subAllOrderBook()
#
# env = Env(trader = trader,
#           t = 2,
#           nTimeStep=1,
#           ODBK_range=5,
#           symbol='AAPL')
class A:
    pass

env = A()
env.name = 'exec_one_asset'

sess = tf.Session()
tf.set_random_seed(seed)
policy = Policy(sess, state_dim, action_dim, 'ppo', hidden_units=(64, 64))

old_policy = Policy(sess, state_dim, action_dim, 'oldppo', hidden_units=(64, 64))

dspg = PPO(env=env,
                policy=policy,
                old_policy=old_policy,
                session=sess,
            restore_fd='2019-03-09_17-01-42',
            policy_learning_rate = 0.001,
            epoch_length = 500,
           c1 = 1,
           c2 = 0.5,
           lam = 0.95,
           gamma = 0.99,
           max_local_step = 50,
            )
dspg.load(itr=72)

time_ls = [1,2,3,4,5]
share_ls = [5]
a = product(time_ls, share_ls)

for t,s in a:

    print(t,s,policy.get_a_v(np.array([ 0.1,
                                0.14912,
                                0.149095,
                                0.,
                                0.,
                                s,
                                t])))



