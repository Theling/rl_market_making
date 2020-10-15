
import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)
import numpy as np
import shift
import tensorflow as tf

from ppo import PPO
from mlp_stoch_policy_dis import Policy
from SHIFT_env_4 import SHIFT_env as Env

def main():
    symbol = 'TRV'

    trader = shift.Trader("test001")
    trader.disconnect()
    trader.connect("initiator.cfg", "password")
    trader.sub_all_order_book()

    env = Env(trader = trader,
            t = 2,
            nTimeStep=5,
            ODBK_range=5,
            symbol=symbol)

    sess = tf.Session()


    policy = Policy(sess, 
                    state_dim = env.obs_space(),
                    action_space=range(env.action_num()),
                    name = 'ppo', 
                    hidden_units=(64, 64))

    old_policy = Policy(sess, 
                    state_dim = env.obs_space(),
                    action_space=range(env.action_num()),
                    name = 'oldppo', 
                    hidden_units=(64, 64))

    dspg = PPO(env=env,
                policy=policy,
                old_policy=old_policy,
                session=sess,
            # restore_fd='2020-02-07_13-33-31',
            # policy_learning_rate = 0.001,
            epoch_length = 500,
            c1 = 1,
            c2 = 0.1,
            lam = 1,
            gamma = 1,
            max_local_step = 50,
            )
    dspg.train()


if __name__ == "__main__":
    main()