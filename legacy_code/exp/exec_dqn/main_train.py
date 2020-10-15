
import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)
import numpy as np
import shift
import tensorflow as tf

from DQN_algo import DQN
from mlp_stoch_policy import Policy
from qfunction import QFunction
from SHIFT_env_4 import SHIFT_env as Env

def main():
    symbol = 'AAPL'

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

    qf = QFunction(sess = sess,
                    state_dim=env.obs_space(),
                    action_dim=env.action_space(),
                    action_num = env.action_num(),
                    name = 'qf',
                    hidden_units=(64,64))
                    
    target_qf  = QFunction(sess = sess,
                            state_dim=env.obs_space(),
                            action_dim=env.action_space(),
                            action_num = env.action_num(),
                            name = 'target_qf',
                            hidden_units=(64,64))



    dqn = DQN(env=env,
                    qf=qf,
                    target_qf=target_qf,
                    session=sess,
                    epsilon = 0.95,
                restore_fd='2019-11-22_16-16-16',
                discount=0.99
                )
    dqn.train()


if __name__ == "__main__":
    main()