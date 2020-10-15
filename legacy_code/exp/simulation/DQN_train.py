
import os, sys
os.chdir('dev')
lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)

# lib_path = os.path.abspath(os.path.join('../..'))
# sys.path.append(lib_path)

import numpy as np
import tensorflow as tf
print(f'tf version: {tf.__version__}')

from Algo.dqn import DQN, QFunction
from Env import RLBroker, Brokerage, Market

def main():
    
    print(os.getcwd())

    mkt = Market('Env/mkt_data/TRV20200220',
                 'sc',
                 0.00025,
                 threshold=0,
                 reverse_position=0,
                 MAX_BID=0)

    env = RLBroker(Brokerage(mkt))

    sess = tf.Session()

    qf = QFunction(sess = sess,
                    state_dim=env.obs_dim,
                    action_dim = env.act_dim,
                    action_num = len(env.act_space),
                    name = 'qf',
                    hidden_units=(64,64))
                    
    target_qf = QFunction(sess = sess,
                    state_dim=env.obs_dim,
                    action_dim = env.act_dim,
                    action_num = len(env.act_space),
                    name = 'target_qf',
                    hidden_units=(64,64))



    dqn = DQN(env=env,
                    qf=qf,
                    target_qf=target_qf,
                    session=sess,
                    epsilon = 0.95,
                # restore_fd='2019-04-21_19-14-17',
                discount=0.99
                )
    dqn.train()


if __name__ == "__main__":
    main()