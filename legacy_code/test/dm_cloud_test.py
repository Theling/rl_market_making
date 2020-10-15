import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

# from dm_control import suite
from env.cloud_dm_env import dm_env
import numpy as np
from algos.ddpg_dm_cloud_algo import DSPG
from policy.ddpg_policy import Policy
from qfunction.ddpg_qfunction import QFunction
import numpy as np
import tensorflow as tf
from ploter_opencv.ploter import Ploter

env = dm_env()
action_dim = 1
action_bound = 1.0
state_dim = 5
# time_step = env.reset()
# a = time_step.observation
# reward = time_step.reward
# a = np.append(a['position'],(a['velocity']))
sess = tf.Session()
policy = Policy(sess,state_dim,action_dim,action_bound,0.01,0.01)
qf = QFunction(sess,state_dim,action_dim,0.01,0.01,num_actor_vars=len(policy.network_params))
# ploter = Ploter(640,480)
dspg = DSPG(env = env,
            policy = policy,
            qf = qf,
            session = sess,
            ploter = None)
dspg.train()
