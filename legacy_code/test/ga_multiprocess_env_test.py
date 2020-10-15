import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
import time
# from env.ga_cloud_env import cloud_env
from env.base.base_thread import env_base as cloud_env
#from env.fake_env import fake_env

import numpy as np

env = cloud_env(1,4,identifier = 'cart_pole')
print('env inited')
# env = fake_env(1,4)
action = np.random.normal(size=1)

done = False
obs = env.reset()
print('reset!')
while True:
    if done:
        obs = env.reset()
        done = False
    next_obs, reward, done, _ = env.step(np.random.normal(size=1))
    print('obs:', next_obs, end='   ')
    print('reward:', reward, end='   ')
    print('done:', done)
    time.sleep(0.05)
# sess = tf.Session()
# policy = Policy(sess,4,1,3,0.01,0.01)
# qf = QFunction(sess,4,1,0.01,0.01,num_actor_vars=len(policy.network_params))
# dspg = DSPG(env = env,
#             policy = policy,
#             qf = qf,
#             session = sess)
# dspg.train()
