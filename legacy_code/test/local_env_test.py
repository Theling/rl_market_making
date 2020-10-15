import os, sys
lib_path = os.path.abspath(os.path.join('..'))
if not lib_path in sys.path:
    sys.path.append(lib_path)

import time
from env.local_halfCheetah_ga import HalfCheetah
import numpy as np


action_bound = 1000
action_dim = 6
obs_dim = 25

env = HalfCheetah(action_dim,obs_dim,action_bound, time_step = 0.1, identifier = 'half_cheetah',user='tianye')

counter = 0
while True:
    print(env.done, env.obs)
    # env.action = np.random.normal(size=1)
    env.action = np.zeros(action_dim)
    counter += 1
    if env.done or counter > 100:
        print('resetting',env.done)
        counter = 0
#        env.reset()
#        print('after reset:', env.done)
#         reset_button = input('reset?')
        time.sleep(0.01)
        env.reset()
