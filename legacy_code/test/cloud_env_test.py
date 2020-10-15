import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

from env.cloud_cartpole_ga import CartPole
import numpy as np
import time

env = CartPole(1,4,identifier = 'cart_pole')

while True:
    print(env.done, env.obs)
    env.action = np.random.normal(size=1)
    if env.done:
        print('resetting',env.done)
#        env.reset()
#        print('after reset:', env.done)
        reset_button = input('reset?')
        if reset_button:
            env.reset()
