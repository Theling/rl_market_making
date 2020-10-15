import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

from algos.ppo_mujoco import PPO
from env.local_swimmer_ga import Swimmer
from env.fake_env import fake_env
from dm_control import suite

from policy.mlp_stoch_policy import Policy

import numpy as np
import tensorflow as tf

from algos.reward import Reward
from algos.noise import Noise


# env = fake_env(1,4)
# from gym import wrappers
MAX_EPISODES = 1000
# Max episode length
MAX_EP_STEPS = 1000
# Episodes with noise
NOISE_MAX_EP = 200
# Noise parameters - Ornstein Uhlenbeck
DELTA = 0.5 # The rate of change (time)
SIGMA = 0.5 # Volatility of the stochastic processes
OU_A = 3. # The rate of mean reversion
OU_MU = 0. # The long run average interest rate
# Reward parameters
REWARD_FACTOR = 0.1 # Total episode reward factor
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = False
# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
ENV_NAME = 'CartPole-v0' # Discrete: Reward factor = 0.1
#ENV_NAME = 'CartPole-v1' # Discrete: Reward factor = 0.1
#ENV_NAME = 'Pendulum-v0' # Continuous: Reward factor = 0.01
# Directory for storing gym results
MONITOR_DIR = './results/' + ENV_NAME
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 100000
MINIBATCH_SIZE = 100
RANDOM_SEED = 1234





discrete = True

env = suite.load(domain_name="cartpole", task_name="balance")
action_spec = env.action_spec()

sess = tf.Session()
obs_dict = env.reset().observation


obs = np.append(obs_dict['position'], obs_dict['velocity'])
state_dim = obs.shape[0]
action_dim = action_spec.shape[0]
action_bound = action_spec.maximum
print('action dim=',action_dim)
print('state_dim=', state_dim)
print('action_bound=,',action_bound,action_spec.minimum)
policy = Policy(sess, state_dim, action_dim, action_bound, 'ppo', hidden_units=(128, 64))

old_policy = Policy(sess, state_dim, action_dim, action_bound, 'oldppo', hidden_units=(128, 64))

noise = Noise(DELTA, SIGMA, OU_A, OU_MU)
reward = Reward(REWARD_FACTOR, GAMMA)
# ploter = Ploter(640,480)
dspg = PPO(env=env,
                policy=policy,
                old_policy=old_policy,
                session=sess,
            # restore_fd='2018-08-29_03-32-32',
            policy_learning_rate = 0.001,
            epoch_length = 100,
           c1 = 1,
           c2 = 0,
           lam = 0.95,
           gamma = 0.99,
           max_local_step = 2048,
            )
dspg.train()


