import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

# from dm_control import suite
from dm_control import suite
import numpy as np
from algos.ddpg_dm_local_algo import DSPG
from policy.ddpg_policy import Policy
from qfunction.ddpg_qfunction import QFunction
import numpy as np
import tensorflow as tf
from ploter_opencv.ploter import Ploter

from algos.reward import Reward
from algos.noise import Noise

from gym import wrappers
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
# env = gym.make('CartPole-v0')
# action_dim = 1
# action_bound = 1.0
# state_dim = 4
env = suite.load(domain_name="cartpole", task_name="balance")
action_dim = 1
action_bound = 5.0
state_dim = 5
discrete = True
# time_step = env.reset()
# a = time_step.observation
# reward = time_step.reward
# a = np.append(a['position'],(a['velocity']))
sess = tf.Session()
policy = Policy(sess, state_dim, action_dim, action_bound, 0.0001, 0.01, hidden_size=(400, 300))
qf = QFunction(sess, state_dim, action_dim, 0.001, 0.01, num_actor_vars=len(policy.network_params),
               hidden_size=(400, 300))
tmp_length = len(policy.network_params) + len(qf.network_params)
target_policy = Policy(sess, state_dim, action_dim, action_bound, 0.0001, 0.01, hidden_size=(400, 300),
                       num_var_before=tmp_length)
tmp_length += len(target_policy.network_params)
target_qf = QFunction(sess, state_dim, action_dim, 0.001, 0.01, num_actor_vars=tmp_length, hidden_size=(400, 300))
tmp_length += len(target_qf.network_params)

noise = Noise(DELTA, SIGMA, OU_A, OU_MU)
reward = Reward(REWARD_FACTOR, GAMMA)
# ploter = Ploter(640,480)
dspg = DSPG(env=env,
                policy=policy,
                qf=qf,
                target_policy=target_policy,
                target_qf=target_qf,
                session=sess,
                actor=None,
                critic=None,
                noise = noise,
                reward = reward,
                discrete = discrete)
dspg.train()
