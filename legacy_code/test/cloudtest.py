import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

from algos.ddpg_ga_cloud_multiThread import DSPG
# from algos.ddpg_ga_cloud_algo import DSPG
from env.local_cartpole_ga import CartPole
from env.fake_env import fake_env

from policy.ddpg_policy import Policy
from qfunction.ddpg_qfunction import QFunction
import numpy as np
import tensorflow as tf

from algos.reward import Reward
from algos.noise import Noise


env = CartPole(1,4,identifier = 'cart_pole', user='tianye', time_step=0.1)
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
# env = gym.make('CartPole-v0')
# action_dim = 1
# action_bound = 1.0
# state_dim = 4
####################### For cart_pole ########################
action_dim = 1
action_bound = 100
state_dim =4
discrete = True
##############################################################
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
                discrete = discrete,
            max_path_length=1000)
dspg.train()

# batch = {}
# batch['obs'] = []
# batch['action'] = []
# batch['next_obs'] = []
# batch['reward'] = []
# obs = env.obs
#
#
# for _ in range(40):
#     next_obs, reward, terminal, _ = env.step(action)
#     pool.add_sample(obs, action, reward, terminal)
#     obs = next_obs
#
# batch = pool.random_batch(20)
# obs = batch['observations']
# actions = batch['actions']
# rewards = batch['rewards']
# next_obs = batch['next_observations']
# print(batch)
# print(a,b,c,d)

# policy_update_method = tf.train.AdamOptimizer(0.01)
# q_a_grad = tf.placeholder(dtype=tf.float32)
# init = tf.global_variables_initializer()
# sess.run(init)
#
# print(qf.predict(obs,actions,False))
# print(policy.get_action(obs,False))
# a_gradient = qf.action_gradients(obs,actions)
#
# # q_a_grad = qf.action_gradients(obs, policy.get_action(obs, True))[0]
#
# tmp = tf.gradients(policy.scaled_out, policy.network_params, -q_a_grad)
# print(type(tmp),tmp)
# J_grad = list(map(lambda x: tf.div(x, 20), tmp))
# optimize = policy_update_method.apply_gradients(zip(J_grad, policy.network_params))
# init = tf.global_variables_initializer()
# sess.run(init)
# action_gradient = qf.action_gradients(obs, policy.get_action(obs, True))[0]
#
# sess.run(optimize, feed_dict={policy.inputs: obs,
#                                    policy.is_traning: True,
#                               q_a_grad:action_gradient})

#
# sess.run(optimize, feed_dict={
#     policy.inputs: obs,
#     policy.action_gradient: a_gradient[0],
#     policy.is_traning: False
# })
