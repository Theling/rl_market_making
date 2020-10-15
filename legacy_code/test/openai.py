import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

# from dm_control import suite
from env.cloud_dm_env import dm_env
import numpy as np
from algos.ddpg_openai_local_algo import DSPG
from policy.ddpg_policy import Policy
from qfunction.ddpg_qfunction import QFunction
import numpy as np
import tensorflow as tf
from ploter_opencv.ploter import Ploter
from algos.actor import ActorNetwork
from algos.critic import CriticNetwork
import gym

from algos.replay_buffer import ReplayBuffer
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
env = gym.make('CartPole-v0')
action_dim = 2
action_bound = 1.0
state_dim = 4
# time_step = env.reset()
# a = time_step.observation
# reward = time_step.reward
# a = np.append(a['position'],(a['velocity']))
# sess = tf.Session()

# actor = ActorNetwork(sess, 4, 2, 1,ACTOR_LEARNING_RATE, TAU,tmp_length)
#
# critic = CriticNetwork(sess, 4, 2,CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars()+tmp_length)
# actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
#                              ACTOR_LEARNING_RATE, TAU)
#
# critic = CriticNetwork(sess, state_dim, action_dim,
#                                CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())
with tf.Session() as sess:
    env = gym.make(ENV_NAME)
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)

    print(env.observation_space)
    print(env.action_space)

    state_dim = env.observation_space.shape[0]
    # print(state_dim)
    try:
        action_dim = 1
        action_bound = 1
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)
        discrete = False
        print('Continuous Action Space')
    except AttributeError:
        action_dim = env.action_space.n
        action_bound = 1
        discrete = True
        print('Discrete Action Space')
    print(action_dim)
    # actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
    #                      ACTOR_LEARNING_RATE, TAU)
    #
    # critic = CriticNetwork(sess, state_dim, action_dim,
    #                        CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())
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

    # if GYM_MONITOR_EN:
    #     if not RENDER_ENV:
    #         env.monitor.start(MONITOR_DIR, video_callable=False, force=True)
    #     else:
    #         env.monitor.start(MONITOR_DIR, force=True)

    try:
        dspg.train()
    except KeyboardInterrupt:
        pass

    if GYM_MONITOR_EN:
        env.monitor.close()
# ploter = Ploter(640,480)
# tf.set_random_seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
# env.seed(RANDOM_SEED)

dspg.train()
