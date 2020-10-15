import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

from algos.ppo_openai import PPO
from env.local_swimmer_ga import Swimmer
from env.fake_env import fake_env
from dm_control import suite

from policy.mlp_stoch_policy import Policy

import tensorflow as tf
import gym
ENV_NAME = 'Hopper-v2' # Discrete: Reward factor = 0.1
# env = gym.make(ENV_NAME)
# env.seed(1)
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
env = gym.make(ENV_NAME)


sess = tf.Session()


state_dim = env.observation_space.shape[0]

action_dim = 3
action_bound = env.action_space.high
print('action dim=',action_dim)
print('state_dim=', state_dim)
print('action_bound=,',action_bound,env.action_space.low)
policy = Policy(sess, state_dim, action_dim, 'pi', hidden_units=(64, 64))
old_policy = Policy(sess, state_dim, action_dim, 'oldpi', hidden_units=(64, 64))
# ploter = Ploter(640,480)
dspg = PPO(env=env,
                policy=policy,
                old_policy=old_policy,
                session=sess,
            # restore_fd='2018-08-21_15-00-13',
            # policy_learning_rate = 0.001,
            epoch_length = 100,
           c1 = 1,
           c2 = 0,
           lam = 0.95,
           gamma = 0.99,
           max_local_step = 2048,
            )
dspg.train()


