import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

from algo.ddpg import DSPG
# from env.cloud_halfCheetah_ga import HalfCheetah
from env.one_asset_env import one_asset_env as Env


from policy.ddpg_policy import Policy
from qfunction.ddpg_qfunction import QFunction
import numpy as np
import tensorflow as tf



action_dim = 1
action_bound = 1
state_dim = 5
discrete = True

env = Env(action_dim,state_dim, 'AAPL', '2015-02-02', '2018-01-02')
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


# ploter = Ploter(640,480)
dspg = DSPG(env=env,
                policy=policy,
                qf=qf,
                target_policy=target_policy,
                target_qf=target_qf,
                session=sess,
                discrete = discrete,
            max_path_length=1000,
            # restore_fd='2018-08-08_15-46-55',
            policy_learning_rate = 0.00001,
            epoch_length = 1000,
            )
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
