import numpy as np
import tensorflow as tf
import time
import RLutils.logger as logger
# import pickle as pickle
# from tensorflow.python.training.adam import AdamOptimizer
# import copy
#
# from RLutils.tensorflow_tools import initialize_uninitialized
# import pandas as pd
class SimpleReplayPool(object):
    def __init__(
            self, max_pool_size, observation_dim, action_dim):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_pool_size = max_pool_size
        self._observations = np.zeros(
            (max_pool_size, observation_dim),
        )
        self._actions = np.zeros(
            (max_pool_size, action_dim),
        )
        self._rewards = np.zeros(max_pool_size)
        self._terminals = np.zeros(max_pool_size, dtype='uint8')
        self._bottom = 0
        self._top = 0
        self._size = 0


    def add_sample(self, observation, action, reward, terminal):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._top = (self._top + 1) % self._max_pool_size
        if self._size >= self._max_pool_size:
            self._bottom = (self._bottom + 1) % self._max_pool_size
        else:
            self._size += 1

    def random_batch(self, batch_size):
        assert self._size > batch_size
        indices = np.zeros(batch_size, dtype='uint64')
        transition_indices = np.zeros(batch_size, dtype='uint64')
        count = 0
        while count < batch_size:
            index = np.random.randint(self._bottom, self._bottom + self._size) % self._max_pool_size
            # make sure that the transition is valid: if we are at the end of the pool, we need to discard
            # this sample
            if index == self._size - 1 and self._size <= self._max_pool_size:
                continue
            # if self._terminals[index]:
            #     continue
            transition_index = (index + 1) % self._max_pool_size
            indices[count] = index
            transition_indices[count] = transition_index
            count += 1
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[transition_indices]
        )

    def size(self):
        return self._size

    def full(self):
        return self._size >= self._max_pool_size

MAX_EPISODES = 1000
# Max episode length
MAX_EP_STEPS = 1000
# Episodes with noise
NOISE_MAX_EP = 200
# Noise parameters - Ornstein Uhlenbeck
DELTA = 0.5  # The rate of change (time)
SIGMA = 0.5  # Volatility of the stochastic processes
OU_A = 3.  # The rate of mean reversion
OU_MU = 0.  # The long run average interest rate
# Reward parameters
REWARD_FACTOR = 0.1  # Total episode reward factor
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
ENV_NAME = 'CartPole-v0'  # Discrete: Reward factor = 0.1
# ENV_NAME = 'CartPole-v1' # Discrete: Reward factor = 0.1
# ENV_NAME = 'Pendulum-v0' # Continuous: Reward factor = 0.01
# Directory for storing gym results
MONITOR_DIR = './results/' + ENV_NAME
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 100000
MINIBATCH_SIZE = 100


class DSPG:
    def __init__(self,
                env,
                policy,
                qf,
                target_policy,
                target_qf,
                session,

                discrete,
                # es,
                explore_epoch = 500,
                batch_size=50,
                n_epochs=10000,
                epoch_length=100,
                min_pool_size=80,
                replay_pool_size=2000,
                discount=0.99,
                max_path_length=250,
                qf_weight_decay=0.,
                qf_update_method='adam',
                qf_learning_rate=1e-3,
                policy_weight_decay=0,
                policy_update_method='adam',
                policy_learning_rate=1e-4,
                eval_samples=10000,
                soft_target=True,
                soft_target_tau=0.001,
                n_updates_per_sample=1,
                scale_reward=1.0,
                include_horizon_terminal_transitions=False,
                plot=False,
                pause_for_plot=False,
                restore_fd = None):
        self.isplot = plot
        self.sess = session
        self.env = env

        # self.ploter = Ploter(480, 480)

        self.discrete = discrete
        self.policy = policy
        self.qf = qf
        self.target_policy = target_policy
        self.target_qf = target_qf
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.explore_epoch = explore_epoch
        self.epoch_length = epoch_length
        self.len_epochs = epoch_length
        self.min_poolsize = min_pool_size
        self.pool_size = replay_pool_size
        self.disc = discount
        self.max_path_length = max_path_length
        self.qf_weight_decay = qf_weight_decay
        self.qf_learning_rate = qf_learning_rate
        self.policy_weight_decay = policy_weight_decay
        self.policy_learning_rate = policy_learning_rate
        self.eval_samples = eval_samples
        self.n_updates_per_sample = n_updates_per_sample
        self.include_horizon_terminal_transitions = include_horizon_terminal_transitions
        self.total_return = []
        self.qf_update_method = tf.train.AdamOptimizer(
            qf_learning_rate,
            # initial_accumulator_value=0.1,
            # use_locking=False,
            # name='Adagrad'
        )
        self.policy_update_method = tf.train.AdamOptimizer(
            policy_learning_rate,
            # initial_accumulator_value=0.1,
            # use_locking=False,
            # name='Adagrad'
        )
        self.accumulators = dict()
        self.accumulators['qf_loss'] = []
        self.accumulators['policy_surr'] = []
        self.accumulators['q_val'] = []
        self.accumulators['y_val'] = []
        self.opt_info = None
        # self.noise = Noise(DELTA, SIGMA, OU_A, OU_MU)
        self.tau = TAU
        # logger.task_name_scope = self.env.name
        self.restore_fd = restore_fd


    def train(self):
        epoch = 1
        pool = SimpleReplayPool(
            max_pool_size=self.pool_size,
            observation_dim=self.env.obs_space(),
            action_dim=self.env.action_space(),
        )
        self.init_opt()
        sample_policy = self.policy
        # terminal = False
        while not self.env.done:
            time.sleep(0.001)
        obs = self.env.reset()
        path_return = 0
        path_length = 0
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if self.restore_fd:
            info = logger.init(restore=True, dir_name=self.restore_fd, session=self.sess)
            epoch = info['epoch']
        else:
            logger.init()
        for epoch in range(epoch, epoch+self.n_epochs,1):
            # logger.push_prefix('epoch #%d | ' % epoch)
            # logger.log("Training started")
            self.ep_ave_max_q = 0
            path_return = 0
            path_length = 0
            for step in range(self.epoch_length):
                if self.env.done:
                    break

                print([obs])
                action = sample_policy.get_action([obs],False)+0.1*np.random.uniform(-1,1,self.env.action_space())*self.policy.action_bound/np.sqrt(epoch)
                # print('action:',action)
                next_obs, reward, terminal, _ =self.env.step(action)
                print('%s: reward = %s, action = %s' % (step, reward, action))
                # print('%s: action = %s'% (path_length,action))
                # print('action, reward:', (action.tolist(),reward))
                # print('%s: reward = %s'% (step, reward))
                # print(self.env.done)
                path_length += 1
                path_return += reward
                print('path reward:' ,path_return)
                # if not terminal and path_length >= self.max_path_length:
                #     terminal = True
                #     # only include the terminal transition in this case if the flag was set
                #     if self.include_horizon_terminal_transitions:
                #         pool.add_sample(obs, action, reward, terminal)
                # else:
                pool.add_sample(obs, action, reward, terminal)

                obs = next_obs

                if pool.size() >= self.min_poolsize:
                    for _ in range(self.n_updates_per_sample):
                        # start_time = time.time()
                        batch = pool.random_batch(self.batch_size)
                        # print('batch time consuming: ', time.time() - start_time)
                        # start_time = time.time()
                        self.do_training(batch)
                        # print('training time consuming: ', time.time()-start_time)
                    # sample_policy.set_param(self.policy.get_param())
            logger.save_itr_params(epoch, self.sess, self.get_epoch_snapshot(epoch))
            obs = self.env.reset()
            self.total_return.append(path_return)
            sample_policy.reset()
            logger.record_tabular('epoch', epoch)
            logger.record_tabular('Qmax', self.ep_ave_max_q / float(step + 1))
            logger.record_tabular('path reward', path_return)
            logger.dump_tabular(with_prefix=True)
            logger.pop_prefix()
        self.env.terminate()
        self.policy.terminate()
        # save = pd.DataFrame(self.accumulators)
        # save.to_csv('result.csv')


    def init_opt(self):
        target_qf = self.target_qf
        target_policy = self.target_policy

        # self.length = length+len(target_qf.get_param())+len(target_policy.get_param())
        # self.length = length+len(target_policy.get_param())+len(target_qf.get_param())
        # self.actor = ActorNetwork(self.sess, 4, 2, 1,
        #                           ACTOR_LEARNING_RATE, TAU)
        #
        # self.critic = CriticNetwork(self.sess, 4, 2,
        #                             CRITIC_LEARNING_RATE, TAU, self.actor.get_num_trainable_vars())
        # Define graph for updating q function
        yval_ = tf.placeholder(tf.float32, [None, 1])
        qf_loss = tf.losses.mean_squared_error(labels=yval_, predictions=self.qf.out)
        qf_updt = self.qf_update_method.minimize(qf_loss)

        def qf_update(yval, obs, action):
            # self.sess.run([qf_updt,self.qf.out],feed_dict={self.qf.inputs:obs,
            #                                  yval_:yval,
            #                                  self.qf.action:action})
            return self.sess.run((qf_updt, qf_loss, self.qf.out), feed_dict={self.qf.inputs: obs,
                                                                             yval_: yval,
                                                                             self.qf.action: action,
                                                                             self.qf.is_traning: True
                                                                             })

        # end

        # Define graph for updating policy
        q_a_grad = tf.placeholder(tf.float32, [None, self.env.action_space()])
        tmp = tf.gradients(self.policy.scaled_out, self.policy.network_params, -q_a_grad)
        # J_grad = list(map(lambda x: tf.div(x, 20), tmp))
        optimize = self.policy_update_method.apply_gradients(zip(tmp, self.policy.network_params))

        def policy_update(obs):
            action = self.policy.get_action(obs, False)
            # print('action',action[0:1])
            # print('obs',obs[0:1])
            gradient = self.qf.action_gradients(obs, action)
            # print('gradient',gradient)
            # a_outs = self.actor.predict(obs)
            # print('a_outs',a_outs)
            # grads = self.critic.action_gradients(obs, a_outs)
            # print('grads', grads)
            return self.sess.run(optimize, feed_dict={self.policy.inputs: obs,
                                                      self.policy.is_traning: True,
                                                      q_a_grad: gradient[0]})
            # return np.mean(self.qf.predict(obs, self.policy.get_action(obs, False),False))

        update_target_policy_network = \
            [target_policy.network_params[i].assign(tf.multiply(self.policy.network_params[i], self.tau) \
                                                    + tf.multiply(target_policy.network_params[i], 1. - self.tau))
             for i in range(len(target_policy.network_params))]

        update_target_qfunction_network = \
            [target_qf.network_params[i].assign(tf.multiply(self.qf.network_params[i], self.tau) \
                                                + tf.multiply(target_qf.network_params[i], 1. - self.tau))
             for i in range(len(target_qf.network_params))]

        def update_target_network():
            self.sess.run((update_target_policy_network, update_target_qfunction_network))

        # end
        init = tf.global_variables_initializer()
        self.sess.run(init)
        update_target_network()

        self.opt_info = dict(
            qf_update=qf_update,
            policy_update=policy_update,
            target_qf=target_qf,
            target_policy=target_policy,
            update_target_network=update_target_network
        )
        # if self.isplot:
        #     self.ploter.init()
        print('opt done')

    def do_training(self,batch):
        # print('training')
        obs = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminals']
        next_obs = batch['next_observations']
        # ep_ave_max_q = self.opt_info['ep_ave_max_q']
        target_qf = self.opt_info['target_qf']
        target_policy = self.opt_info['target_policy']
        qf_update = self.opt_info['qf_update']
        policy_update = self.opt_info['policy_update']
        update_target_network = self.opt_info['update_target_network']
        # a = time.time()
        next_actions = target_policy.get_action(next_obs,False)
        # print('get action: ',time.time()-a)
        # a = time.time()
        next_qvals = target_qf.predict(next_obs, next_actions, False)
        # print('predict: ',time.time()-a)
        # a = time.time()
        rewards = rewards.reshape(-1, 1)
        terminals = terminals.reshape(-1, 1)
        yval = rewards+(1-terminals)*self.disc*next_qvals


        _,qf_loss, qval = qf_update(yval,obs, actions)
        self.ep_ave_max_q += np.amax(qval)
        # self.ep_ave_max_q += np.amax(qval)
        # ep_ave_max_q += np.amax(qval)
        # print('policy update: ',time.time()-a)
        # a = time.time()
        policy_update(obs)
        # print('policy update: ',time.time() - a)
        # a = time.time()
        # target_policy.set_param(self.policy.get_param())
        # # print('policy param set: ',time.time() - a)
        #
        # target_qf.set_param(self.qf.get_param())
        update_target_network()
        # print('qf param set: ',time.time() - a)
        # a = time.time()
        # all_variable = len(tf.trainable_variables())
        # print("How many variables: ", all_variable)
        # self.accumulators['qf_loss'].append(qf_loss)
        # self.accumulators['q_val'].append(qval)
        # self.accumulators['policy_surr'].append(policy_surr)
        # self.accumulators['y_val'].append(yval)
        # print(self.accumulators['policy_surr'][-1])

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
        )

    def run(self, ckpt):
        epoch = 1

        self.init_opt()
        sample_policy = self.policy
        # terminal = False
        while not self.env.done:
            time.sleep(0.001)
        obs = self.env.reset()
        path_return = 0
        path_length = 0
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # if self.restore_fd:
        info = logger.init(restore=True, dir_name=ckpt, session=self.sess)
        epoch = info['epoch']
        # else:
        #     logger.init()
        for epoch in range(epoch, epoch+self.n_epochs,1):
            # logger.push_prefix('epoch #%d | ' % epoch)
            # logger.log("Training started")
            self.ep_ave_max_q = 0
            path_return = 0
            path_length = 0
            for step in range(self.epoch_length):
                if self.env.done:
                    break
                action = sample_policy.get_action([obs],False)+0.1*np.random.uniform(-1,1,self.env.action_space())*self.policy.action_bound/np.sqrt(epoch)
                # print('action:',action)
                next_obs, reward, terminal, _ =self.env.step(action)
                print('%s: reward = %s, action = %s' % (step, reward, action))
                # print('%s: action = %s'% (path_length,action))
                # print('action, reward:', (action.tolist(),reward))
                # print('%s: reward = %s'% (step, reward))
                # print(self.env.done)
                path_length += 1
                path_return += reward
                print('path reward:' ,path_return)
                # if not terminal and path_length >= self.max_path_length:
                #     terminal = True
                #     # only include the terminal transition in this case if the flag was set
                #     if self.include_horizon_terminal_transitions:
                #         pool.add_sample(obs, action, reward, terminal)
                # else:
                # pool.add_sample(obs, action, reward, terminal)

                obs = next_obs

                        # print('training time consuming: ', time.time()-start_time)
                    # sample_policy.set_param(self.policy.get_param())
            # logger.save_itr_params(epoch, self.sess, self.get_epoch_snapshot(epoch))
            obs = self.env.reset()
            self.total_return.append(path_return)
            sample_policy.reset()
            # logger.record_tabular('epoch', epoch)
            # logger.record_tabular('Qmax', self.ep_ave_max_q / float(step + 1))
            # logger.record_tabular('path reward', path_return)
            # logger.dump_tabular(with_prefix=True)
            # logger.pop_prefix()
        self.env.terminate()
        self.policy.terminate()
        # save = pd.DataFrame(self.accumulators)
        # save.to_csv('result.csv')
















