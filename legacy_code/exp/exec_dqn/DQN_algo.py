import numpy as np
import tensorflow as tf
import time
from tensorflow import keras
import RLutils.logger as logger
import joblib
# import pickle as pickle
# from tensorflow.python.training.adam import AdamOptimizer

# import copy
#
# from RLutils.tensorflow_tools import initialize_uninitialized
# import pandas as pd
class SimpleReplayPool(object):
    def __init__(
            self, max_pool_size, observation_dim, action_dim, action_num):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_pool_size = max_pool_size
        self._observations = np.zeros(
            (max_pool_size, observation_dim),
        )
        self._actions = np.zeros(
            (max_pool_size, action_dim),
        )
        self._qpreds = np.zeros(
            (max_pool_size, action_num),
        )
        self._rewards = np.zeros(max_pool_size)
        self._terminals = np.zeros(max_pool_size, dtype='uint8')
        self._bottom = 0
        self._top = 0
        self._size = 0


    def add_sample(self, observation, action, qpred, reward, terminal):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._qpreds[self._top] = qpred
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
            qpreds = self._qpreds[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[transition_indices]
        )

    def size(self):
        return self._size

    def full(self):
        return self._size >= self._max_pool_size

class DQN:
    def __init__(self,
                env,
                qf,
                target_qf,
                session,
                epsilon,
                # es,
                tau = 0.01,
                explore_epoch = 1000,
                batch_size=40,
                n_epochs=1000000,
                epoch_length=200,
                min_pool_size=200,
                replay_pool_size=1000,
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
                n_updates_per_sample=10,
                scale_reward=1.0,
                include_horizon_terminal_transitions=False,
                plot=False,
                pause_for_plot=False,
                restore_fd = None):
        self.isplot = plot
        self.sess = session
        self.env = env

        # self.ploter = Ploter(480, 480)
        # self.reward = reward
        # self.noise = noise
        # self.discrete = discrete
        # self.policy = policy
        self.qf = qf
        self.epsilon = epsilon
        # self.target_policy = target_policy
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
        self.n_updates_per_sample = n_updates_per_sample
        self.include_horizon_terminal_transitions = include_horizon_terminal_transitions
        self.total_return = []
        self.qf_update_method = tf.train.AdamOptimizer(
            qf_learning_rate,
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
        self.tau = tau
        logger.task_name_scope = self.env.name
        self.restore_fd = restore_fd


    def train(self, from_itr = 'latest'):
        epoch = 0

        pool = SimpleReplayPool(
            max_pool_size=self.pool_size,
            observation_dim=self.env.obs_space(),
            action_dim=self.env.action_space(),
            action_num = self.env.action_num()
        )
        self.init_opt()

        sign = 1
        obj_share = -6
        self.env.set_objective(share= obj_share, remained_time= 5)

        obs = self.env.reset()
        terminal = True
        steps = []
        step = 0
        path_reward = 0
        path_length = 0
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if self.restore_fd:
            info = logger.init(restore=True, dir_name=self.restore_fd, session=self.sess,  itr=from_itr)
            epoch = info['epoch']
            pool = info['pool']
        else:
            logger.init()
        path_reward = 0
        path_length = 0

        for epoch in range(epoch, epoch+self.n_epochs,1):
            logger.push_prefix('epoch #%d | ' % epoch)
            logger.log("Training started")
            path_return = []
            path_value = []
            for step in range(self.epoch_length):

                action_idx, qpred = self.qf.get_action(obs) 
                action_idx = action_idx if np.random.uniform()<self.epsilon \
                                                                else np.random.choice(np.arange(0,self.env.action_num()))
                # print('action:',action)
                print(f'step: {step}')
                next_obs, reward, terminal, _ =self.env.step(action_idx)
                
                # print('%s: action = %s'% (path_length,action))
                # print('action, reward:', (action.tolist(),reward))
                # print('%s: reward = %s'% (step, reward))
                # print('%s: action = %s'% (step, action_idx))
                path_length += 1
                path_reward += reward
                path_return.append(reward)
                path_value.append(qpred.max())
                # print('singal reward:' ,reward)
                # if not terminal and path_length >= self.max_path_length:
                #     terminal = True
                #     # only include the terminal transition in this case if the flag was set
                #     if self.include_horizon_terminal_transitions:
                #         pool.add_sample(obs, action, reward, terminal)
                # else:
                pool.add_sample(obs, action_idx, qpred, reward, terminal)

                obs = next_obs
                if terminal:
                    
                    steps.append(step)
                    step = 0

                    sign *= -1
                    if obj_share < 0:
                        pass
                    else:
                        obj_share = (obj_share % 6)+1
                    obj_share *= -1
                    self.env.set_objective(share=obj_share, remained_time=5)
                    print('%s: reward = %s'% (path_length, path_reward))
                    path_reward = 0
                    path_length = 0
                    obs = self.env.reset()
            
            if pool.size() >= self.min_poolsize:
                for _ in range(self.n_updates_per_sample):
                    # start_time = time.time()
                    batch = pool.random_batch(self.batch_size)
                    # print('batch time consuming: ', time.time() - start_time)
                    # start_time = time.time()
                    self.do_training(batch)
                    # print('training time consuming: ', time.time()-start_time)
                obs = self.env.reset()
            self.env.save_to_csv(epoch)
            path_return = np.array(path_return)
            path_value = np.array(path_value)
            logger.record_tabular('epoch', epoch)
            logger.record_tabular('path reward', path_return.sum())
            logger.record_tabular('path reward (mean)', path_return.mean())
            logger.record_tabular('path reward (var)', path_return.var())
            logger.record_tabular('path value (sum)', path_value.sum())
            logger.record_tabular('path value (max)', path_value.max())
            logger.record_tabular('path value (min)', path_value.min())
            logger.record_tabular('path value (mean)', path_value.mean())
            logger.record_tabular('path value (var)', path_value.var())
            joblib.dump(steps, './steps.pkl', compress=3)
            logger.log('Training finished.')
            logger.save_itr_params(epoch, self.sess, self.get_epoch_snapshot(epoch=epoch, pool=pool))
            logger.dump_tabular(with_prefix=True)
            logger.pop_prefix()
        
        # self.policy.terminate()
        # save = pd.DataFrame(self.accumulators)
        # save.to_csv('result.csv')

    def load(self, itr):
        self.init_opt()
        if self.restore_fd:
            info = logger.init(restore=True, dir_name=self.restore_fd, session=self.sess, itr=itr)
        else:
            logger.init()
        return info

    def init_opt(self):
        target_qf = self.target_qf
        yval_ = keras.Input(dtype=tf.float32, name = 'yval', shape = (self.env.action_num()))
        qf_loss = tf.losses.mean_squared_error(labels=yval_, predictions=self.qf.out)
        qf_updt = self.qf_update_method.minimize(qf_loss)

        def qf_update(yval, obs):
            return self.sess.run((qf_updt, qf_loss, self.qf.out), feed_dict={self.qf.state: obs,
                                                                             yval_: yval
                                                                             })
        update_target_qfunction_network = \
            [target_qf.get_params[i].assign(tf.multiply(self.qf.get_params[i], self.tau) \
                                                + tf.multiply(target_qf.get_params[i], 1. - self.tau))
             for i in range(len(target_qf.get_params))]

        def update_target_network():
            self.sess.run(update_target_qfunction_network)

        # end
        init = tf.global_variables_initializer()
        self.sess.run(init)
        update_target_network()

        self.opt_info = dict(
            qf_update=qf_update,
            target_qf=target_qf,
                update_target_network=update_target_network
        )
        # if self.isplot:
        #     self.ploter.init()
        print('opt done')

    def _sub_val(self, row):
        row[int(row[-2])] = row[-1]
        return row[:-2]

    def do_training(self,batch):

        obs = batch['observations']
        actions = batch['actions']
        qpred = batch['qpreds']
        rewards = batch['rewards']
        terminals = batch['terminals']
        next_obs = batch['next_observations']
        # ep_ave_max_q = self.opt_info['ep_ave_max_q']
        target_qf = self.opt_info['target_qf']
        qf_update = self.opt_info['qf_update']
        update_target_network = self.opt_info['update_target_network']
        # a = time.time()
        # next_actions = target_policy.get_action(next_obs,False)
        # print('get action: ',time.time()-a)
        # a = time.time()
        next_qvals = target_qf.predict(next_obs).max(axis = 1).reshape(-1,1)
        # print('predict: ',time.time()-a)
        # a = time.time()
        rewards = rewards.reshape(-1, 1)
        terminals = terminals.reshape(-1, 1)
        # print('next_qvals : should be (bat_s, 1)', next_qvals.shape)
        yval = rewards+(1-terminals)*self.disc*next_qvals
        # print('yval shape: should be (bat_s, 1)', yval.shape)
        tmp = np.hstack((qpred, actions, yval))
        # print('tmp shape: should be (bat_s, act_num+2)', tmp.shape)
        sub_yval = np.apply_along_axis(self._sub_val, 1, tmp)
        # print(sub_yval, sub_yval - qpred, actions, yval)


        _,qf_loss, qval = qf_update(sub_yval, obs)
        # self.ep_ave_max_q += np.amax(qval)
        # self.ep_ave_max_q += np.amax(qval)
        # ep_ave_max_q += np.amax(qval)
        update_target_network()
        # print('qf param set: ',time.time() - a)
        # a = time.time()
        # all_variable = len(tf.trainable_variables())
        # print("How many variables: ", all_variable)
        # self.accumulators['qf_loss'].append(qf_loss)
        # self.accumulators['q_val'].append(qval)
        # self.accumulators['y_val'].append(yval)
        # print(self.accumulators['policy_surr'][-1])

    @staticmethod
    def get_epoch_snapshot(**kwargs):
        return dict(kwargs)

    def execute(self, shares, time):
        self.env.set_objective(share= shares, remained_time= time)
        print(f'Set obj: {shares}, {time}')
        observation = self.env.reset()
        step = 0
        rew = 0
        for i in range(100):
            action, qpred = self.qf.get_action(observation)
            # action = np.array([0,0,-10])
            next_ob, reward, next_terminal, _ = self.env.step(action)
            rew += reward
            step += 1
            observation = next_ob
            terminal = next_terminal
            # print(terminal)
            if terminal:
                break
                # print(ob)
        print(f'Executed in step: {step}!')
        return rew

if __name__ == '__main__':

    from qfunction import QFunction

    import gym

    sess = tf.Session()

    qf = QFunction(sess = sess,
                    state_dim=4,
                    action_dim=1,
                    action_num = 2,
                    name = 'qf',
                    hidden_units=(64,64))
                    
    target_qf  = QFunction(sess = sess,
                    state_dim=4,
                    action_dim=1,
                    action_num = 2,
                            name = 'target_qf',
                            hidden_units=(64,64))
    env = gym.make('CartPole-v0')
    env.action_num = lambda: 2
    env.action_space = lambda: 1
    env.obs_space = lambda: 4
    dqn = DQN(env=env,
                    qf=qf,
                    target_qf=target_qf,
                    session=sess,
                    epsilon = 0.95,
                # restore_fd='2019-04-21_19-14-17',
                )
    dqn.train()

    
    # for i_episode in range(20):
    #     observation = env.reset()
    #     for t in range(100):
    #         env.render()
    #         print(observation)
    #         action = env.action_space.sample()
    #         print(action)
    #         observation, reward, done, info = env.step(action)
    #         if done:
    #             print("Episode finished after {} timesteps".format(t+1))
    #             break
    env.close()




