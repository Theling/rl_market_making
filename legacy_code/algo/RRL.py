import numpy as np
import tensorflow as tf
import RLutils.logger as lg


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

    def isFull(self):
        return self._size >= self._max_pool_size


class RRL:
    def __init__(self,
                 sess,
                 env,
                 policy,
                 learning_rate,
                 restore = None,
                 pool_size = 500,
                 min_pool_size = 80,
                 n_update = 1,
                 batch_size = 30,
                 train_per_epoch = 10):
        self.sess = sess
        self.env = env
        self.policy = policy
        self.learning_rate = learning_rate
        self._restore = restore
        self.pool_size = pool_size
        self.min_pool_size  = min_pool_size
        self.n_update = n_update
        self.batch_size = batch_size
        self.train_per_epoch = train_per_epoch



    def init_opt(self):
        pass

    def train(self):
        self.init_opt()
        pool = SimpleReplayPool(
            max_pool_size=self.pool_size,
            observation_dim=self.env.obs_space(),
            action_dim=self.env.action_space(),
        )
        epoch = 0

        if self._restore:
            info = lg.init(restore=True,
                           dir_name=self._restore,
                           session = self.sess)
            epoch = info['epoch']
        else:
            lg.init()

        obs = self.env.reset()
        terminal = True

        while True:
            epoch += 1
            step = 0
            while True:
                action = self.policy.get_action(obs)
                next_obs, reward, next_terminal, _ = self.env.step(action)
                step += 1

                pool.add_sample(obs, action, reward, terminal)

                obs = next_obs

                if terminal:
                    break

            if pool.size() >= self.min_pool_size and epoch % self.train_per_epoch == 0:
                for _ in range(self.n_update):
                    batch = pool.random_batch(self.batch_size)
                    self.do_training(batch)


    def do_training(self, batch):
        pass




