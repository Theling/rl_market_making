import tensorflow as tf, numpy as np
class Policy():
    def __init__(self,
                 sess,
                 state_dim,
                 action_dim,
                 name,
                 num_units,
                 time_len,
                 ):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.scope = name
        self.time_len = time_len
        self.hidden_units = num_units
        self.state,self.mu, self.logstd, self.value = self.create_network()
        self.sigma = tf.exp(self.logstd)
        self.shape = tf.placeholder(shape=(2),dtype=tf.int32)
        # self.action = 0.1*tf.nn.sigmoid(0.5*(self.mu + self.sigma * tf.random_normal(tf.shape(self.mu))))
        x = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu))
        self.action = x
        self.recurrent = False

    def create_network(self):
        with tf.variable_scope(self.scope):
            states = tf.placeholder(shape=[None, self.time_len, self.state_dim], dtype=tf.float32, name = 'state')
            lstm = tf.nn.rnn_cell.LSTMCell(self.hidden_units)
            input_c = lstm.zero_state(tf.shape(states)[0], dtype=tf.float32)
            # input_c = tf.placeholder(shape = (None, 1), dtype=tf.float32)
            unstacked_input = tf.unstack(states, axis=1)
            # out, _ = tf.contrib.rnn.static_rnn(lstm, unstacked_input, dtype = tf.float32)
            out = self.create_LSTM(unstacked_input, input_c, lstm)
            with tf.variable_scope('vf'):
                pre = tf.layers.dense(out[-1], 32, name='final_1', activation=tf.nn.relu,
                                      kernel_initializer=tf.initializers.random_normal())
                pre = tf.layers.dense(pre, 1, name='final_2',
                                      kernel_initializer=tf.initializers.random_normal())
                value = pre

            with tf.variable_scope('pol'):
                pre = tf.layers.dense(out[-1], 32, name='final_1', activation=tf.nn.tanh,
                                      kernel_initializer=tf.initializers.random_normal())
                pre = tf.layers.dense(pre, self.action_dim, name='final_2', activation=tf.nn.tanh,
                                      kernel_initializer=tf.initializers.random_normal())
                mean = pre
                logstd = tf.get_variable(shape=[1,self.action_dim],name = 'std', initializer=tf.zeros_initializer())

        return states,mean,logstd,value

    @staticmethod
    def create_LSTM(inputs, init_c, lstm):
        ret = []
        state = init_c
        for ipt in inputs:
            out, state = lstm(ipt, state)
            ret.append(out)
        return ret

    def get_a_v(self, ob):
        act,value =  self.sess.run((self.action, self.value), feed_dict={
            self.state: [ob]
        })
        return act[0],value[0][0]


    def get_param(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    def get_trainable_param(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

    def neglogp(self, x):
        print('x is',x)
        return 0.5 * tf.reduce_sum(tf.square((x - self.mu) / self.sigma), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(self.logstd, axis=-1)


    def normc_initializer(self, std=1.0, axis=0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
            out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
            return tf.constant(out)

        return _initializer
