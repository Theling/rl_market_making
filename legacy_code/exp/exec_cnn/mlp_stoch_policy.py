import tensorflow as tf, numpy as np
class Policy():
    def __init__(self,
                 sess,
                 general_state_dim,
                 orderbook_dim,
                 action_dim,
                 name,
                 hidden_units,
                 ):
        self.sess = sess
        self.state_dim = general_state_dim
        self.orderBook_dim = orderbook_dim
        self.range = self.orderBook_dim[0]
        self.action_dim = action_dim
        self.scope = name
        self.hidden_units = hidden_units
        self.state, self.orderbook_ob,self.mu, self.logstd, self.value = self.create_network()
        self.sigma = tf.exp(self.logstd)
        self.shape = tf.placeholder(shape=(2),dtype=tf.int32)
        # self.action = 0.1*tf.nn.sigmoid(0.5*(self.mu + self.sigma * tf.random_normal(tf.shape(self.mu))))
        x = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu))
        self.action = 0.1 * (tf.nn.tanh(0.5 * x)*0.5+0.5)+0.01
        self.recurrent = False

    def cnn(self, in_):
        out = tf.reshape(tensor=in_, shape=[tf.shape(in_)[0], self.range, 3])
        out = tf.layers.conv1d(out,
                               filters=5,
                               kernel_size=5)
        out = tf.layers.max_pooling1d(out,
                                      pool_size=4,
                                      strides=4)
        out = tf.layers.flatten(out)
        return out
    def create_network(self):
        with tf.variable_scope(self.scope):
            general = tf.placeholder(name='add_ob', dtype=tf.float32, shape=[None, self.state_dim])
            orderbook = tf.placeholder(name='orderBook_ob', dtype=tf.float32, shape=[None, *self.orderBook_dim])

            with tf.variable_scope('vf'):
                out = self.cnn(orderbook)
                out = tf.concat([out, general], 1)
                for i, hidden in enumerate(self.hidden_units):
                    out = tf.layers.dense(out, hidden, name="fc%i" % (i + 1), activation=tf.nn.relu,
                                                          kernel_initializer=self.normc_initializer(1.0))
                value = tf.layers.dense(out, 1, name='final', kernel_initializer=self.normc_initializer(1.0))[:,0]
                self.vpred = value
            with tf.variable_scope('pol'):
                out = self.cnn(orderbook)
                out = tf.concat([out, general], 1)
                for i, hidden in enumerate(self.hidden_units):
                    out = tf.layers.dense(out, hidden, name='fc%i' % (i + 1), activation=tf.nn.relu,
                                                          kernel_initializer=self.normc_initializer(1.0))

                mean = tf.layers.dense(out, self.action_dim, name='final', kernel_initializer=self.normc_initializer(0.01))
                logstd = tf.get_variable(shape=[1,self.action_dim],name = 'std', initializer=tf.zeros_initializer())

        return general, orderbook ,mean,logstd,value


    def get_a_v(self, ob):
        act,value =  self.sess.run((self.action, self.value), feed_dict={
            self.orderbook_ob: [ob[0]],
            self.state: [ob[1]]

        })
        return act[0],value[0]


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
