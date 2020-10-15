import tensorflow as tf, numpy as np
from tensorflow import keras
class Policy():
    def __init__(self,
                 sess,
                 state_dim,
                 action_space,
                 name,
                 hidden_units,
                 ):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = 1
        self.action_space = action_space # a list of array
        self.action_num = len(action_space)
        self.action_idx = list(range(self.action_num))
        self.scope = name
        self.hidden_units = hidden_units
        self.state,self.act_dist, self.values = self.create_network()
        # self.shape = tf.placeholder(shape=(2),dtype=tf.int32)
        # self.action = tf.nn.tanh(self.mu + self.sigma * tf.random_normal(tf.shape(self.mu)))
        self.recurrent = False


    def create_network(self):
        with tf.variable_scope(self.scope):
            states = keras.Input(name='ob',dtype=tf.float32,shape=(self.state_dim,))

            # out = tf.reshape(tensor = states, shape = [tf.shape(states)[0], self.state_dim, 1])
            # out = tf.layers.conv1d(out,
            #                        filters=10,
            #                        kernel_size = 5)
            # out = tf.layers.max_pooling1d(out,
            #                              pool_size = 4,
            #                              strides = 4)
            # out = tf.layers.flatten(out)
            
            with tf.variable_scope('vf'):
                out = states
                for i, hidden in enumerate(self.hidden_units):
                    out = keras.layers.Dense(hidden, name=f"fc{i+1}", 
                    kernel_initializer=self.normc_initializer(1.0), 
                    activation=tf.nn.relu)(out)    
                values = keras.layers.Dense(self.action_num , 
                kernel_initializer=self.normc_initializer(1.0), name='final')(out)
                self.value_model = keras.Model(inputs = states, outputs = values)

            with tf.variable_scope('pol'):
                out = states
                for i, hidden in enumerate(self.hidden_units):
                    out = keras.layers.Dense(hidden, name=f"fc{i+1}",kernel_initializer=self.normc_initializer(1.0), activation=tf.nn.relu)(out)
                                                          

                act_dist = keras.layers.Dense(self.action_num, name='final' ,
                kernel_initializer=self.normc_initializer(1.0), activation=tf.nn.softmax)(out)
                self.action_model = keras.Model(inputs = states, outputs = act_dist)   
            self.act_dist_entropy = tf.math.reduce_mean(-tf.math.reduce_sum(tf.math.log(act_dist+1e-16)*act_dist, axis=1))
        return states,act_dist,values


    def get_a_v(self, ob):
        act_dist_real, values_real = self.sess.run([self.act_dist, self.values], feed_dict = {self.state: [ob]})
        try:
            act_idx = np.random.choice(self.action_idx, p = act_dist_real[0])
        except ValueError:
            print(act_dist_real)
        return self.action_space[act_idx], values_real[0][act_idx], act_idx

    def action_encode(self, act_idxs):
        ret = np.zeros(shape=(len(act_idxs), self.action_num))
        ret[np.arange(act_idxs.size), act_idxs] = 1
        return ret

    def get_param(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    def get_trainable_param(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

    # TODO
    # def neglogp(self, x):
    #     print('x is',x)
    #     return x*


    def normc_initializer(self, std=1.0, axis=0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
            out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
            return tf.constant(out)

        return _initializer
