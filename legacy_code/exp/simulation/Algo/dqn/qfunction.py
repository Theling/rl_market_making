import numpy as np
import tensorflow as tf
from tensorflow import keras
# print(tf.__version__)



class QFunction():
    def __init__(self,
                 sess,
                 state_dim,
                 action_dim,
                 action_num,
                 name,
                 hidden_units,
                 ):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_num = action_num
        self.scope = name
        self.hidden_units = hidden_units
        self.state, self.out = self._create_network()
        # self.mu = 0.1 * (tf.nn.tanh(self.mu)*0.5+0.5)+0.01 # shift the output to [0, 0.1]+0.01
        # self.sigma = 0.005 * self.sigma
        # x = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu))
        # self.action = tf.clip_by_value(x, 0.01, 1)

        self.recurrent = False

    def _create_network(self):
        with tf.variable_scope(self.scope):
            states = keras.Input(name='ob',dtype=tf.float32,shape=(self.state_dim,))
            out = states

            for i, hidden in enumerate(self.hidden_units):
                out = keras.layers.Dense(hidden, name="fc%i" % (i + 1), activation=tf.nn.relu,
                                                        kernel_initializer=self.normc_initializer(1.0))(out)
            action_value = keras.layers.Dense(self.action_num, name='final', kernel_initializer=self.normc_initializer(1.0))(out)
            self.model = keras.Model(inputs = states, outputs = action_value)
        return states, action_value

    def get_action(self, ob):
        qpred =  self.sess.run((self.out), feed_dict={
            self.state: [ob]
        })
        
        # print(self.model.summary())
        act_idx = np.argmax(qpred[0])
        return act_idx, qpred[0]

    def predict(self, ob):
        qpred =  self.sess.run((self.out), feed_dict={
            self.state: ob
        })
 
        return qpred

    @property
    def get_params(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    @property
    def get_trainable_params(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)




    def normc_initializer(self, std=1.0, axis=0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
            out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
            return tf.constant(out)

        return _initializer


if __name__ == "__main__":
    sess = tf.Session()
    qf = QFunction(sess = sess,
                   state_dim = 5,
                   action_dim = 1,
                   action_num = 10,
                   name = 'qf',
                   hidden_units = (64,64))

    

    init = tf.global_variables_initializer()
    sess.run(init)
    print(qf.get_action(np.random.randn(5)))

    print(qf.model.inputs, qf.model.outputs)

   
