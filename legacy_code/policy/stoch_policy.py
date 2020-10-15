import tensorflow as tf
import numpy as np
class Policy():
    def __init__(self,
                 sess,
                 state_dim,
                 action_dim,
                 action_bound,
                 name,
                 hidden_units,
                 ):
        self.sess = sess
        self.action_bound=action_bound
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.scope = name
        self.hidden_units = hidden_units
        self.state,self.mu, self.sigma, self.value = self.create_network()
        self.shape = tf.placeholder(shape=(2),dtype=tf.int32)
        self.action = self.mu + tf.random_normal([1,self.action_dim])*self.sigma

    def create_network(self):
        with tf.variable_scope(self.scope):
            states = tf.placeholder(shape=(None, self.state_dim), dtype=tf.float32)
            states = self._normalize_with_moments(states, 1)
            out = states
            for i, hidden in enumerate(self.hidden_units):
                Weights = tf.Variable(tf.truncated_normal([out.shape[1].value, hidden], mean=0., stddev=0.01),
                                      name='layer{}_weight'.format(i))
                biases = tf.Variable(tf.constant(0.03, shape=[hidden]),name='layer{}_biase'.format(i))
                out = tf.add(tf.matmul(out, Weights), biases)
                out = tf.nn.tanh(out)

            ##### mu of distribution ####
            Weights = tf.Variable(tf.truncated_normal([out.shape[1].value, self.action_dim], mean=0., stddev=0.01),
                                  name='mu_weight')
            biases = tf.Variable(tf.constant(0.03,shape = [self.action_dim]),name='mu_biase')
            mu = tf.add(tf.matmul(out,Weights),biases)
            mu = tf.nn.tanh(mu + 1e-5)*self.action_bound
            #### sigma of distribution ####
            Weights = tf.Variable(tf.truncated_normal([out.shape[1].value,self.action_dim],mean = 0, stddev=0.01),
                                  name='sigma_weight')
            biases = tf.Variable(tf.constant(0.03,shape = [self.action_dim]),name='sigma_weight')
            sigma = tf.add(tf.matmul(out,Weights),biases)
            sigma = tf.nn.softplus(sigma + 1e-5)
            # sigma = tf.log(tf.exp(sigma)+1)
            # sigma = tf.nn.relu(sigma)
            sigma = sigma+1e-5
            # dist = tf.distributions.Normal(mu, sigma)
            # queation
            # action = tf.squeeze(dist.sample(self.action_dim))[0]

            #### value ####

            Weights = tf.Variable(tf.truncated_normal([out.shape[1].value,1],mean=0,stddev=0.01),
                                   name = 'value_weight')
            biases = tf.Variable(tf.constant(0.03,shape=[1]),name='value_weight')
            value = tf.add(tf.matmul(out,Weights),biases)

            return states,mu,sigma,value

    def get_a_v(self, state):

        return self.sess.run((self.action,self.value), feed_dict={
            self.state: state
        })

    def get_param(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

    def get_distribution(self):
        return self.dist

    def get_entropy(self):
        return self.dist.entropy()

    def normal(self,x):
        x = tf.clip_by_value(x,self.mu-3.*self.sigma,self.mu+3.*self.sigma)
        re = (1./tf.sqrt(2.*np.pi*tf.pow(self.sigma,2.)))*tf.exp(-tf.pow(x-self.mu,2.)/(2.*tf.pow(self.sigma,2.)))
        return re

    def _normalize_with_moments(x, axes, epsilon=1e-8):
        mean, variance = tf.nn.moments(x, axes=axes)
        x_normed = (x - mean) / tf.sqrt(variance + epsilon)  # epsilon to avoid dividing by zero
        x_normed = tf.clip_by_value(x_normed, -5.0, 5.0)
        return x_normed



