import numpy as np
import tensorflow as tf
class Policy():
    def __init__(
            self,
            sess,
            state_dim,
            action_dim,
            action_bound,
            time_steps,
            batch_size,
            # learning_rate,
            drop_off_rate,
            num_var_before = 0,
            hidden_size = 32
            ):
        self.hidden_size = hidden_size
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        # self.learning_rate = learning_rate
        self.drop_off_rate = drop_off_rate
        self.batch_size = batch_size

        self.time_steps = time_steps
        #Create Policy
        self.inputs, self.out, self.scaled_out, self.is_traning = self.create_policy_net(True)
        self.network_params = tf.trainable_variables()[num_var_before:]

        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

    def create_policy_net(self,norm):
        states = tf.placeholder(shape=(self.batch_size,self.time_steps,self.s_dim),dtype=tf.float32)
        cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True)
        init_state = cell.zero_state(self.batch_size, tf.float32)
        if self.drop_off_rate  > 0:
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=self.drop_off_rate)
        outputs,_ = tf.nn.dynamic_rnn(cell, states, initial_state=init_state)
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))[-1]
        Weights = tf.Variable(tf.truncated_normal([self.hidden_size, 1], mean=0., stddev=0.01))
        biases = tf.Variable(tf.constant(0.03, shape=[1]))
        outputs = tf.matmul(outputs,Weights)+biases
        outputs = tf.nn.tanh(outputs)
        scaled_out = tf.multiply(outputs, self.action_bound)
        return outputs,scaled_out,states

        # states = tf.placeholder(shape=(self.time_steps,self.s_dim),dtype = tf.float32)
        # return_states = states
        # with tf.variable_scope('rnn_cell'):
        #     W = tf.get_variable('W', [self.s_dim + self.a_size, self.a_size])
        #     b = tf.get_variable('b', [self.a_size], initializer=tf.constant_initializer(0.0))
        # def add_layers(inputs,current_state):
        #
        #     with tf.variable_scope('rnn_cell'):
        #         Weights = tf.get_variable('W', [self.s_dim + self.a_size, self.a_size])
        #         biases = tf.get_variable('b', [self.a_size], initializer=tf.constant_initializer(0.0))
        #
        #     Wx_plus_b =tf.matmul(tf.concat([inputs,current_state],1),Weights)+biases
        #
        #     #print(Wx_plus_b,type(Wx_plus_b))
        #     # Wx_plus_b = tf.reshape(Wx_plus_b, [-1, out_size])
        #     # if norm:
        #     #      Wx_plus_b = tf.contrib.layers.batch_norm(
        #     #         inputs = Wx_plus_b,
        #     #         is_training = is_traning
        #     #     )
        #     outputs = tf.tanh(Wx_plus_b)
        #
        #     return outputs
        #
        # # if norm:
        # #     #norm state
        # #     # Weights = tf.Variable(1)
        # #     # states = Weights*states
        # #     # states = tf.reshape(states, shape=[None, self.s_dim])
        # #     states = tf.layers.batch_normalization(
        # #         inputs = states,
        # #         axis = 0,
        # #         training = is_traning
        # #     )
        #
        # output = tf.zeros(tf.float32,[1,self.a_size])
        #
        # for time_stpe in states:
        #     output = add_layers(
        #         inputs = tf.reshape(time_stpe,shape = (1,self.s_dim)),
        #         current_state=output
        #     )
        #
        # scaled_out = tf.multiply(output, self.action_bound)
        # return return_states, output, scaled_out



    # def train(self, inputs, a_gradient):
    #     self.sess.run(self.optimize, feed_dict={
    #         self.inputs: inputs,
    #         self.action_gradient: a_gradient
    #     })

    def get_action(self, inputs, is_traning=False):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs,
            self.is_traning: is_traning
        })

    def get_action_tensor(self,obs_):
        self.inputs = obs_
        return self.scaled_out

    def get_param(self):
        return self.network_params

    def reset(self):
        pass
