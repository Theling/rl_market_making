import numpy as np
import tensorflow as tf

class QFunction():
    def __init__(
            self,
            sess,
            state_dim,
            action_dim,
            time_steps,
            # learning_rate,
            drop_off_rate,
            batch_size,
            num_actor_vars=0,
            hidden_size=32
    ):
        self.hidden_size = hidden_size
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        # self.learning_rate = learning_rate
        self.time_steps = time_steps
        self.drop_off_rate = drop_off_rate
        self.batch_size = batch_size
        # Create Qfunction
        self.out, self.state, self.action = self.create_qfunction_net()
        # Save param
        self.network_params = tf.trainable_variables()[num_actor_vars:]
        # Compute grads
        self.action_grads = tf.gradients(self.out, self.action)

    def create_qfunction_net(self):
        actions = tf.placeholder(shape=[self.batch_size,self.time_steps, self.a_dim], dtype=tf.float32)
        states = tf.placeholder(shape=(self.batch_size, self.time_steps, self.s_dim), dtype=tf.float32)
        inputs = tf.concat([actions,states],axis=2)
        # define cell
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, state_is_tuple=True)
        # define init_state
        init_state = cell.zero_state(self.batch_size, tf.float32)
        if self.drop_off_rate > 0:
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=self.drop_off_rate)
        outputs, _ = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state)
        outputs = tf.unstack(tf.transpose(outputs,[1,0,2]))[-1]
        Weights = tf.Variable(tf.truncated_normal([self.hidden_size, 1], mean=0., stddev=0.01))
        biases = tf.Variable(tf.constant(0.03, shape=[1]))
        outputs = tf.matmul(outputs, Weights) + biases

        return outputs, states,actions

        # return_actions = actions
        # return_states = states
        #
        #
        # with tf.variable_scope('rnn_cell'):
        #     W = tf.get_variable('W', [self.s_dim + self.a_size+self.a_dim, self.a_size])
        #     b = tf.get_variable('b', [self.a_size], initializer=tf.constant_initializer(0.0))
        # def add_layers(inputs, current_state,activation_function=None):
        #     with tf.variable_scope('rnn_cell'):
        #         Weights = tf.get_variable('W', [self.s_dim + self.a_size+self.a_dim, self.a_size])
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
        #     if activation_function is None:
        #         outputs = Wx_plus_b
        #     else:
        #         outputs = activation_function(Wx_plus_b)
        #
        #     return outputs
        #
        # # if norm:
        # #     #norm action
        # #     actions = tf.reshape(actions, shape=[None, self.a_dim])
        # #     actions = tf.layers.batch_normalization(
        # #         inputs = actions,
        # #         axis = 0,
        # #         training = is_traning
        # #     )
        # #     #norm state
        # #     states = tf.reshape(states,shape=[None, self.s_dim])
        # #     states = tf.layers.batch_normalization(
        # #         inputs = states,
        # #         axis = 0,
        # #         training = is_traning
        # #     )
        #
        # output = tf.zeros(tf.float32, [1, self.a_size])
        # inputs = tf.concat([states,actions],1)
        # counter = 0
        # for input in inputs:
        #     if counter < self.time_steps-1:
        #         output = add_layers(
        #             inputs=tf.reshape(input,shape = (1,self.s_dim+self.a_dim)),
        #             current_state=output,
        #             activation_function=tf.tanh
        #         )
        #     else:
        #         output = add_layers(
        #             inputs=tf.reshape(input, shape=(1, self.s_dim + self.a_dim)),
        #             current_state=output,
        #         )
        #
        # return return_states, return_actions, output

    # def train(self, inputs, action, predicted_q_value):
    #     return self.sess.run(self.optimize, feed_dict={
    #         self.inputs: inputs,
    #         self.action: action,
    #         self.predicted_q_value: predicted_q_value
    #     })

    def predict_tensor(self, obs_, action_):
        self.inputs = obs_
        self.action = action_
        return self.out, self.is_traning

    def predict(self, inputs, action, is_traning=False):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.is_traning: is_traning
        })

    def get_param(self):
        return self.network_params

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions,
            self.is_traning: False
        })


    def reset(self):
        pass





