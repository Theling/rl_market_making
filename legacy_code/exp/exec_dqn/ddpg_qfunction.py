import numpy as np
import tensorflow as tf
n_hidden_1 = 400
n_hidden_2 = 300

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.03, shape=shape)
    return tf.Variable(initial)
class QFunction():
    def __init__(
            self,
            sess, state_dim, action_dim, 
            learning_rate, tau, gamma=0, num_actor_vars=0,
            hidden_size = (32,32), merge_layer = 1
            ):
        self.hidden_size = hidden_size
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.merge_layer = merge_layer
        self.gamma = gamma
         #Create Qfunction
        self.inputs, self.action, self.out, self.is_traning = self.create_qfunction_net(True)
        self.network_params = tf.trainable_variables()[num_actor_vars:]


        # #Create the target Qfunction
        # self.target_inputs, self.target_action, self.target_out = self.create_qfunction_net(True)
        # self.target_network_params = tf.trainable_variables()[len(self.network_params)+num_actor_vars:]

        # self.update_target_network_params = \
        #     [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
        #     + tf.multiply(self.target_network_params[i], 1. - self.tau))
        #         for i in range(len(self.target_network_params))]
        # self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # self.loss = tf.losses.mean_squared_error(self.predicted_q_value, self.out)
        # extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(extra_update_ops):
        #     self.optimize = tf.train.AdamOptimizer(
        #         self.learning_rate).minimize(self.loss)
        self.action_grads = tf.gradients(self.out, self.action)

    
    def create_qfunction_net(self, norm = False):
        actions = tf.placeholder(shape=[None,self.a_dim], dtype = tf.float32)
        states = tf.placeholder(shape=[None,self.s_dim],dtype = tf.float32)
        is_traning = tf.placeholder(dtype = tf.bool)
        return_actions = actions
        return_states = states
        def add_layers(inputs, in_size, out_size, activation_function=None, norm=False):
            Weights = tf.Variable(tf.truncated_normal([in_size, out_size], mean=0., stddev=0.01))
            biases = tf.Variable(tf.constant(0.03, shape = [out_size]))
            # W+b
            Wx_plus_b = tf.matmul(inputs, Weights) + biases


            # if norm:
            #     Wx_plus_b = tf.contrib.layers.batch_norm(
            #         inputs = Wx_plus_b,
            #         is_training = is_traning
            #     )
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)

            return outputs

        # if norm:
        #     #norm action
        #     actions = tf.reshape(actions, shape=[None, self.a_dim])
        #     actions = tf.layers.batch_normalization(
        #         inputs = actions,
        #         axis = 0,
        #         training = is_traning
        #     )
        #     #norm state
        #     states = tf.reshape(states,shape=[None, self.s_dim])
        #     states = tf.layers.batch_normalization(
        #         inputs = states,
        #         axis = 0,
        #         training = is_traning
        #     )

        output = states
        for index, layer in enumerate(self.hidden_size):

            output = add_layers(
                inputs = output,
                in_size = output.get_shape()[1].value,
                out_size = layer,
                activation_function=tf.nn.relu
            )
            if index == 0:
                Weights = tf.Variable(tf.truncated_normal([self.a_dim, layer], mean=0., stddev=0.01))
                output = tf.matmul(actions,Weights)+output

        output = add_layers(
            inputs = output,
            in_size = output.get_shape()[1].value,
            out_size = 1
        )
        return return_states, return_actions, output, is_traning

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
            self.is_traning:is_traning
        })

    # def clone(self,num):
    #     clone_qf = QFunction(self.sess, self.s_dim, self.a_dim,
    #                            self.learning_rate, self.tau, self.gamma, num,
    #                             self.hidden_size, self.merge_layer)
    #     update = [clone_qf.network_params[i].assign(self.network_params[i]) for i in range(len(self.network_params))]
    #     self.sess.run(update)
    #     clone_qf.update = [clone_qf.network_params[i].assign(self.network_params[i]) for i in range(len(self.network_params))]
    #     return clone_qf

    # def set_param(self,qfun):
    #
    #     self.sess.run(self.update)



    def get_param(self):
        return self.network_params
    # def predict_target(self, inputs, action):
    #     return self.sess.run(self.target_out, feed_dict={
    #         self.target_inputs: inputs,
    #         self.target_action: action
    #     })

    def action_gradients(self, inputs, actions):
        # print(inputs)
        # print(actions)
        # print(self.network_params)
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions,
            self.is_traning: False
        })

    # def update_target_network(self):
    #     self.sess.run(self.update_target_network_params)

    def reset(self):
        pass



    
        
