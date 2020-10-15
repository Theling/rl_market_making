import numpy as np
import tensorflow as tf
class Policy():
    def __init__(
            self,
            sess, state_dim, action_dim, action_bound,
            learning_rate, tau, batch_size=1, num_var_before = 0,
            hidden_size = (32,32)
            ):
        self.hidden_size = hidden_size
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        #Create Policy
        self.inputs, self.out, self.scaled_out, self.is_traning = self.create_policy_net(True)
        print('action bound = ', self.action_bound)
        self.network_params = tf.trainable_variables()[num_var_before:]
        #Create the target Qfunction
        # self.target_inputs, self.target_out, self.target_scaled_out = self.create_policy_net(True)
        # self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # self.update_target_network_params = \
        #     [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
        #     + tf.multiply(self.target_network_params[i], 1. - self.tau))
        #         for i in range(len(self.target_network_params))]
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # # Optimization Op
        # self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
        #     apply_gradients(zip(self.actor_gradients, self.network_params))

        # self.num_trainable_vars = len(
        #     self.network_params) + len(self.target_network_params)



    def create_policy_net(self,norm):
        states = tf.placeholder(shape=(None,self.s_dim),dtype = tf.float32)
        return_states = states
        is_traning = tf.placeholder(dtype = tf.bool)
        def add_layers(inputs, in_size, out_size, activation_function=None, norm=True, stochastic=False):
            Weights = tf.Variable(tf.truncated_normal([in_size,out_size],mean=0. ,stddev=0.01))
            biases = tf.Variable(tf.constant(0.03,shape=[out_size]))
            Wx_plus_b =tf.add(tf.matmul(inputs,Weights),biases)
            #print(Wx_plus_b,type(Wx_plus_b))
            Wx_plus_b = tf.reshape(Wx_plus_b, [-1, out_size])
            # if norm:
            #      Wx_plus_b = tf.contrib.layers.batch_norm(
            #         inputs = Wx_plus_b,
            #         is_training = is_traning
            #     )
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)

            return outputs

        # if norm:
        #     #norm state
        #     # Weights = tf.Variable(1)
        #     # states = Weights*states
        #     # states = tf.reshape(states, shape=[None, self.s_dim])
        #     states = tf.layers.batch_normalization(
        #         inputs = states,
        #         axis = 0,
        #         training = is_traning
        #     )

        output = states

        for layer in self.hidden_size:
            output = add_layers(
                inputs = output,
                in_size = output.get_shape()[1].value,
                out_size = layer,
                activation_function=tf.nn.relu
            )

        output = add_layers(
            inputs = output,
            in_size = output.get_shape()[1].value,
            out_size = self.a_dim,
            activation_function=tf.nn.tanh
        )
        scaled_out = tf.multiply(output, self.action_bound)
        return return_states, output, scaled_out, is_traning



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

    # def clone(self, num):
    #     clone_policy = Policy(self.sess, self.s_dim, self.a_dim, self.action_bound,
    #         self.learning_rate, self.tau, self.batch_size,num,
    #         self.hidden_size,)
    #     self.policy = clone_policy
    #     update = [clone_policy.network_params[i].assign(self.network_params[i]) for i in range(len(self.network_params))]
    #     self.sess.run(update)
    #     clone_policy.update = [clone_policy.network_params[i].assign(self.network_params[i]) for i in range(len(self.network_params))]
    #     return clone_policy

    # def set_param(self,policy):
    #
    #
    #     self.sess.run(self.update)

    def get_param(self):
        return self.network_params

    def reset(self):
        pass
    # def predict_target(self, inputs):
    #     return self.sess.run(self.target_scaled_out, feed_dict={
    #         self.target_inputs: inputs
    #     })

    # def update_target_network(self):
    #     self.sess.run(self.update_target_network_params)

    # def get_num_trainable_vars(self):
    #     return self.num_trainable_vars