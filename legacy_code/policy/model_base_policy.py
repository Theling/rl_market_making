import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
# def linear(args, output_size, dtype=tf.float32, scope=None):
#     with tf.variable_scope(scope or "linear"):
#         if isinstance(args, list) or isinstance(args, tuple):
#             if len(args) != 1:
#                 inputs = tf.concat(args, axis=1)
#             else:
#                 inputs = args[0]
#         else:
#             inputs = args
#             args = [args]
#         total_arg_size = 0
#         shapes = [a.get_shape() for a in args]
#         for shape in shapes:
#             if shape.ndims != 2:
#                 raise ValueError("linear is expecting 2D arguments: %s" % shapes)
#             else:
#                 total_arg_size += shape[1].value
#         dtype = args[0].dtype
#         weights = tf.get_variable(
#             "weights",
#             [total_arg_size, output_size],
#             dtype=dtype,
#             initializer=tf.contrib.layers.xavier_initializer(dtype=dtype))
#         output = tf.matmul(inputs, weights)
#     return output
#
# def multiplicative_integration(
#             list_of_inputs,
#             output_size,
#             initial_bias_value=0.0,
#             weights_already_calculated=False,
#             reg_collection=None,
#             dtype=tf.float32,
#             scope=None):
#     '''
#     expects len(2) for list of inputs and will perform integrative multiplication
#     weights_already_calculated will treat the list of inputs as Wx and Uz and is useful for batch normed inputs
#     '''
#     with tf.variable_scope(scope or 'double_inputs_multiple_integration'):
#         if len(list_of_inputs) != 2: raise ValueError('list of inputs must be 2, you have:', len(list_of_inputs))
#
#         # TODO can do batch norm in FC
#         if weights_already_calculated:  # if you already have weights you want to insert from batch norm
#             Wx = list_of_inputs[0]
#             Uz = list_of_inputs[1]
#
#         else:
#             Wx = linear(
#                 list_of_inputs[0],
#                 output_size,
#                 dtype=dtype,
#                 reg_collection=reg_collection,
#                 scope="Calculate_Wx_mulint")
#
#             Uz = linear(
#                 list_of_inputs[1],
#                 output_size,
#                 dtype=dtype,
#                 reg_collection=reg_collection,
#                 scope="Calculate_Uz_mulint")
#
#         with tf.variable_scope("multiplicative_integration"):
#             alpha = tf.get_variable(
#                 'mulint_alpha',
#                 [output_size],
#                 dtype=dtype,
#                 initializer=tf.truncated_normal_initializer(
#                     mean=1.0,
#                     stddev=0.1,
#                     dtype=dtype))
#
#             beta1, beta2 = tf.split(
#                 tf.get_variable(
#                     'mulint_params_betas',
#                     [output_size * 2],
#                     dtype=dtype,
#                     initializer=tf.truncated_normal_initializer(
#                         mean=0.5,
#                         stddev=0.1,
#                         dtype=dtype)),
#                 2,
#                 axis=0)
#
#             original_bias = tf.get_variable(
#                 'mulint_original_bias',
#                 [output_size],
#                 dtype=dtype,
#                 initializer=tf.truncated_normal_initializer(
#                     mean=initial_bias_value,
#                     stddev=0.1,
#                     dtype=dtype))
#
#         final_output = alpha * Wx * Uz + beta1 * Uz + beta2 * Wx + original_bias
#
#     return final_output
#
# class DpRNNCell(tf.nn.rnn_cell.BasicRNNCell):
#     def __init__(
#             self,
#             num_units,
#             dropout_mask=None,
#             activation=tf.tanh,
#             dtype=tf.float32,
#             num_inputs=None,
#             weights_scope=None):
#         self._num_units = num_units
#         self._dropout_mask = dropout_mask
#         self._activation = activation
#         self._dtype = dtype
#
#         with tf.variable_scope(weights_scope or type(self).__name__):
#             self._weights = tf.get_variable(
#                 "weights",
#                 [num_inputs + num_units, num_units],
#                 dtype=dtype,
#                 initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
#                 regularizer=tf.contrib.layers.l2_regularizer(0.5))
#
#     def __call__(
#             self,
#             inputs,
#             state,
#             scope=None):
#         """Most basic RNN: output = new_state = tanh(W * input + U * state + B). With same dropout at every time step."""
#         with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
#
#             ins = tf.concat([inputs, state], axis=1)
#             output = self._activation(tf.matmul(ins, self._weights))
#
#             if self._dropout_mask is not None:
#                 output = output * self._dropout_mask
#
#         return output, output
#
#
# class DpMulintRNNCell(DpRNNCell):
#     def __init__(
#             self,
#             num_units,
#             dropout_mask=None,
#             activation=tf.tanh,
#             dtype=tf.float32,
#             num_inputs=None,
#             use_layer_norm=False,
#             weights_scope=None):
#
#         self._num_units = num_units
#         self._dropout_mask = dropout_mask
#         self._activation = activation
#         self._dtype = dtype
#         self._use_layer_norm = use_layer_norm
#
#         with tf.variable_scope(weights_scope or type(self).__name__):
#             self._weights_W = tf.get_variable(
#                 "weights_W",
#                 [num_inputs, num_units],
#                 dtype=dtype,
#                 initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
#                 regularizer=tf.contrib.layers.l2_regularizer(0.5))
#
#             self._weights_U = tf.get_variable(
#                 "weights_U",
#                 [num_units, num_units],
#                 dtype=dtype,
#                 initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
#                 regularizer=tf.contrib.layers.l2_regularizer(0.5))
#
#     def __call__(
#             self,
#             inputs,
#             state,
#             scope=None):
#         """Most basic RNN: output = new_state = tanh(W * input + U * state + B)."""
#         with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
#             Wx = tf.matmul(inputs, self._weights_W)
#             Uz = tf.matmul(state, self._weights_U)
#             if self._use_layer_norm:
#                 Wx = tf.contrib.layers.layer_norm(
#                     Wx,
#                     center=False,
#                     scale=False)
#                 Uz = tf.contrib.layers.layer_norm(
#                     Uz,
#                     center=False,
#                     scale=False)
#             output = self._activation(
#                 multiplicative_integration(
#                     [Wx, Uz],
#                     self._num_units,
#                     dtype=self._dtype,
#                     weights_already_calculated=True))
#
#             if self._dropout_mask is not None:
#                 output = output * self._dropout_mask
#
#         return output, output

class Policy():
    def __init__(
            self,
            sess,
            state_dim,
            action_dim,
            time_steps,
            # batch_size,
            # drop_off_rate,
            gamma,
            K,
            scope,
            num_var_before = 0,
            hidden_size = 32,
            num_cell = 1,
            reuse = False
            ):
        self.hidden_size = hidden_size
        self.sess = sess
        self.K = K
        self.gamma = gamma
        self.s_dim = state_dim
        self.a_dim = action_dim
        # self.drop_off_rate = drop_off_rate
        # self.batch_size = batch_size
        self.num_cell = num_cell
        self.time_steps = time_steps
        self.scope = scope
        self.reuse = reuse
        self.frame = 1
        # Create Policy
        with tf.variable_scope(self.scope):
            self.scalarInput, low_dim_obs = self.cnn(True)
        with tf.variable_scope(self.scope):
            self.actions_ph = tf.placeholder(tf.float32, [None, self.time_steps, action_dim], name='tf_actions_ph')
            self.rewards, self.values = self.create_policy_net(low_dim_obs,self.actions_ph,True)

        tf_nstep_rewards = tf.unstack(tf.reshape(self.rewards, (-1, self.time_steps)), axis=1)
        tf_nstep_values = tf.unstack(tf.reshape(self.values, (-1, self.time_steps)), axis=1)
        self.J = self._graph_calculate_value(self.time_steps-2, tf_nstep_rewards, tf_nstep_values)
        self.action, self.max_val = self.get_action_random(low_dim_obs,True)


    def cnn(self, is_traning):
        scalarInput = tf.placeholder(shape=[None]+list(self.s_dim), dtype=tf.float32)
        # imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 4])
        conv1 = slim.conv2d(inputs=scalarInput, num_outputs=32, kernel_size=[8, 8], stride=[4, 4],
                            padding='VALID', biases_initializer=None)
        conv2 = slim.conv2d(inputs=conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding='VALID',
                            biases_initializer=None)
        conv3 = slim.conv2d(inputs=conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='VALID',
                            biases_initializer=None)
        output = tf.reshape(conv3,shape=(-1,3136))
        output = self.fcnn(output,is_traning,'obs_fcnn',num_output=128)
        return scalarInput,output

    def fcnn(self,next_layer_input,is_training,scope,reuse=False,num_output = 16):
        with tf.variable_scope(scope, reuse=reuse):
            normalizer_fn = tf.contrib.layers.batch_norm
            normalizer_params = {
                'is_training': is_training,
                'data_format': 'NHWC',
                'fused': True,
                'decay': 0.9,
                'zero_debias_moving_mean': True,
                'scale': True,
                'center': True,
                'updates_collections': None
            }
            output = tf.contrib.layers.fully_connected(
                inputs=next_layer_input,
                num_outputs=num_output,
                activation_fn=tf.nn.relu,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params,
                weights_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                biases_initializer=tf.constant_initializer(0., dtype=tf.float32),
                weights_regularizer=tf.contrib.layers.l2_regularizer(0.5),
                trainable=True)
            return output

    def rnn(self,inputs,initial_state,scope = 'rnn', reuse = False):
        with tf.variable_scope(scope, reuse=reuse):
            cell_type = tf.contrib.rnn.LayerNormBasicLSTMCell
            # print(initial_state.shape)
            states = tf.split(initial_state, 2 * self.num_cell, axis=1)
            # print('state',initial_state.shape)
            # print('input',inputs.shape)
            # print(states)
            num_units = states[0].get_shape()[1].value
            initial_state = []
            num_inputs = inputs.get_shape()[-1]
            for i in range(self.num_cell):
                initial_state.append(tf.nn.rnn_cell.LSTMStateTuple(states[i * 2], states[i * 2 + 1]))
            initial_state = tuple(initial_state)
            print(initial_state)
            cells = []
            for i in range(self.num_cell):
                if i == 0:
                    num_inputs = inputs.get_shape()[-1]
                else:
                    num_inputs = num_units
                cell = cell_type(num_units)
                cells.append(cell)
            multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            # a = tf.placeholder(tf.int32,[1])
            # init_state = multi_cell.zero_state(a, dtype=tf.float32)
            # print(init_state)
            rnn_outputs, state = tf.nn.dynamic_rnn(
                multi_cell,
                inputs=tf.cast(inputs, tf.float32),
                initial_state=initial_state,
                dtype=tf.float32,
                time_major=False)
            return rnn_outputs

    def create_policy_net(self,low_dim_obs, action_ph, is_traning):
        #action graph
        # num_obs = tf.shape(low_dim_obs)[0]
        # action = tf.one_hot(tf.random_uniform([100, 16], minval=0, maxval=self.a_dim, dtype=tf.int32),
        #                         depth=self.a_dim,
        #                         axis=2)
        # actions = tf.tile(action, (num_obs, 1, 1))
        # # actions = tf.reshape(action, (-1, self.a_dim))
        # next_layer_input = actions
        # action_graph_outputs=self.fcnn(next_layer_input, is_traning)
        # H = action_ph.get_shape()[1].value
        actions = tf.reshape(action_ph,(-1,self.a_dim))
        action_graph_outputs = self.fcnn(actions,is_traning, scope = 'fcnn_actions')

        #rnn graph
        rnn_inputs = tf.reshape(action_graph_outputs, (-1, self.time_steps, 16))
        initial_state = low_dim_obs
        rnn_outputs = self.rnn(rnn_inputs,initial_state)


        #output graph
        rnn_output_dim = rnn_outputs.get_shape()[2].value
        # print(rnn_outputs.shape)
        rnn_outputs = tf.reshape(rnn_outputs, (-1, rnn_output_dim))
        rewards = self.fcnn(rnn_outputs,is_traning, scope = 'fcnn_values1')
        values = self.fcnn(rnn_outputs,is_traning, scope = 'fcnn_reward1')
        rewards = self.fcnn(rewards, is_traning, scope='fcnn_values2',num_output=1)
        values = self.fcnn(values, is_traning, scope='fcnn_reward2',num_output=1)
        # print(type(values))
        values = tf.reshape(values,shape=(-1,self.time_steps))
        rewards = tf.reshape(rewards,shape=(-1,self.time_steps))
        return rewards, values
        # tf_nstep_rewards = tf.unstack(tf.reshape(rewards, (-1, self.time_steps)), axis=1)
        # tf_nstep_values = tf.unstack(tf.reshape(values, (-1, self.time_steps)), axis=1)

        # return scalarInput, rewards, values

    def get_action_random(self,obs_low_dim,reuse):
        H = self.time_steps
        num_obs = tf.shape(obs_low_dim)[0]
        tf_actions = tf.one_hot(tf.random_uniform([self.K, H], minval=0, maxval=self.a_dim, dtype=tf.int32),
                                depth=self.a_dim,
                                axis=2)
        # print('action shape:',tf_actions.shape)
        tf_actions = tf.tile(tf_actions, (num_obs, 1, 1))
        obs_low_dim = repeat_2d(obs_low_dim, self.K, 0)
        with tf.variable_scope(self.scope,reuse=reuse):
            values,rewards = self.create_policy_net(obs_low_dim,tf_actions,False)
        tf_nstep_rewards = tf.unstack(tf.reshape(rewards, (-1, self.time_steps)), axis=1)
        tf_nstep_values = tf.unstack(tf.reshape(values, (-1, self.time_steps)), axis=1)

        tf_values_list = [self._graph_calculate_value(h, tf_nstep_rewards, tf_nstep_values) for h in
                          range(self.time_steps)]
        ###################################################
        # tf_values_list += [tf_values_list[-1]] * (N - H)#
        ###################################################
        tf_values = tf.stack(tf_values_list, 1)
        max_values = tf.reshape(tf_values_list[-1],(num_obs,self.K))
        max_value = tf.reduce_max(max_values,1)
        # print('max value:',max_values.shape)
        tf_values_softmax = (1. / float(self.time_steps)) * tf.ones([tf.shape(obs_low_dim)[0], self.time_steps])

        # get action#
        tf_values_select = tf.reduce_sum(tf_values * tf_values_softmax, reduction_indices=1)
        tf_values_select = tf.reshape(tf_values_select, (num_obs, self.K))  # [num_obs, K]
        tf_values_argmax_select = tf.one_hot(tf.argmax(tf_values_select, 1), depth=self.K)  # [num_obs, K]
        tf_get_action = tf.reduce_sum(
            tf.tile(tf.expand_dims(tf_values_argmax_select, 2), (1, 1, self.a_dim)) *
            tf.reshape(tf_actions, (num_obs, self.K, self.time_steps, self.a_dim))[:, :, 0, :],
            reduction_indices=1)  # [num_obs, action_dim]
        return tf_get_action, max_value

    def get_action(self, input):
        action = self.sess.run(self.action,feed_dict={
            self.scalarInput:input
        })
        return action

    def max_value(self, input):
        maxval = self.sess.run(self.max_val,feed_dict={
            self.scalarInput:input
        })
        return maxval

    @property
    def network_params(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

    def _graph_calculate_value(self, n, tf_nstep_rewards, tf_nstep_values):
        tf_returns = tf_nstep_rewards[:n] + [tf_nstep_values[n]]
        tf_value = np.sum([np.power(self.gamma, i) * tf_return for i, tf_return in enumerate(tf_returns)])
        return tf_value

    def reset(self):
        pass

def repeat_2d(x, reps, axis):
    assert(axis == 0 or axis == 1)

    if axis == 1:
        x = tf.transpose(x)

    static_shape = list(x.get_shape())
    dyn_shape = tf.shape(x)
    x_repeat = tf.reshape(tf.tile(x, [1, reps]), (dyn_shape[0] * reps, dyn_shape[1]))
    if static_shape[0].value is not None:
        static_shape[0] = tf.Dimension(static_shape[0].value *reps)
    x_repeat.set_shape(static_shape)

    if axis == 1:
        x_repeat = tf.transpose(x_repeat)

    return x_repeat