import tensorflow as tf

class LSTM():
    def __init__(self,
                 sess,
                 inputs,
                 timesteps,
                 num_input = 1,
                 num_hidden = 32,
                 out_size = 1,
                 ):
        self.sess = sess
        self.inputs = inputs
        self.timesteps = timesteps
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.out_size = out_size
        #self.batch_size = batch_size
        self.input, self.output = self.LSTM_net()

    def LSTM_net(self):
        # tf input
        inputs = tf.placeholder(shape=[None, self.timesteps, self.num_input], dtype=tf.float32)
        inputs_ = tf.unstack(inputs, self.timesteps, 1)

        # Define weights and biases
        weights = tf.Variable(tf.truncated_normal([self.num_hidden, self.out_size], mean=0., stddev=0.01))
        biases = tf.Variable(tf.constant(0.03, shape=[self.out_size]))

        # Define a lstm cell with tensorflow
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden, forget_bias=1.0)

        # state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        # outputs = []
        # for current_input in inputs:
        #     output, state = lstm_cell(current_input, state)
        #
        #     logits = tf.add(tf.matmul(output, weights), biases)
        #     outputs.append(logits)

        # Get the lstm cell output
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, inputs_, dtype=tf.float32)

        # activation of the output
        outputs = tf.add(tf.matmul(outputs[-1], weights), biases)
        outputs = tf.nn.tanh(outputs)

        return inputs, outputs

    def get_action(self, inputs):
        return self.sess.run(self.output, feed_dict={self.input: inputs})

    def get_action_tensor(self):
        return self.output


if __name__=='__main__':
    import numpy as np
    import pandas as pd

    timesteps = 3
    num_input = 1
    num_hidden = 128
    batch_size = 3950
    out_size = 1
    learning_rate = 0.01

    # input_data = np.sin(np.arange(0, 2500*np.pi, np.pi/2)).reshape(1000, 5)
    # #print(input_data)
    # label = np.zeros((1000, 1))
    # for i in range(1000):
    #     label[i] = -1*input_data[i, 4]
    #     if np.abs(label[i]) < 1 :
    #         if input_data[i, 3] > 0:
    #             label[i] = -0.5 + 0.1*np.random.random_sample()
    #         else:
    #             label[i] = 0.5 + 0.1 * np.random.random_sample()
    # #print(label)

    dataa = pd.read_csv('../data/test_lstm.csv', header=None)
    input_data = dataa.iloc[:-1, 1:4].values
    print(input_data.shape)
    label = dataa.iloc[:-1, 4].values
    print(label.shape)


    with tf.Session() as sess:

        inputs = tf.placeholder(shape=(None, timesteps, num_input), dtype=tf.float32)
        input_ = tf.unstack(inputs, timesteps, 1)
        targets = tf.placeholder(shape=(None, out_size), dtype=tf.float32)

        # Define weights and biases
        weights = tf.Variable(tf.truncated_normal([num_hidden, out_size], mean=0., stddev=0.01))
        biases = tf.Variable(tf.constant(0.03, shape=[out_size]))

        # Define a lstm cell with tensorflow
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0)

        # Get the lstm cell output
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, input_, dtype=tf.float32)

        # activation of the output
        outputs = tf.add(tf.matmul(outputs[-1], weights), biases)
        pred = tf.nn.relu(outputs)

        # loss
        loss_op = tf.reduce_mean(tf.square(pred-targets))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        minimize = optimizer.minimize(loss_op)


        # def tf_round(x, decimals=2):
        #     multiplier = tf.constant(10 ** decimals, dtype=x.dtype)
        #     return tf.round(x * multiplier) / multiplier
        correct_pred = tf.equal(tf.rint(pred), tf.rint(targets))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        print({1:2})

        sess.run(tf.global_variables_initializer())
        for i in range(int(138000/batch_size)):
            sample_x = input_data[i:i+batch_size, :].reshape((batch_size, timesteps, num_input))
            sample_y = label[i:i+batch_size].reshape(-1, 1)
            sess.run([minimize], feed_dict = {inputs: sample_x, targets: sample_y})
            loss, acc = sess.run([loss_op, accuracy], feed_dict={inputs: sample_x, targets: sample_y})
            print("step " + str(i+1) + ", loss: {:.4f}, accuracy: {:.4f}".format(loss, acc))
