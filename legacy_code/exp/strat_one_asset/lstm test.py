import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# data
state_dim = 100
action_dim = 1
num_ipt = 1
num_data = 100000
raw_data = np.random.normal(size = ( num_data, state_dim))
t_weight = 0.5
weight = np.random.normal(size=state_dim)
cons_weight = 0.1 * np.random.normal(size = num_ipt)

print(f'weight: {weight}, t_weight: {t_weight}, const: {cons_weight}')

seq = np.sum(raw_data[0:state_dim, 0:num_ipt].transpose()*weight)+cons_weight

gen_func = lambda s, x: sum(s*t_weight)+sum(x*weight)+0.1*np.random.normal()

for i in range(num_ipt, num_data):
    seq = np.append(seq, gen_func(seq[-num_ipt:], raw_data[i, :]))

plt.plot(range(len(seq)), seq)
plt.show()

time_len = 100

data_x_naive = raw_data[time_len:,:]
data_x = np.array([np.array(raw_data[(i-time_len):i]) for i in range(time_len+1, num_data+1)])
data_y = seq[time_len:]
data_y = np.reshape(data_y, (-1,1))

idx = np.arange(data_y.shape[0])
train_idx = np.random.choice(idx, size=7000, replace= False)
test_idx = np.array([i for i in idx
                     if i not in train_idx])

train_x = data_x[train_idx]
train_x_naive = data_x_naive[train_idx]
train_y = data_y[train_idx]

test_x = data_x[test_idx]
test_x_naive = data_x_naive[test_idx]
test_y = data_y[test_idx]


# Model
def create_network(ipt, c, hidden_units=(64, 64)):
    out = tf.concat([ipt, c], axis=1)
    for i, hidden in enumerate(hidden_units):
        out = tf.layers.dense(out, hidden, name=f'fc{i}', activation=tf.nn.relu,
                              kernel_initializer=tf.initializers.random_normal())
    ret = tf.layers.dense(out, 1, name='out',
                          kernel_initializer=tf.initializers.random_normal())
    return ret

# test block
tf.reset_default_graph()
sess = tf.Session()
input_ = tf.placeholder(shape= (None, state_dim), dtype=tf.float32)
input_c = tf.placeholder(shape = (None, 1), dtype=tf.float32)
label = tf.placeholder(shape = (None, 1), dtype=tf.float32)
with tf.variable_scope('dense') as sc:
    output_ = create_network(input_, input_c)
tloss = tf.reduce_mean(tf.square(label-output_))
optimizer = tf.train.AdamOptimizer(0.001)
op = optimizer.minimize(tloss)
                         #,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'dense'))

writer = tf.summary.FileWriter("/home/zhiyuan/log/...", sess.graph)
init = tf.global_variables_initializer()
sess.run(init)

for _ in range(1000):
    _, l1 = sess.run((op, tloss), feed_dict={input_: train_x_naive,
                                            input_c: np.array([[0.0] for _ in range(len(train_idx))]),
                                             label: train_y})
    l2 = sess.run(tloss, feed_dict={input_: test_x_naive,
                                            input_c: np.array([[0.0] for _ in range(len(test_idx))]),
                                             label: test_y})
    print(l1, l2)
    # result = sess.run(output_, feed_dict={input_: data_x_naive})
    # print(result)


# RNN
def create_RNN(inputs, init_c):
    ret = []
    out = init_c

    with tf.variable_scope('RNN_cell') as sc:
        for ipt in inputs:
            out = create_network(ipt, out, hidden_units=(64, 64))
            sc.reuse_variables()
            ret.append(out)
    return ret[-1]


tf.reset_default_graph()

sess = tf.Session()
input_ = tf.placeholder(shape=[ None, time_len, state_dim], dtype=tf.float32)
input_c = tf.placeholder(shape = [None, 1], dtype = tf.float32)
unstacked_input = tf.unstack(input_, axis=1)
out = create_RNN(unstacked_input, input_c)
label = tf.placeholder(shape = (None, 1), dtype=tf.float32)
tloss = tf.reduce_mean(tf.square(label-out))
optimizer = tf.train.AdamOptimizer(0.1)
op = optimizer.minimize(tloss)
                         #,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'dense'))

writer = tf.summary.FileWriter("/home/zhiyuan/log/...", sess.graph)
init = tf.global_variables_initializer()
sess.run(init)
for _ in range(1000):
    _, l1 = sess.run((op, tloss), feed_dict={input_: train_x,
                                            input_c: np.array([[0.0] for _ in range(len(train_idx))]),
                                             label: train_y})
    l2 = sess.run(tloss, feed_dict={input_: test_x,
                                            input_c: np.array([[0.0] for _ in range(len(test_idx))]),
                                             label: test_y})
    print(l1, l2)
    # result = sess.run(output_, feed_dict={input_: data_x_naive})
    # print(result)

# LSTM

def create_LSTM(inputs, init_c, lstm):
    ret = []
    state = init_c
    for ipt in inputs:
        out, state = lstm(ipt, state)
        ret.append(out)
    return ret

tf.reset_default_graph()

sess = tf.Session()
lstm = tf.nn.rnn_cell.GRUCell(64)
input_ = tf.placeholder(shape=[ None, time_len, state_dim], dtype=tf.float32)
input_c = lstm.zero_state(tf.shape(input_)[0], dtype=tf.float32)
# input_c = tf.placeholder(shape = (None, 1), dtype=tf.float32)
unstacked_input = tf.unstack(input_, axis=1)
# out, _ = tf.contrib.rnn.static_rnn(lstm, unstacked_input, dtype = tf.float32)
out = create_LSTM(unstacked_input, input_c, lstm)
pre = tf.layers.dense(out[-1], 1, name = 'final',
                              kernel_initializer=tf.initializers.random_normal())
#
label = tf.placeholder(shape = (None, 1), dtype=tf.float32)
tloss = tf.reduce_mean(tf.square(label-pre))
optimizer = tf.train.AdamOptimizer(0.001)
op = optimizer.minimize(tloss)
                         #,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'dense'))

writer = tf.summary.FileWriter("/home/zhiyuan/log/...", sess.graph)
init = tf.global_variables_initializer()
sess.run(init)

for _ in range(1000):
    _, l1 = sess.run((op, tloss), feed_dict={input_: train_x[0:1000],
                                            # input_c: np.array([[0.0] for _ in range(64)]),
                                             label: train_y[0:1000]})
    l2 = sess.run(tloss, feed_dict={input_: test_x[0:1000],
                                            # input_c: np.array([[0.0] for _ in range(64)]),
                                             label: test_y[0:1000]})
    print(l1, l2)




