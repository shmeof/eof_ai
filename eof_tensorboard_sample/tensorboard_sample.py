#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import input_data

# 卷积层
def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        t_normal = tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1)
        w = tf.Variable(t_normal, name="W")
        t_const = tf.constant(0.1, shape=[size_out])
        b = tf.Variable(t_const, "B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)

        # 可视化
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)

        pool = tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        return pool

# 全连接层
def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        t_normal = tf.truncated_normal([size_in, size_out], stddev=0.1)
        w = tf.Variable(t_normal)
        t_const = tf.constant(0.1, shape=[size_out])
        b = tf.Variable(t_const)
        act = tf.nn.relu(tf.matmul(input, w) + b)
        return act

def mnist_fun(hparam_str, mnist_data, learning_rate, use_two_fc, use_two_conv, writer):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    conv = conv_layer(x_image, 1, 32, "conv1")
    if use_two_conv:
        conv = conv_layer(conv, 32, 64, "conv2")

    flattened = tf.reshape(conv, [-1, 7 * 7 * 64])

    if use_two_fc:
        fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
        logits = fc_layer(fc1, 1024, 10, "fc2")
    else:
        logits = fc_layer(flattened, 7 * 7 * 64, 10, "fc1")

    # 交叉熵
    with tf.name_scope("xent"):
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
        cross_entropy = tf.reduce_mean(softmax_cross_entropy)
    # 训练算法及步长
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # 计算精度
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # 可视化
    # writer = tf.summary.FileWriter("./tmp/mnist_demo/1")
    # writer = tf.summary.FileWriter("./tmp/mnist_demo/2")
    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('input', x_image, 3)
    merged_summary = tf.summary.merge_all()
    # writer = tf.summary.FileWriter("./tmp/mnist_demo/3")

    writer.add_graph(sess.graph)

    for i in range(200):
        batch = mnist_data.train.next_batch(100)

        # if i % 500 == 0:
        #     [train_accuracy] = sess.run([accuracy], feed_dict={x:batch[0], y:batch[1]})
        #     print("step %d, training accuracy %g" % (i, train_accuracy))

        if i % 5 == 0:
            s = sess.run(merged_summary, feed_dict={x:batch[0], y:batch[1]})
            writer.add_summary(s, i)
            print("step %d" % (i))
        sess.run(train_step, feed_dict={x:batch[0], y:batch[1]})

    writer.close()

def make_hparam_str(learning_rate, use_two_fc, use_two_conv):
    hp = str(learning_rate)
    if use_two_conv:
        hp += ",conv-2"
    else:
        hp += ",conv-1"
    if use_two_fc:
        hp += ",fc-2"
    else:
        hp += ",fc-1"
    return hp

# 对比
for learning_rate in [1E-3, 1E-4, 1E-5]:
    # fc层
    for use_two_fc in [True, False]:
        # conv层
        for use_two_conv in [True, False]:
            hparam_str = make_hparam_str(learning_rate, use_two_fc, use_two_conv)
            writer = tf.summary.FileWriter("./tmp/mnist_tutorial/" + hparam_str)
            mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
            mnist_fun(hparam_str, mnist, learning_rate, use_two_fc, use_two_conv, writer)