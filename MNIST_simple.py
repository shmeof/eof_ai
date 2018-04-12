#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x = [?, 784]：训练集的像素行
x = tf.placeholder("float", [None, 784])
# 想训练出来的每个像素分别在10个结果上的贡献权重
W = tf.Variable(tf.zeros([784,10]))
# 偏置值
b = tf.Variable(tf.zeros([10]))

# y = [?, 10]：对训练集预测的概率分布
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 交叉熵
# y_ = [? , 10]：训练集的在10个结果上的概率分布
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(10):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # print("------------")
    # print("batch_xs")
    # print(batch_xs)
    # print("batch_ys")
    # print(batch_ys)
    # print("train_step")
    # print(train_step)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
print("correct_prediction")
print(correct_prediction)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
