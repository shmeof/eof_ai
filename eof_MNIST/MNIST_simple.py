#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x = [?, 784]：训练集的像素行
x = tf.placeholder("float", [None, 784])
# 想训练出来的每个像素分别在10个结果上的贡献权重
W = tf.Variable(tf.zeros([784,10]))
# 偏置值 [1, 10]
b = tf.Variable(tf.zeros([10]))

# y = [?, 10]：对训练集预测的概率预测分布
y = tf.nn.softmax(tf.matmul(x, W) + b)

# y_ = [? , 10]：训练集的在10个结果上的概率真实分布
y_ = tf.placeholder("float", [None,10])
# 交叉熵，y是非真实分布
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 用最速下降法让交叉熵下降，步长为0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 初始化参数
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "==="
b = tf.Print(b, [b], message="This is a: ")


for i in range(10):
    # 加载100个训练样本，然后执行一次train_step，并通过feed_dict将x 和 y_张量占位符用训练训练数据替代
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
print (correct_prediction)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
