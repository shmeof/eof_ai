#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf

# # 打印值
# initial = tf.constant(0.1, shape=[3,4])
# sess = tf.Session()
# # print(sess.run(initial))
#
# filter = tf.constant([[1,
#                        2],
#                       [3,
#                        4]],dtype=tf.float32)
# print(sess.run(filter))

# # 打印值
# # Initialize session
# import tensorflow as tf
# sess = tf.InteractiveSession()
# # Some tensor we want to print the value of
# # a = tf.constant([1.0, 3.0])
# a = tf.zeros([10])
# # Add print operation
# a = tf.Print(a, [a], message="This is a: ")
# # Add more elements of the graph using a
# b = tf.add(a, a).eval()

ttt = tf.constant(0.1, shape=[5])
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print("===")
    print(sess.run(ttt))
    print("===")

#case 2
input = tf.Variable(tf.random_normal([1,3,3,5]))
filter = tf.Variable(tf.random_normal([1,1,5,1]))

op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
#case 3
input = tf.Variable(tf.random_normal([1,3,3,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))

op3 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
#case 4
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))

op4 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
#case 5
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))

op5 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
#case 6
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,7]))

op6 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
#case 7
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,7]))

op7 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')
#case 8
input = tf.Variable(tf.random_normal([10,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,7]))

op8 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print("case 2")
    print(sess.run(op2))
    print("case 3")
    print(sess.run(op3))
    print("case 4")
    print(sess.run(op4))
    print("case 5")
    print(sess.run(op5))
    print("case 6")
    print(sess.run(op6))
    print("case 7")
    print(sess.run(op7))
    print("case 8")
    print(sess.run(op8))