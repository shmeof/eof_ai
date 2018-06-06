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

# ttt = tf.constant(0.1, shape=[5])
# init = tf.initialize_all_variables()
# with tf.Session() as sess:
#     sess.run(init)
#     print("===")
#     print(sess.run(ttt))
#     print("===")
#
# #case 2
# input = tf.Variable(tf.random_normal([1,3,3,5]))
# filter = tf.Variable(tf.random_normal([1,1,5,1]))
#
# op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
# #case 3
# input = tf.Variable(tf.random_normal([1,3,3,5]))
# filter = tf.Variable(tf.random_normal([3,3,5,1]))
#
# op3 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
# #case 4
# input = tf.Variable(tf.random_normal([1,5,5,5]))
# filter = tf.Variable(tf.random_normal([3,3,5,1]))
#
# op4 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
# #case 5
# input = tf.Variable(tf.random_normal([1,5,5,5]))
# filter = tf.Variable(tf.random_normal([3,3,5,1]))
#
# op5 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
# #case 6
# input = tf.Variable(tf.random_normal([1,5,5,5]))
# filter = tf.Variable(tf.random_normal([3,3,5,7]))
#
# op6 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
# #case 7
# input = tf.Variable(tf.random_normal([1,5,5,5]))
# filter = tf.Variable(tf.random_normal([3,3,5,7]))
#
# op7 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')
# #case 8
# input = tf.Variable(tf.random_normal([10,5,5,5]))
# filter = tf.Variable(tf.random_normal([3,3,5,7]))
#
# op8 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')
#
# init = tf.initialize_all_variables()
# with tf.Session() as sess:
#     sess.run(init)
#     print("case 2")
#     print(sess.run(op2))
#     print("case 3")
#     print(sess.run(op3))
#     print("case 4")
#     print(sess.run(op4))
#     print("case 5")
#     print(sess.run(op5))
#     print("case 6")
#     print(sess.run(op6))
#     print("case 7")
#     print(sess.run(op7))
#     print("case 8")
#     print(sess.run(op8))

# # 保存到文件
# v1 = tf.Variable(tf.constant(55), "v1")
# v2 = tf.Variable(tf.constant(66), "v2")
# init_op = tf.initialize_all_variables();
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init_op)
#     save_path = saver.save(sess, "./practise_tensorflow_tmp/model.ckpt")
#     print "Model saved in file: ", save_path
#
#     saver.restore(sess, "./practise_tensorflow_tmp/model.ckpt")
#     print "Model resotred"
#
#     print(sess.run(v1))
#     print(sess.run(v2))

# # 可视化
# import tensorflow as tf
# a = tf.constant([1.0,2.0,3.0],name='input1')
# b = tf.Variable(tf.random_uniform([3]),name='input2')
# add = tf.add_n([a,b],name='addOP')
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     writer = tf.summary.FileWriter("./tensorboard_temp/test", sess.graph)
#     print(sess.run(add))
# writer.close()

import tensorflow as tf

# 定义一个简单的计算图，实现向量加法的操作。
input1 = tf.constant([1.0, 2.0, 3.0], name = 'input111')
input2 = tf.Variable(tf.random_uniform([3]), name = 'input222')
output = tf.add_n([input1, input2], name = 'add')

# 生成一个写日志的writer，并将当前的tensorflow计算图写入日志。
# tensorflow提供了多种写日志文件的API
writer = tf.summary.FileWriter('./tensorboard_temp', tf.get_default_graph())
writer.close()
