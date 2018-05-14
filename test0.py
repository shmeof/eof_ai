#!/usr/bin/python
# -*- coding: UTF-8 -*-

# import tensorflow as tf
#
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
