#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf

# 打印值
initial = tf.constant(0.1, shape=[3,4])
sess = tf.Session()
# print(sess.run(initial))

filter = tf.constant([[1,
                       2],
                      [3,
                       4]],dtype=tf.float32)
print(sess.run(filter))
