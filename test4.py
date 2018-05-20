#!/usr/bin/python
# -*- coding: UTF-8 -*-

import random
import numpy as np

sizes = [2, 3, 1]
biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

print "biases:", biases
print "weights:", weights

(b, w) = zip(biases, weights)
print b
print w

# epochs = 5;
# for j in xrange(epochs):
#     print(j)

# # 2-D array: 2 x 3
# two_dim_matrix_one = np.array([[1, 2, 3], [4, 5, 6]])
# # 2-D array: 3 x 2
# two_dim_matrix_two = np.array([[1, 2], [3, 4], [5, 6]])
#
# two_multi_res = np.dot(two_dim_matrix_one, two_dim_matrix_two)
# print('two_multi_res: %s' %(two_multi_res))

# 矩阵乘法
a = np.matrix([[1,2], [3,4]])
b = np.matrix([[5,6], [7,8]])
ret1 = np.dot(a, b)
print ret1
ret2 = a * b
print ret2

# hadamard乘积
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
ret3 = a * b
print ret3
