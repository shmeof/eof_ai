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

epochs = 5;
for j in xrange(epochs):
    print(j)