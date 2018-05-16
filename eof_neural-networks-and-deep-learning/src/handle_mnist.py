#!/usr/bin/python
# -*- coding: UTF-8 -*-

import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print "training_data:", len(training_data)
print "validation_data:", len(validation_data)
print "test_data:", len(test_data)

import network
# 输入层：784个神经元
# 隐藏层：30个神经元
# 输出层：10个神经元
net = network.Network([784, 30, 10])

# epochs: 30
net.SGD(training_data, 30, 10, 3.0, test_data = test_data)