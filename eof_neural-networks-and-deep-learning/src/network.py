#!/usr/bin/python
# -*- coding: UTF-8 -*-

import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # 随机化偏置值（符合正态分布）
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # 随机化权重（符合正态分布）
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    #
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = (np.dot(w, a) + b)
        return a

    # 梯度下降SGD
    # training_data：训练数据 (x, y)列表
    # epochs：代
    # mini_batch_size：小批量数据
    # eta：学习速率
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                # 应用一次增量梯度下降
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0} : {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    #
    def update_mini_batch(self, mini_batch, eta):
        # 梯度向量：nabla_b、nabla_w
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            self.weights = [w - (eta / len(mini_batch)) * nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (eta / len(mini_batch)) * nb
                           for b, nb in zip(self.biases, nabla_b)]

    # 反向传播
    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        activation = x
        print "backprop len(activation):", len(activation)
        activations = [x]
        zs = []
        # 前向传播
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # 反向传播
        # 输出误差：delta，"*"左右是np.array()，self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) 结果才是hadamard乘积
        print "backprop activations[-1]：", activations[-1]
        print "backprop sigmoid_prime(zs[-1])：", sigmoid_prime(zs[-1])
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        print "backprop delta：", delta
        #
        nabla_b[-1] = delta
        #
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    # 评估函数
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # 评估函数对activation的导数
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

# 逻辑函数sigmoid
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# sigmoid导数
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
