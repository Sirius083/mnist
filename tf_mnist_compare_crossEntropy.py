# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 20:13:27 2018

@author: Sirius

比较传统和tf.nn.softmax_cross_entropy_with_logits_v2的计算cross entropy的方法
"""

#--------------------------------------------------------------------------------------
# 计算一个cross_entropy 的计算公式和 tf.nn.softmax_cross_entropy_with_logits_v2是否一样
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
batch_size = 1000
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train = mnist.train.next_batch(batch_size)
train_data = train[0]
train_label = train[1]

W = tf.Variable(tf.truncated_normal([784,10], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[10]))

# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10]) # correct answer
# version1
y1 = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy1 = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y1), reduction_indices=[1]))

y2 = tf.matmul(x, W) + b
cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_ , logits = y2)


with tf.Session() as sess:
    tf.global_variables_initializer().run() # 初始化权重矩阵
    w_1 = sess.run(W)
    b_1 = sess.run(b)
    ce1 = sess.run(cross_entropy1, feed_dict={x: train_data, y_: train_label})
    ce2 = sess.run(cross_entropy2, feed_dict={x: train_data, y_: train_label})
    # print('cross_entropy1',sess.run(cross_entropy1, feed_dict={x: train_data, y_: train_label}))
    # print('cross_entropy2',sess.run(cross_entropy2, feed_dict={x: train_data, y_: train_label}))

print(ce1)
print(np.mean(ce2))  