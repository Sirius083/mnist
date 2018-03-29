# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:10:14 2018

@author: Sirius
# one fully connected layer
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_ = tf.placeholder(tf.float32, [None, 10]) # correct answer
y = tf.matmul(x, W) + b
# version1: y = tf.nn.softmax(tf.matmul(x, W) + b)
# Then tf.reduce_sum adds the elements in the second dimension of y
# version1: cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_, logits = y)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    # 将 True False 映射成为0,1数字
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # res1 = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
    # res2 = sess.run(cross_entropy2, feed_dict={x: batch_xs, y_: batch_ys})
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    