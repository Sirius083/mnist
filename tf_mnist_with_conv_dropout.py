# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 20:43:47 2018

@author: Sirius

代码原网页
https://www.tensorflow.org/versions/r1.1/get_started/mnist/pros

解释 tf.nn.conv2d 比较清楚的博客
https://blog.csdn.net/u012609509/article/details/71215859
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10]) # correct answer
x_image = tf.reshape(x,[-1,28,28,1]) # 2: height, 3:width, 1:number of color channel

# 初始化权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

# 初始化卷积层和池化层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1],padding = 'SAME')



# first convolution layer
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolution layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer: 用到了1024个神经元
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout: turn dropout on during training, turn if off during testing
# tf.nn.dropout automatically handels scaling
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# train and evaluate the model
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_, logits = y_conv)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch = mnist.train.next_batch(100)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict = {x:batch[0], y_:batch[1], keep_prob:1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict = {x:batch[0], y_:batch[1], keep_prob:0.5})
    print("test accuracy %g" % accuracy.eval(feed_dict = {x:mnist.test.images, y_:mnist.test.labels, keep_prob:0.5}))

'''
# 4d by 2d, numpy arrray
# tf.nn.conv2d 的作用机制和 np.dot对nd.array的作用机制相同
import numpy as np
a = np.ones([2,3,4,5])
b = np.ones([5,6])
c = np.dot(a,b)
c.shape # [2,3,4,6]

from random import seed
seed(1)
a = np.random.rand(1,2,3,4)
b = np.random.rand(4,5)
c = np.dot(a,b)
print('a',a)
print('b',b)
print('c',c)
# 检查结果
print(np.dot(a[0,0,::],b))
print(c[0,0,::]) # 相等
print('---------------------')
print(np.dot(a[0,1,::],b))
print(c[0,1,::])
'''