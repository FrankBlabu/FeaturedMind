#!/usr/bin/python3
#
# train_border_detection.py - Train net to recognize border structures
#                             in feature images
#
# Frank Blankenburg, Mar. 2017
#

import argparse
import pickle
import sys
import h5py

import tensorflow as tf
import numpy as np

from common import Vec2d
    
FLAGS = None

#--------------------------------------------------------------------------
# CLASS TrainingData
#
# Container keeping the training data sets
#
class TrainingData:
    
    def __init__ (self, file):
        self.sample_size = file.attrs['sample_size']

        self.data = file['data']
        self.labels = file['labels']
        
        self.offset = 0
        
            
    def size (self):
        return len (self.data)
    
    def reset (self):
        self.offset = 0
    
    def get_next_batch (self, size):
        data = []
        labels = []
        
        for i in range (self.offset, self.offset + size):
            data.append (self.data[i % len (self.data)])
            labels.append (self.labels[i % len (self.labels)])
            
        self.offset = (self.offset + size) % len (self.data)
        
        return (data, labels)
    
    def get_data (self):
        return (self.data, self.labels)
        
    

#--------------------------------------------------------------------------
# Train border detection with a multilayer convolutional network
#
def train (config, data):
    
    #
    # Generate variable with given shape and a truncated normal distribution
    #
    def create_weight_variable (shape):
        initial = tf.truncated_normal (shape, stddev=0.1)
        return tf.Variable (initial)
    
    #
    # Generate constant for bias variable with a given shape
    #
    def create_bias_variable (shape):
        initial = tf.constant (0.1, shape=shape)
        return tf.Variable (initial)
    
    #
    # Configure summarizing noded for a variable
    #  
    def add_variable_summary (var):
        with tf.name_scope ('summaries'):
            mean = tf.reduce_mean (var)
            tf.summary.scalar ('mean', mean)
            
            with tf.name_scope ('stddev'):
                stddev = tf.sqrt (tf.reduce_mean (tf.square (var - mean)))                
                tf.summary.scalar ('stddev', stddev)
                
            tf.summary.scalar ('max', tf.reduce_max (var))
            tf.summary.scalar ('min', tf.reduce_min (var))
            tf.summary.histogram ('histogram', var)

    #
    # Create the model
    #
    with tf.name_scope ('input'):
        x = tf.placeholder (tf.float32, [None, data.sample_size * data.sample_size])
        y_ = tf.placeholder (tf.float32, [None, 2])
      
    W_conv1 = create_weight_variable ([5, 5, 1, 32])
    b_conv1 = create_bias_variable ([32])
    
    with tf.name_scope ('image'):
        x_image = tf.reshape (x, [-1, data.sample_size, data.sample_size, 1])
        tf.summary.image ('input', x_image, 10)
    
    with tf.name_scope ('pooling'):
        h_conv1 = tf.nn.conv2d (x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_relu1 = tf.nn.relu (h_conv1 + b_conv1)
        h_pool1 = tf.nn.max_pool (h_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            
    with tf.name_scope ('convolution'):
        W_conv2 = create_weight_variable ([5, 5, 32, 64])
        b_conv2 = create_bias_variable ([64])
            
    with tf.name_scope ('convolution_pooling'):
        h_conv2 = tf.nn.conv2d (h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_relu2 = tf.nn.relu (h_conv2 + b_conv2)
        h_pool2 = tf.nn.max_pool (h_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            
    W_fc1 = create_weight_variable ([8 * 8 * 64, 1024])
    b_fc1 = create_bias_variable ([1024])
    
    h_pool2_flat = tf.reshape (h_pool2, [-1, 8 * 8 * 64])
    h_fc1 = tf.nn.relu (tf.matmul (h_pool2_flat, W_fc1) + b_fc1)
    
    with tf.name_scope ('dropout'):
        keep_prob = tf.placeholder (tf.float32)
        h_fc1_drop = tf.nn.dropout (h_fc1, keep_prob)
        
        tf.summary.scalar ('dropout_keep_probability', keep_prob)
    
    W_fc2 = create_weight_variable ([1024, 2])
    b_fc2 = create_bias_variable ([2])
    
    with tf.name_scope ('y_conv'):
        y_conv = tf.matmul (h_fc1_drop, W_fc2) + b_fc2
        
        add_variable_summary (y_conv)
    
    with tf.name_scope ('cross_entropy'):
        cross_entropy = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits (labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer (1e-4).minimize (cross_entropy)
        correct_prediction = tf.equal (tf.argmax (y_conv, 1), tf.argmax (y_, 1))
        accuracy = tf.reduce_mean (tf.cast (correct_prediction, tf.float32))
        
    tf.summary.scalar ('cross_entropy', cross_entropy)
    tf.summary.scalar ('accuracy', accuracy)
    
    session = tf.InteractiveSession ()
    
    merged_summary = tf.summary.merge_all ()
    
    if args.log != None:
        train_writer = tf.summary.FileWriter (args.log, session.graph)
     
    session.run (tf.global_variables_initializer ())
    
    for i in range (args.steps):
        batch = data.get_next_batch (args.batchsize)
        
        if i % 100 == 0:
            train_accuracy = accuracy.eval (feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

        if args.log != None:    
            summary, _ = session.run ([merged_summary, train_step], feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
            train_writer.add_summary (summary, i)
        else:
            session.run (train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
            
    #print("test accuracy %g" % accuracy.eval (feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    
    if args.log != None:
        train_writer.close ()
    
    #
    # Save model if configured
    #
    if args.output != None:
        saver = tf.train.Saver ()
        saver.save (session, args.output)


#--------------------------------------------------------------------------
# MAIN
#

#
# Parse command line arguments
#
parser = argparse.ArgumentParser ()

parser.add_argument ('file',              type=str,               help='Test dataset file name')
parser.add_argument ('-s', '--steps',     type=int, default=1000, help='Number of steps')
parser.add_argument ('-l', '--log',       type=str, default=None, help='Log file directory')
parser.add_argument ('-o', '--output',    type=str, default=None, help='Model output file name')
parser.add_argument ('-b', '--batchsize', type=int, default=50,   help='Number of samples per training batch')

args = parser.parse_args ()

assert args.steps >= 100

#
# Load sample data
#
file = h5py.File (args.file, 'r')
data = TrainingData (file)
    
print ("Training model...")
print ("  Number of samples: ", data.size ())
print ("  Sample size      : ", data.sample_size)

    
train (args, data)

file.close ()

