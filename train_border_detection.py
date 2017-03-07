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
        return len (self.data['data'])
    
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
# Train border detection with simple linear model
#
def train_softmax_regression (data):

    data_size = data.sample_size * data.sample_size
    all_data = data.get_data ()

    print ("Training model...")
    print ("  Number of samples: ", len (all_data[0]))

    # Create the model
    x = tf.placeholder (tf.float32, [None, data_size])
    W = tf.Variable (tf.zeros ([data_size, 2]))
    b = tf.Variable (tf.zeros ([2]))
    y = tf.matmul (x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder (tf.float32, [None, 2])

    cross_entropy = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits (labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer (0.5).minimize (cross_entropy)

    session = tf.Session ()
    tf.global_variables_initializer ().run ()

    # Train
    for step in range (2000):
        if step > 0 and step % 100 == 0:
            print ("  Step", step)
            
        batch_xs, batch_ys = data.get_next_batch (100)
        session.run (train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal (tf.argmax (y, 1), tf.argmax (y_, 1))
    accuracy = tf.reduce_mean (tf.cast (correct_prediction, tf.float32))
        
    print ("  Accuracy:", session.run (accuracy, feed_dict={x: all_data[0], y_: all_data[1]}))

#--------------------------------------------------------------------------
# Train border detection with higher level TensorFlow estimator objects
#
def train_estimators (data):

    features = [tf.contrib.layers.real_valued_column ("x", dimension=1)]
    
    # An estimator is the front end to invoke training (fitting) and evaluation
    # (inference). There are many predefined types like linear regression,
    # logistic regression, linear classification, logistic classification, and
    # many neural network classifiers and regressors. The following code
    # provides an estimator that does linear regression.
    estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
    
    # TensorFlow provides many helper methods to read and set up data sets.
    # Here we use `numpy_input_fn`. We have to tell the function how many batches
    # of data (num_epochs) we want and how big each batch should be.
    x = np.array([1., 2., 3., 4.])
    y = np.array([0., -1., -2., -3.])
    input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4,
                                                  num_epochs=1000)
    
    # We can invoke 1000 training steps by invoking the `fit` method and passing the
    # training data set.
    estimator.fit(input_fn=input_fn, steps=1000)
    
    # Here we evaluate how well our model did. In a real example, we would want
    # to use a separate validation and testing data set to avoid overfitting.
    estimator.evaluate(input_fn=input_fn)


#--------------------------------------------------------------------------
# Train border detection with a multilayer convolutional network
#
def train_multilayer_convolution (data):
    
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
      
    data_size = data.sample_size * data.sample_size
    all_data = data.get_data ()

    print ("Training model...")
    print ("  Number of samples: ", len (all_data[0]))

    # Create the model
    with tf.name_scope ('input'):
        x = tf.placeholder (tf.float32, [None, data_size])
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
    train_writer = tf.summary.FileWriter ('log', session.graph) 
    session.run (tf.global_variables_initializer ())
    
    for i in range (300):
        batch = data.get_next_batch (50)
        
        if i % 100 == 0:
            train_accuracy = accuracy.eval (feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
    
        summary, _ = session.run ([merged_summary, train_step], feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
        train_writer.add_summary (summary, i)
        
    
    #print("test accuracy %g" % accuracy.eval (feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    
    train_writer.close ()


#--------------------------------------------------------------------------
# MAIN
#

#
# Parse command line arguments
#
parser = argparse.ArgumentParser ()

parser.add_argument ('file', type=str, help='Output file name')

args = parser.parse_args ()

#
# Load sample data
#
file = h5py.File (args.file, 'r')
data = TrainingData (file)
    
#train_softmax_regression (data)
#train_estimator (data)
train_multilayer_convolution (data)

file.close ()

