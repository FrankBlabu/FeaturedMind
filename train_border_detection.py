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
import PIL.Image
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
    
    def __init__ (self, data):
        self.sample_size = data['sample_size']
        self.training_data = np.zeros (shape=(len (data['samples']), self.sample_size * self.sample_size))
        self.labels = np.zeros (shape=(len (data['samples']), 2))
        self.offset = 0
        
        count = 0
        for sample in data['samples']:
            
            assert len (sample['data']) == self.sample_size * self.sample_size
            
            self.training_data[count] = sample['data']
            self.labels[count] = [0, 1] if sample['label'] else [1, 0]
            
            count += 1
            
    def size (self):
        return len (self.training_data)
    
    def get_next_batch (self, size):
        data = []
        labels = []
        
        for i in range (self.offset, self.offset + size):
            data.append (self.training_data[i % len (self.training_data)])
            labels.append (self.labels[i % len (self.labels)])
        
        self.offset = (self.offset + size) % len (self.training_data)
        
        return (data, labels)
    
    def get_data (self):
        return (self.training_data, self.labels)
        
    

#--------------------------------------------------------------------------
# Train border detection with simple linear model
#
def train_simple (data):

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

    session = tf.InteractiveSession ()
    tf.global_variables_initializer ().run ()

    # Train
    for step in range (20000):
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
def train_estimators ():

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
training_data = None

with open (args.file, 'rb') as file:
    training_data = TrainingData (pickle.load (file))
    
train_simple (training_data)
#train_estimator (training_data)

