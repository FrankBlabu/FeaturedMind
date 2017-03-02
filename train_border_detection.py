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
        self.training_data = []
        self.labels = []
        
        for sample in data['samples']:            
            
            assert len (sample['data']) == self.sample_size * self.sample_size
            
            self.training_data.append (sample['data'])
            self.labels.append ([0, 1] if sample['label'] else [1, 0])
            
    def size (self):
        return len (self.training_data)
    
    def get_batch (self, offset, size):
        return (self.training_data[offset:offset + size],
                self.labels[offset:offset + size])
        
    def get_data (self):
        return (self.training_data, self.labels)
        
    

#--------------------------------------------------------------------------
# Train border detection with simple model
#
def train_simple (data):

    # Create the model
    x = tf.placeholder (tf.float32, [None, data.sample_size * data.sample_size])
    W = tf.Variable (tf.zeros ([data.sample_size * data.sample_size, 2]))
    b = tf.Variable (tf.zeros ([2]))
    y = tf.matmul (x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder (tf.float32, [None, 2])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean (-tf.reduce_sum (y_ * tf.log (tf.nn.softmax (y)),
    #                                   reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits (labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer (0.5).minimize (cross_entropy)

    session = tf.InteractiveSession ()
    tf.global_variables_initializer ().run ()

    # Train
    count = 0
    while count < data.size ():
        batch_xs, batch_ys = data.get_batch (count, 100)
        session.run (train_step, feed_dict={x: batch_xs, y_: batch_ys})
        count += 100

    # Test trained model
    correct_prediction = tf.equal (tf.argmax (y, 1), tf.argmax (y_, 1))
    accuracy = tf.reduce_mean (tf.cast (correct_prediction, tf.float32))
    all_data = data.get_data ()
    print (session.run (accuracy, feed_dict={x: all_data[0], y_: all_data[1]}))




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

