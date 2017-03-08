#!/usr/bin/python3
#
# border_detection.py - Detect borders in image using a previously trained
#                       deep learning graph
#
# Frank Blankenburg, Mar. 2017
#

import argparse
import h5py
import math

import tensorflow as tf
import numpy as np

from test_image_generator import TestImage

#--------------------------------------------------------------------------
# MAIN
#

#
# Parse command line arguments
#
parser = argparse.ArgumentParser ()

parser.add_argument ('model',               type=str,              help='Trained model data (meta data file)')
parser.add_argument ('-x', '--width',       type=int, default=640, help='Width of generated image in pixels')
parser.add_argument ('-y', '--height',      type=int, default=480, help='Height of generated image in pixels')
parser.add_argument ('-s', '--sample-size', type=int, default=32,  help='Edge size of each sample')

args = parser.parse_args ()

assert args.width < 4096
assert args.height < 4096

#
# Load and construct model
#
session = tf.Session ()
loader = tf.train.import_meta_graph (args.model)
loader.restore (session, tf.train.latest_checkpoint ('./'))

#
# Create test image and setup input tensors
#
test_image = TestImage (args.width, args.height)

samples_x = int (math.floor (args.width / args.sample_size))
samples_y = int (math.floor (args.height / args.sample_size))
sample_size = args.sample_size

x = np.zeros ((samples_x * samples_y, sample_size * sample_size))
y = np.zeros ((samples_x * samples_y, 2))

count = 0
for ys in range (0, samples_y):
    for xs in range (0, samples_x):
        sample, flag = test_image.get_sample (xs * sample_size, ys * sample_size, sample_size)

        x[count] = sample
        y[count] = [0 if flag else 1, 1 if flag else 0]
        
        count += 1
        
#
# Run border detection network
#
y_conv_node = tf.get_default_graph ().get_tensor_by_name ('y_conv:0')
accuracy_node = tf.get_default_graph ().get_tensor_by_name ('cross_entropy/accuracy:0')

y_conv, accuracy = session.run ([y_conv_node, accuracy_node], feed_dict={'input/x:0': x, 'input/y_:0': y, 'dropout/keep_prob:0': 1.0})

print ('Accuracy: ', accuracy)



