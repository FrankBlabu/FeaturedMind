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
import time

import tensorflow as tf
import numpy as np

from training.training_data import TrainingData

from test_image_generator import TestImage
from display_sampled_image import create_result_image

#--------------------------------------------------------------------------
# MAIN
#

#
# Parse command line arguments
#
parser = argparse.ArgumentParser ()

parser.add_argument ('model',               type=str,                           help='Trained model data (meta data file)')
parser.add_argument ('-x', '--width',       type=int,            default=640,   help='Width of generated image in pixels')
parser.add_argument ('-y', '--height',      type=int,            default=480,   help='Height of generated image in pixels')
parser.add_argument ('-s', '--sample-size', type=int,            default=32,    help='Edge size of each sample')
parser.add_argument ('-r', '--runs',        type=int,            default=1,     help='Number of runs')
parser.add_argument ('-i', '--show-image',  action='store_true', default=False, help='Show results as image')

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
y = np.zeros ((samples_x * samples_y))

count = 0
for ys in range (0, samples_y):
    for xs in range (0, samples_x):
        sample, label = test_image.get_sample (xs * sample_size, ys * sample_size, sample_size)

        x[count] = sample
        y[count] = label
        
        count += 1
        
#
# Run border detection network
#
result_node = tf.get_default_graph ().get_tensor_by_name ('result:0')
accuracy_node = tf.get_default_graph ().get_tensor_by_name ('training/accuracy:0')

if args.show_image:
    if args.runs > 1:
        print ("Warning: '-r' ignored when displaying the result as an image")

    result, accuracy = session.run ([result_node, accuracy_node], feed_dict={'input/x:0': x, 'input/y_:0': y, 'dropout/keep_prob:0': 1.0})
    print ('Accuracy: ', accuracy)
   
    image = create_result_image (test_image, args.sample_size, result.reshape ((samples_y, samples_x)))
    image.show ()

else:
    start_time = time.process_time ()
    
    for _ in range (args.runs):
        result, accuracy = session.run ([result_node, accuracy_node], feed_dict={'input/x:0': x, 'input/y_:0': y, 'dropout/keep_prob:0': 1.0})
        print ('Accuracy: ', accuracy)
    
    elapsed_time = time.process_time () - start_time
    
    print ('Duration ({0} runs): {1:.2f} s'.format (args.runs, elapsed_time))

