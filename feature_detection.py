#!/usr/bin/python3
#
# border_detection.py - Detect features in images
#
# Frank Blankenburg, Mar. 2017
#

import argparse
import gc
import keras
import math
import time

import numpy as np
import tensorflow as tf
import common.metrics

from keras.models import load_model
from keras import backend as K

from common.geometry import Point2d, Size2d, Rect2d
from test_image_generator import TestImage
from display_sampled_image import create_result_image

#--------------------------------------------------------------------------
# MAIN
#

np.set_printoptions (threshold=np.nan)

#
# Parse command line arguments
#
parser = argparse.ArgumentParser ()

parser.add_argument ('model',               type=str,               help='Trained model data (meta data file)')
parser.add_argument ('-x', '--width',       type=int, default=1024, help='Width of generated image in pixels')
parser.add_argument ('-y', '--height',      type=int, default=768,  help='Height of generated image in pixels')
parser.add_argument ('-s', '--sample-size', type=int, default=16,   help='Edge size of each sample')

args = parser.parse_args ()

assert args.width < 4096
assert args.height < 4096

#
# Load and construct border detection model
#
model = load_model (args.model, custom_objects={'precision': common.metrics.precision, 'recall': common.metrics.recall})


#
# STEP 1: Border detection
#
# Create test image and setup input tensors
#
print ("Step 1")

test_image = TestImage (args)

samples_x = int (math.floor (args.width / args.sample_size))
samples_y = int (math.floor (args.height / args.sample_size))
sample_size = Size2d (args.sample_size, args.sample_size)

x = np.zeros ((samples_x * samples_y, args.sample_size * args.sample_size))
y = np.zeros (samples_x * samples_y)

count = 0
for ys in range (0, samples_y):
    for xs in range (0, samples_x):
        sample, label = test_image.get_sample (Rect2d (Point2d (xs * args.sample_size, ys * args.sample_size), sample_size))

        x[count] = sample
        y[count] = 1 if label > 0 else 0
        
        count += 1
  
if K.image_data_format () == 'channels_first':
    x = x.reshape (x.shape[0], 1, args.sample_size, args.sample_size)
else:
    x = x.reshape (x.shape[0], args.sample_size, args.sample_size, 1)
    
x = x.astype ('float32')
y = keras.utils.to_categorical (y, 2)

result = model.predict (x)
result = np.argmax (result, axis=1).reshape ((samples_y, samples_x))


#
# STEP 2: Use k-means algorithm to find clusters
#
# Collect coordinates of the segments containing a border
#

print ("Step 2")

indices = []
values = []

for ys in range (0, samples_y):
    for xs in range (0, samples_x):
        if result[ys][xs] > 0:
            indices.append ((xs, ys))
            values.append ((xs, ys))

tiles = np.array (values, dtype=np.float)

N = len (tiles)
K = 8
MAX_ITERS = 1000

points = tf.placeholder (tf.float32, [None, 2])
cluster_assignments = tf.Variable (tf.zeros ((N), dtype=tf.int64))

# Silly initialization:  Use the first K points as the starting
# centroids.  In the real world, do this better.
centroids = tf.Variable (tf.slice (points, [0,0], [K,2]))

# Replicate to N copies of each centroid and K copies of each
# point, then subtract and compute the sum of squared distances.
rep_centroids = tf.reshape (tf.tile (centroids, [N, 1]), [N, K, 2])
rep_points = tf.reshape (tf.tile (points, [1, K]), [N, K, 2])
sum_squares = tf.reduce_sum (tf.square (rep_points - rep_centroids), reduction_indices=2)

# Use argmin to select the lowest-distance point
best_centroids = tf.argmin (sum_squares, 1)
did_assignments_change = tf.reduce_any (tf.not_equal (best_centroids, cluster_assignments))

def bucket_mean (data, bucket_ids, num_buckets):
    total = tf.unsorted_segment_sum (data, bucket_ids, num_buckets)
    count = tf.unsorted_segment_sum (tf.ones_like(data), bucket_ids, num_buckets)
    return total / count

means = bucket_mean (points, best_centroids, K)

# Do not write to the assigned clusters variable until after
# computing whether the assignments have changed - hence with_dependencies
with tf.control_dependencies ([did_assignments_change]):
    do_updates = tf.group (centroids.assign (means), cluster_assignments.assign (best_centroids))

sess = tf.Session ()
sess.run (tf.global_variables_initializer (), feed_dict={points: tiles})

changed = True
iters = 0

while changed and iters < MAX_ITERS:
    changed, _ = sess.run ([did_assignments_change, do_updates], feed_dict={points: tiles})
    iters += 1

centroids, assignments = sess.run ([centroids, cluster_assignments],feed_dict={points: tiles})

assert len (indices) ==  assignments.shape[0]

for index, assignment in zip (indices, assignments):
    result[index[1]][index[0]] = assignment + 1

image = create_result_image (test_image, sample_size, result)
image.show ()


print ("Step 2 done")

#
# Tensorflow termination bug workaround
#
gc.collect ()

