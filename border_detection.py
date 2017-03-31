#!/usr/bin/python3
#
# border_detection.py - Detect borders in image using a previously trained
#                       deep learning graph
#
# Frank Blankenburg, Mar. 2017
#

import argparse
import gc
import keras
import math
import time

import numpy as np
import common.metrics

from keras.models import load_model
from keras import backend as K

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

parser.add_argument ('model',               type=str,              help='Trained model data (meta data file)')
parser.add_argument ('-x', '--width',       type=int, default=640, help='Width of generated image in pixels')
parser.add_argument ('-y', '--height',      type=int, default=480, help='Height of generated image in pixels')
parser.add_argument ('-s', '--sample-size', type=int, default=32,  help='Edge size of each sample')
parser.add_argument ('-p', '--performance', type=int, default=0,   help='Number of runs for performance measurement')       

args = parser.parse_args ()

assert args.width < 4096
assert args.height < 4096

#
# Load and construct model
#
model = load_model (args.model, custom_objects={'precision': common.metrics.precision, 'recall': common.metrics.recall})


#
# Create test image and setup input tensors
#
test_image = TestImage (args)

samples_x = int (math.floor (args.width / args.sample_size))
samples_y = int (math.floor (args.height / args.sample_size))
sample_size = args.sample_size

x = np.zeros ((samples_x * samples_y, sample_size * sample_size))
y = np.zeros ((samples_x * samples_y))

count = 0
for ys in range (0, samples_y):
    for xs in range (0, samples_x):
        sample, label, cluster = test_image.get_sample (xs * sample_size, ys * sample_size, sample_size)

        x[count] = sample
        y[count] = label.value
        
        count += 1
  
if K.image_data_format () == 'channels_first':
    x = x.reshape (x.shape[0], 1, args.sample_size, args.sample_size)
else:
    x = x.reshape (x.shape[0], args.sample_size, args.sample_size, 1)
    
x = x.astype ('float32')
y = keras.utils.to_categorical (y, len (TestImage.Direction))

        
#
# Run border detection network
#
score = model.evaluate (x, y, verbose=0)

result = model.predict (x)
result = np.argmax (result, axis=1).reshape ((samples_y, samples_x))

print ('Test loss:', score[0])
print ('Test accuracy:', score[1])

if args.performance > 0:
    start_time = time.process_time ()
    
    for _ in range (args.performance):
        score = model.evaluate (x, y, verbose=0)
    
    elapsed_time = time.process_time () - start_time
    
    print ('Duration ({0} runs): {1:.2f} s'.format (args.performance, elapsed_time))
    
image = create_result_image (test_image, args.sample_size, result)
image.show ()

#
# Tensorflow termination bug workaround
#
gc.collect ()

