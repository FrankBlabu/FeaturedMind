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
import common.metrics as metrics
import common.utils as utils

import skimage.color

from common.geometry import Point2d, Size2d, Rect2d
from keras.models import load_model
from test_image_generator import TestImage


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
parser.add_argument ('-p', '--performance', type=int, default=0,    help='Number of runs for performance measurement')       

args = parser.parse_args ()

assert args.width < 4096
assert args.height < 4096

#
# Load and construct model
#
model = load_model (args.model, custom_objects={'precision': metrics.precision, 'recall': metrics.recall})

#
# Create test image and setup input tensors
#
source = TestImage (args)

x_steps = int (math.floor (args.width / args.sample_size))
y_steps = int (math.floor (args.height / args.sample_size))

x_predict = np.zeros ((x_steps * y_steps, args.sample_size, args.sample_size, 1))
y_predict = np.zeros ((x_steps * y_steps,))
    
count = 0

for y in range (y_steps):
    for x in range (x_steps):
        
        rect = Rect2d (Point2d (x * args.sample_size, y * args.sample_size), Size2d (args.sample_size, args.sample_size))            
        sample, label = source.get_sample (rect)
        
        x_predict[count] = utils.image_to_tf (sample)
        y_predict[count] = label
        count += 1

y_predict[y_predict > 0] = 1
y_predict = keras.utils.to_categorical (y_predict, 2)
       
#
# Run border detection network
#
score = model.evaluate (x_predict, y_predict, verbose=0)

result = model.predict (x_predict)
result = np.argmax (result, axis=1).reshape ((y_steps, x_steps))

print ('Test loss:', score[0])
print ('Test accuracy:', score[1])

if args.performance > 0:
    start_time = time.process_time ()
    
    for _ in range (args.performance):
        score = model.evaluate (x, y, verbose=0)
    
    elapsed_time = time.process_time () - start_time
    
    print ('Duration ({0} runs): {1:.2f} s'.format (args.performance, elapsed_time))

image = skimage.color.gray2rgb (source.image)
overlay = source.create_result_overlay (result)

image = utils.add_overlay_to_image (image, overlay)
utils.show_image ((image, 'Result'))

#
# Tensorflow termination bug workaround
#
gc.collect ()

