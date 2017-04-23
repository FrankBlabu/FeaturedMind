#!/usr/bin/python3
#
# border_detection.py - Detect borders in image using a previously trained
#                       deep learning graph
#
# Frank Blankenburg, Mar. 2017
#

import argparse
import gc
import math
import time

import numpy as np
import common.log as log
import common.metrics as metrics
import common.utils as utils

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
parser.add_argument ('-s', '--sample-size', type=int, default=64,   help='Edge size of each sample')
parser.add_argument ('-l', '--log',         type=str,               help='Directory used for generating log data (HTML file and images)')
parser.add_argument ('-p', '--performance', type=int,               help='Do performance measurement runs')

args = parser.parse_args ()

assert args.width < 4096
assert args.height < 4096

#
# Load and construct model
#
model = load_model (args.model, custom_objects={'precision': metrics.precision, 
                                                'recall': metrics.recall})

#
# Create test image and setup input tensors
#
source = TestImage (args)

image = source.image
mask, _ = source.get_feature_mask ()

x_steps = int (math.floor (args.width / args.sample_size))
y_steps = int (math.floor (args.height / args.sample_size))

x_predict = np.zeros ((x_steps * y_steps, args.sample_size, args.sample_size, 1))
y_predict = np.zeros ((x_steps * y_steps, args.sample_size, args.sample_size, 1))
    
count = 0

for y in range (y_steps):
    for x in range (x_steps):
        
        rect = Rect2d (Point2d (x * args.sample_size, y * args.sample_size), Size2d (args.sample_size, args.sample_size))  
        
        image_sample = utils.cutout (image, rect)
        mask_sample = utils.cutout (mask, rect)        
        
        x_predict[count] = utils.mean_center (image_sample)
        y_predict[count] = mask_sample
                
        count += 1
       
#
# Run border detection network
#
score = model.evaluate (x_predict, y_predict, verbose=0)

print ('Test loss:', score[0])
print ('Test accuracy:', score[1])

if args.performance:
    start_time = time.process_time ()
    
    for _ in range (args.performance):
        model.predict (x_predict)

    elapsed_time = (time.process_time () - start_time) / (10 * args.performance)

    print ('Single run duration: {0:.2f} s'.format (elapsed_time))


result_image = np.zeros ((y_steps * args.sample_size, x_steps * args.sample_size, 1))
result = model.predict (x_predict)

if args.log:
    with log.HTMLLogger (args.log, 'Prediction result') as log:
        rows = []
        
        rows.append (['Image', 'Mask', 'Predicted'])
        
        for i in range (y_steps * x_steps):
            
            rows.append ([utils.to_rgb (x_predict[i]), 
                          utils.to_rgb (y_predict[i]),
                          utils.to_rgb (result[i])])
            
        log.add_caption ('Sample results')
        log.add_table (rows, has_header=True)

count = 0

for y in range (y_steps):
    for x in range (x_steps):

        xs = x * args.sample_size
        ys = y * args.sample_size
        
        result_image[ys:ys + args.sample_size, xs:xs + args.sample_size] = result[count]

        if count == 0:
            print (result[count].shape)
            print (result[count])

        count += 1

result_image = utils.mean_uncenter (result_image)
result_image = result_image.reshape ((result_image.shape[0], result_image.shape[1]))

utils.show_image ([utils.to_rgb (image), 'Image'], 
                  [utils.to_rgb (result_image), 'Border prediction'])



#
# Tensorflow termination bug workaround
#
gc.collect ()

