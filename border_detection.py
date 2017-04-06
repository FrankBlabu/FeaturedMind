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

import numpy as np
import common.metrics

from keras.models import load_model
from keras import backend as K

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
model = load_model (args.model, custom_objects={'precision': common.metrics.precision, 'recall': common.metrics.recall})


#
# Create test image and setup input tensors
#
test_image = TestImage (args)

x = test_image.samples.reshape ((-1, test_image.samples.shape[2]))
y = np.clip (test_image.labels.reshape ((-1)), 0, 1)

if K.image_data_format () == 'channels_first':
    x = x.reshape (x.shape[0], 1, args.sample_size, args.sample_size)
else:
    x = x.reshape (x.shape[0], args.sample_size, args.sample_size, 1)
    
x = x.astype ('float32')
y = keras.utils.to_categorical (y, 2)
       
#
# Run border detection network
#
score = model.evaluate (x, y, verbose=0)

result = model.predict (x)
result = np.argmax (result, axis=1).reshape ((test_image.samples.shape[0], test_image.samples.shape[1]))

print ('Test loss:', score[0])
print ('Test accuracy:', score[1])

if args.performance > 0:
    start_time = time.process_time ()
    
    for _ in range (args.performance):
        score = model.evaluate (x, y, verbose=0)
    
    elapsed_time = time.process_time () - start_time
    
    print ('Duration ({0} runs): {1:.2f} s'.format (args.performance, elapsed_time))
    
image = test_image.to_rgb ()
overlay = test_image.create_result_overlay (result)
image.paste (overlay, (0, 0), overlay)
image.show ()

#
# Tensorflow termination bug workaround
#
gc.collect ()

