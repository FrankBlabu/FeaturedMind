#!/usr/bin/python3
#
# border_detection.py - Detect features in images
#
# Frank Blankenburg, Mar. 2017
#

import argparse
import gc

import numpy as np
import common.metrics
import common.utils as utils

from keras.models import load_model
from skimage.color import gray2rgb

from test_image_generator import TestImage


#--------------------------------------------------------------------------
# MAIN
#

np.set_printoptions (threshold=np.nan)

#
# Parse command line arguments
#
parser = argparse.ArgumentParser ()

parser.add_argument ('model',          type=str,              help='Trained model data (meta data file)')
parser.add_argument ('-x', '--width',  type=int, default=320, help='Width of generated image in pixels')
parser.add_argument ('-y', '--height', type=int, default=240, help='Height of generated image in pixels')

args = parser.parse_args ()

assert args.width < 4096
assert args.height < 4096

#
# Load and construct border detection model
#
model = load_model (args.model, custom_objects={'precision'     : common.metrics.precision, 
                                                'recall'        : common.metrics.recall})

#
# Generate new image and predict feature pixel mask
#
args.sample_size = 16
test_image = TestImage (args)

image = test_image.image
mask = model.predict (np.reshape (image, (1, image.shape[0], image.shape[1], 1)))

mask[mask < 0.5] = 0

utils.show_image ([gray2rgb (image), 'Generated image'],
                  [mask[0,:,:,0],    'Result mask'])

#
# Tensorflow termination bug workaround
#
gc.collect ()
