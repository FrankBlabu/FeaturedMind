#!/usr/bin/python3
#
# specimen_detection.py - Detect specimen using a previously trained model
#
# Frank Blankenburg, Mar. 2017
#

import argparse
import gc
import time

import numpy as np
import common.losses as losses
import common.metrics as metrics
import common.utils as utils

from keras.models import load_model
from generator.sheetmetal import SheetMetalGenerator



#--------------------------------------------------------------------------
# MAIN
#

np.set_printoptions (threshold=np.nan)

#
# Parse command line arguments
#
parser = argparse.ArgumentParser ()

parser.add_argument ('model',               type=str,                           help='Trained model data (meta data file)')
parser.add_argument ('-x', '--width',       type=int,            default=640,   help='Width of generated image in pixels')
parser.add_argument ('-y', '--height',      type=int,            default=480,   help='Height of generated image in pixels')
parser.add_argument ('-l', '--log',         type=str,                           help='Directory used for generating log data (HTML file and images)')
parser.add_argument ('-p', '--performance', type=int,                           help='Do performance measurement runs')
parser.add_argument ('-v', '--verbose',     action='store_true', default=False, help='Verbose output')

args = parser.parse_args ()

assert args.width < 4096
assert args.height < 4096

#
# Load and construct model
#
model = load_model (args.model, custom_objects={'dice_coef': losses.dice_coef,
                                                'precision': metrics.precision, 
                                                'recall'   : metrics.recall,
                                                'f1_score' : metrics.f1_score})

#
# Create test specimen and setup input tensors
#
sheet = SheetMetalGenerator (args.width, args.height)

image = utils.mean_center (sheet.image)
image = image.reshape (1, args.height, args.width, 1)

mask = sheet.mask
mask = mask.reshape (1, args.height, args.width, 1)
        
#
# Run border detection network
#
scores = model.evaluate (image, mask, verbose=args.verbose)

for score in zip (scores, model.metrics_names):
    print ('{0}: {1}'.format (score[1], score[0]))

if args.performance:
    start_time = time.process_time ()
    
    for _ in range (args.performance):
        model.predict (image)

    elapsed_time = (time.process_time () - start_time) / (10 * args.performance)

    print ('Single run duration: {0:.2f} s'.format (elapsed_time))


result = np.zeros ((1, args.height, args.width, 1))
result = model.predict (image)

utils.show_image ([utils.to_rgb (image[0]),  'Generated image'], 
                  [utils.to_rgb (result[0]), 'Predicted specimen borders'])



#
# Tensorflow termination bug workaround
#
gc.collect ()

