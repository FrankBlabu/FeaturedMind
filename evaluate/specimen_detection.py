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
import generator.background as background
import skimage.color
import skimage.filters
import tensorflow as tf

import generator.background
import generator.fixture
import generator.generator
import generator.sheetmetal


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

background.BackgroundGenerator.add_to_args_definition (parser)

args = parser.parse_args ()

assert args.width < 4096
assert args.height < 4096

#
# Load and construct model
#
model = tf.keras.models.load_model (args.model, custom_objects={
    'dice_coef': losses.dice_coef,
    'precision': metrics.precision,
    'recall'   : metrics.recall,
    'f1_score' : metrics.f1_score})

#
# Create test specimen and setup input tensors
#
generators = [ generator.background.BackgroundGenerator.create (args),
               generator.sheetmetal.SheetMetalGenerator (args.width, args.height),
               generator.fixture.FixtureGenerator (args.width, args.height) ]

data = generator.generator.batch_generator (generator.generator.StackedGenerator (args.width, args.height, 3, generators), 10)

images, masks = next (data)

#
# Run border detection network
#
scores = model.evaluate (images, masks, verbose=args.verbose)

for score in zip (scores, model.metrics_names):
    print ('{0}: {1:.02f}'.format (score[1], score[0]))

if args.performance:
    start_time = time.process_time ()

    images, masks = next (data.generate)
    model.predict (images)

    elapsed_time = (time.process_time () - start_time) / (10 * args.performance)

    print ('Single run duration: {0:.4f} s'.format (elapsed_time))

start_time = time.process_time ()

result = model.predict (images)[0]

print ('Duration: {0:.4f}s'.format ((time.process_time () - start_time) / 10))

#result[result < 0.5] = 0.0

mask = np.zeros (result.shape[0:2])

mask[result[:,:,0] > 0.5] = 255
#mask[result[:,:,1] > 0.5] = 1.0

#edges = result.reshape ((result.shape[0], result.shape[1]))
#edges = skimage.filters.sobel (edges)

utils.show_image ([images[0],                     'Generated image'],
                  [skimage.color.gray2rgb (mask), 'Predicted specimen features'])



#
# Tensorflow termination bug workaround
#
gc.collect ()
