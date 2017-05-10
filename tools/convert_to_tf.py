#!/usr/bin/python3
#
# convert_to_tf.py - Export trained keras model into tensorflow format file
#
# Frank Blankenburg, May 2017
#

import argparse
import gc
import os
import tensorflow as tf

import common.losses as losses
import common.metrics as metrics

from keras.models import load_model
from keras import backend as K


#--------------------------------------------------------------------------
# MAIN
#

#
# Parse command line arguments
#
parser = argparse.ArgumentParser ()
parser.add_argument ('model',          type=str, help='Trained model data')
parser.add_argument ('-o', '--output', type=str, help='Output file')

args = parser.parse_args ()


#
# Load and construct model
#
model = load_model (args.model, custom_objects={'dice_coef': losses.dice_coef,
                                                'precision': metrics.precision, 
                                                'recall'   : metrics.recall,
                                                'f1_score' : metrics.f1_score})


saver = tf.train.Saver ()
saver.save (K.get_session (), os.path.abspath (args.output))

#
# Tensorflow termination bug workaround
#
gc.collect ()


