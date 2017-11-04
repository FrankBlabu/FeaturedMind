#!/usr/bin/python3
#
# model_info.py - Print information about a given model
#
# Frank Blankenburg, Apr. 2017
#

import argparse
import gc

import common.losses as losses
import common.metrics as metrics

from tf.keras.models import load_model



#--------------------------------------------------------------------------
# MAIN
#

#
# Parse command line arguments
#
parser = argparse.ArgumentParser ()
parser.add_argument ('model', type=str,  help='Trained model data')

args = parser.parse_args ()


#
# Load and construct model
#
model = load_model (args.model, custom_objects={'dice_coef': losses.dice_coef,
                                                'precision': metrics.precision,
                                                'recall'   : metrics.recall,
                                                'f1_score' : metrics.f1_score})

#
# Summarize information
#
model.summary ()

#
# Tensorflow termination bug workaround
#
gc.collect ()
