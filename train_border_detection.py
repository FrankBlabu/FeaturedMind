#!/usr/bin/python3
#
# train_border_detection.py - Train net to recognize border structures
#                             in feature images
#
# Frank Blankenburg, Mar. 2017
#

import argparse
import pickle
import PIL.Image

from common import Vec2d
    

#--------------------------------------------------------------------------
# MAIN
#

#
# Parse command line arguments
#
parser = argparse.ArgumentParser ()

parser.add_argument ('file', type=str, help='Output file name')

args = parser.parse_args ()

#
# Load sample data
#
training_data = None

with open (args.file, 'rb') as file:
    training_data = pickle.load (file)

