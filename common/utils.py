#!/usr/bin/python3
#
# utils.py - General utilities
#
# Frank Blankenburg, Apr. 2017
#

import numpy as np


#----------------------------------------------------------------------------
# Convert pillow image into TensorFlow compatible numpy array
#
# @param image Image (PIL) to convert
#
def image_to_tf (image):
    return np.asarray ([float (d) / 255 for d in image.getdata ()], dtype=np.float32).reshape ((image.height, image.width, 1))

