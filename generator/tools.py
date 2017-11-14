#!/usr/bin/python3
#
# tools.py - Generator tools
#
# Frank Blankenburg, Nov. 2017
#

import argparse
import random
import math
import numpy as np

import common.utils as utils

import skimage.filters
import skimage.util


#----------------------------------------------------------------------------------------------------------------------
# Create a metal texture
#
# See http://www.jhlabs.com/ip/brushed_metal.html for the algorithm
#
def create_metal_texture (width, height, color, shine):

    image = np.zeros ((height * 2, width * 2, 1), dtype=np.float32)
    image[:,:] = color

    image = skimage.util.random_noise (image, mode='gaussian', mean=0, var=0.005)
    image = skimage.filters.gaussian (image, sigma=[0, 12, 0], mode='nearest')

    offset = np.random.uniform ()

    for y in range (image.shape[0]):
        for x in range (image.shape[1]):
            factor = shine * math.sin (x * math.pi / width - math.pi * offset / 2)
            image[y,x] += factor

    image = skimage.transform.rotate (image, np.random.uniform (0, 90.0), resize=False)
    image = image[height - int (height / 2):height + int (height / 2),width - int (width / 2):width + int (width / 2),:]

    return np.dstack ((image, image, image))


#--------------------------------------------------------------------------
# MAIN
#
if __name__ == '__main__':

    random.seed ()

    #
    # Parse command line arguments
    #
    parser = argparse.ArgumentParser ()

    parser.add_argument ('-x', '--width',   type=int,   default=640,  help='Width of the generated images')
    parser.add_argument ('-y', '--height',  type=int,   default=480,  help='Height of the generated images')
    parser.add_argument ('-c', '--color',   type=float, default=0.3,  help='Color seed')
    parser.add_argument ('-s', '--shine',   type=float, default=0.1,  help='Shine seed')

    args = parser.parse_args ()

    image = create_metal_texture (width=args.width, height=args.height, color=args.color, shine=args.shine)
    utils.show_image ([image, 'Metal texture'])
