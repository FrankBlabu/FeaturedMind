#!/usr/bin/python3
#
# background.py - Background generators
#
# Frank Blankenburg, May 2017
#

import argparse
import random
import math
import numpy as np
import common.utils as utils 

import skimage.filters
import skimage.util

from common.geometry import Point2d, Size2d, Rect2d


#--------------------------------------------------------------------------
# Generate pattern based background
#
# This class will generate an image background consisting of random, blurred,
# noisy and variated rectangles
#
def generate_noisy_rects (size): 

    shape = (int (size.height), int (size.width))

    image = np.zeros (shape, dtype=np.float32)
    image = skimage.util.random_noise (image, mode='gaussian', seed=None, clip=True, mean=0.2, var=0.0001)
    
    number_of_shapes = random.randint (30, 80)
    for _ in range (number_of_shapes):

        rect_image = np.zeros (shape, dtype=np.float32)
        
        rect = Rect2d (Point2d (2 * size.width / 10, 4.5 * size.height / 10),
                       Point2d (8 * size.width / 10, 5.5 * size.height / 10))

        rect = rect.move_to (Point2d (random.randint (int (-1 * size.width / 10), int (9 * size.width / 10)),
                                      random.randint (int (-1 * size.height / 10), int (9 * size.height / 10))))

        color = random.uniform (0.02, 0.1)
        rect.draw (rect_image, color, fill=True)
        
        if random.randint (0, 3) == 0:
            rect_image = skimage.util.random_noise (rect_image, mode='gaussian', seed=None, clip=True, mean=color, var=0.005)

        rect_image = utils.transform (rect_image, size,
                                      scale=(random.uniform (0.3, 1.4), random.uniform (0.8, 1.2)), 
                                      shear=random.uniform (0.0, 0.3), 
                                      rotation=random.uniform (0.0, 2 * math.pi))
        
        image[rect_image >= color] = rect_image[rect_image >= color]
        
    return skimage.filters.gaussian (image, sigma=3)


#--------------------------------------------------------------------------
# MAIN
#
if __name__ == '__main__':

    random.seed ()

    #
    # Parse command line arguments
    #
    parser = argparse.ArgumentParser ()
    
    parser.add_argument ('-x', '--width',  type=int, default=512, help='Width of the generated image')
    parser.add_argument ('-y', '--height', type=int, default=512, help='Height of the generated image')

    args = parser.parse_args ()

    image = generate_noisy_rects (Size2d (args.width, args.height))

    utils.show_image ([utils.to_rgb (image), 'Noisy rectangles background'])
