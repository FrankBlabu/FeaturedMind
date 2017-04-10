#!/usr/bin/python3
#
# display_sampled_image.py - Generate set of samples from image and display
#                            it in a raster
#
# Frank Blankenburg, Mar. 2017
#

import argparse
import math
import numpy as np
import random

from common.geometry import Point2d, Size2d, Rect2d
from test_image_generator import TestImage


#--------------------------------------------------------------------------
# MAIN
#
if __name__ == '__main__':

    random.seed ()
    
    #
    # Parse command line arguments
    #
    parser = argparse.ArgumentParser ()
    
    parser.add_argument ('-x', '--width',       type=int, default=1024, help='Width of the generated images')
    parser.add_argument ('-y', '--height',      type=int, default=768,  help='Height of the generated images')
    parser.add_argument ('-s', '--sample-size', type=int, default=16,   help='Edge size of each sample in pixels')
    
    args = parser.parse_args ()
    
    if args.width > 2048:
        assert 'Training image width is too large.'
    if args.height > 2048:
        assert 'Training image height is too large.'
    
    #
    # Create test data set
    #
    test_image = TestImage (args)

    x_steps = int (math.floor (args.width / args.sample_size))
    y_steps = int (math.floor (args.height / args.sample_size))
        
    labels = np.zeros ((y_steps, x_steps))
        
    for y in range (y_steps):
        for x in range (x_steps):
                
            rect = Rect2d (Point2d (x * args.sample_size, y * args.sample_size), Size2d (args.sample_size, args.sample_size))            
            _, labels[y][x] = test_image.get_sample (rect)


    image = test_image.to_rgb ()
    overlay = test_image.create_result_overlay (labels)
    
    image.paste (overlay, mask=overlay)    
    image.show ()
