#!/usr/bin/python3
#
# display_sampled_image.py - Generate set of samples from image and display
#                            it in a raster
#
# Frank Blankenburg, Mar. 2017
#

import argparse
import random

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

    image = test_image.to_rgb ()
    overlay = test_image.create_result_overlay (test_image.labels)
    
    image.paste (overlay, mask=overlay)    
    image.show ()
