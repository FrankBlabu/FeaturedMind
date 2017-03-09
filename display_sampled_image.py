#!/usr/bin/python3
#
# display_sampled_image.py - Generate set of samples from image and display
#                            it in a raster
#
# Frank Blankenburg, Mar. 2017
#

import argparse
import io
import math
import numpy as np
import random

from test_image_generator import TestImage

import PIL.Image


#----------------------------------------------------------------------------
# Display test image together with border detection flag overlay
#
def create_result_image (test_image, sample_size, border_flags):

    #
    # Paste samples into displayable image
    #
    image = PIL.Image.new ('RGB', (test_image.width, test_image.height))
    
    draw = PIL.ImageDraw.Draw (image, 'RGBA')
    
    for y in range (0, int (math.floor (test_image.height / sample_size))):
        for x in range (0, int (math.floor (test_image.width / sample_size))):
            sample, generated_flag = test_image.get_sample (x * sample_size, y * sample_size, sample_size)
    
            data = np.array ([int (round (d * 255)) for d in sample], np.uint8)
            sample_image = PIL.Image.frombuffer ('L', (sample_size, sample_size), data.tostring (), 'raw', 'L', 0, 1)
            sample_image = sample_image.convert ('RGBA')
            
            rect = (x * sample_size, y * sample_size, (x + 1) * sample_size, (y + 1) * sample_size)
            r = (rect[0], rect[1], rect[2] - 1, rect[3] - 1)
            
            image.paste (sample_image, rect)
            
            #
            # Add overlay showing the generated 'border flag' status
            #
            if generated_flag:
                draw.rectangle (r, fill=(0x00, 0xff, 0x00, 0x20), outline=(0x00, 0xff, 0x00))
            
            if border_flags != None and border_flags[y][x]:
                draw.rectangle (r, fill=None, outline=(0xff, 0x00, 0x00))
                
            
                
    return image



#--------------------------------------------------------------------------
# MAIN
#
if __name__ == '__main__':

    random.seed ()
    
    #
    # Parse command line arguments
    #
    parser = argparse.ArgumentParser ()
    
    parser.add_argument ('-x', '--width',       type=int, default=640,  help='Width of the generated images')
    parser.add_argument ('-y', '--height',      type=int, default=480,  help='Height of the generated images')
    parser.add_argument ('-s', '--sample-size', type=int, default=32,   help='Edge size of each sample in pixels')
    
    args = parser.parse_args ()
    
    if args.width > 2048:
        assert 'Training image width is too large.'
    if args.height > 2048:
        assert 'Training image height is too large.'
    
    #
    # Create test data set
    #
    test_image = TestImage (args.width, args.height)

    image = create_result_image (test_image, args.sample_size, None)
    image.show ()
    