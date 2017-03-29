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

from test_image_generator import TestImage

import PIL.Image


#----------------------------------------------------------------------------
# Display test image together with border detection flag overlay
#
def create_result_image (test_image, sample_size, result):

    #
    # Paste samples into displayable image
    #
    image = PIL.Image.new ('RGB', (test_image.width, test_image.height))
    
    draw = PIL.ImageDraw.Draw (image, 'RGBA')
    
    x_steps = int (math.floor (test_image.width / sample_size))
    y_steps = int (math.floor (test_image.height / sample_size))
    
    for y in range (0, y_steps):
        for x in range (0, x_steps):
            sample, direction, cluster = test_image.get_sample (x * sample_size, y * sample_size, sample_size)
    
            data = np.array ([int (round (d * 255)) for d in sample], np.uint8)
            sample_image = PIL.Image.frombuffer ('L', (sample_size, sample_size), data.tostring (), 'raw', 'L', 0, 1)
            sample_image = sample_image.convert ('RGBA')
            
            rect = (x * sample_size, y * sample_size, (x + 1) * sample_size, (y + 1) * sample_size)
            r = (rect[0], rect[1], rect[2] - 1, rect[3] - 1)
            
            image.paste (sample_image, rect)

            #
            # Add overlay showing the direction 
            #
            if direction != TestImage.Direction.NONE:
                color = test_image.get_color_for_direction (direction)                
                draw.rectangle (r, fill=(color[0], color[1], color[2], 0x20), outline=color)
                
            #
            # Add overlay showing if the result matches
            #
            if result is not None:
                
                result_direction = TestImage.Direction (result[y][x])
                
                if result_direction != direction:
                    color = test_image.get_color_for_direction (result_direction)
                    draw.line ((r[0], r[1], r[2], r[3]), fill=color)
                    draw.line ((r[2], r[1], r[0], r[3]), fill=color)
            
            #
            # Add overlay with the cluster id
            #    
            if cluster > 0:    
                draw.text ((rect[0], rect[1]), str (cluster))
                    
    return image



#--------------------------------------------------------------------------
# MAIN
#
if __name__ == '__main__':

    random.seed ()
    np.set_printoptions (threshold=np.nan)
    
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
