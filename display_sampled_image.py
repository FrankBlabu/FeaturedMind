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

import PIL.Image


#----------------------------------------------------------------------------
# Display test image together with border detection flag overlay
#
def create_result_image (test_image, sample_size, result):

    assert type (sample_size) is Size2d

    #
    # Paste samples into displayable image
    #
    image = PIL.Image.new ('RGB', (test_image.width, test_image.height))
    
    draw = PIL.ImageDraw.Draw (image, 'RGBA')
    
    x_steps = int (math.floor (test_image.width / sample_size.width))
    y_steps = int (math.floor (test_image.height / sample_size.height))
    
    for y in range (0, y_steps):
        for x in range (0, x_steps):
            rect = Rect2d (Point2d (x * sample_size.width, y * sample_size.height), sample_size)
            
            sample, label = test_image.get_sample (rect)
    
            data = np.array ([int (round (d * 255)) for d in sample], np.uint8)
            
            sample_image = PIL.Image.frombuffer ('L', (sample_size.width, sample_size.height), data.tostring (), 'raw', 'L', 0, 1)
            sample_image = sample_image.convert ('RGBA')
            
            image.paste (sample_image, (rect + Size2d (1, 1)).as_tuple ())

            #
            # Add overlay showing the label 
            #
            if label > 0:

                #
                # Add overlay showing if the result matches
                #
                if result is not None:
                    
                    #
                    # Case 1: Miss
                    #
                    if (result[y][x] == 0) != (label == 0):
                        #
                        # Case 1.1: False positive
                        #
                        if result[y][x] > 0:
                            draw.rectangle (rect.as_tuple (), fill=(0x00, 0x00, 0xff, 0x20), outline=(0x00, 0x00, 0xff))
                            
                        #
                        # Case 1.2: False negative
                        #
                        else:
                            draw.rectangle (rect.as_tuple (), fill=(0xff, 0x00, 0x00, 0x20), outline=(0xff, 0x00, 0x00))
                        
                    #
                    # Case 2: Hit
                    #
                    else:
                        draw.rectangle (rect.as_tuple (), fill=(0x00, 0xff, 0x00, 0x20), outline=(0x00, 0xff, 0x00))
                        
                
                #
                # Overlay just showing the samples
                #
                else:
                    draw.rectangle (rect.as_tuple (), fill=(0x00, 0xff, 0x00, 0x20), outline=(0x00, 0xff, 0x00))
                                    

            
            #
            # Add overlay with the cluster id
            #    
            if result is not None and result[y][x] > 0:
                draw.text (rect.p0.as_tuple (), str (result[y][x]))
                    
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

    image = create_result_image (test_image, Size2d (args.sample_size, args.sample_size), None)
    image.show ()
