#!/usr/bin/python3
#
# display_sampled_image.py - Generate set of samples from image and display
#                            it in a raster
#
# Frank Blankenburg, Mar. 2017
#

import argparse
import numpy as np
import random

from common.geometry import Point2d, Size2d, Rect2d
from test_image_generator import TestImage

import PIL.Image

#----------------------------------------------------------------------------
# Create image displaying the samples from a test image class
#
def create_test_image (test_image):

    #
    # Paste samples into displayable image
    #
    image = PIL.Image.new ('RGB', (int (test_image.size.width), int (test_image.size.height)))
    
    draw = PIL.ImageDraw.Draw (image, 'RGBA')
    
    for y in range (test_image.samples.shape[0]):
        for x in range (test_image.samples.shape[1]):
            sample = test_image.samples[y][x]
    
            data = np.array ([int (round (d * 255)) for d in sample], np.uint8)
            
            sample_image = PIL.Image.frombuffer ('L', (int (test_image.sample_size.width), int (test_image.sample_size.height)), data.tostring (), 'raw', 'L', 0, 1)
            sample_image = sample_image.convert ('RGB')
            
            rect = Rect2d (Point2d (x * test_image.sample_size.width, y * test_image.sample_size.height), test_image.sample_size)
            image.paste (sample_image, (rect + Size2d (1, 1)).as_tuple ())
            
    return image


#----------------------------------------------------------------------------
# Create result overlay
#
def create_result_overlay (test_image, labels):

    image = PIL.Image.new ('RGBA', (int (test_image.size.width), int (test_image.size.height)))
    
    draw = PIL.ImageDraw.Draw (image, 'RGBA')

    assert test_image.labels.shape == labels.shape
    
    for y in range (labels.shape[0]):
        for x in range (labels.shape[1]):
            rect = Rect2d (Point2d (x * test_image.sample_size.width, y * test_image.sample_size.height), test_image.sample_size)
            
            expected = test_image.labels[y][x]
            found    = labels[y][x]

            #
            # Case 1: Hit
            #
            if expected > 0 and found > 0:
                draw.rectangle (rect.as_tuple (), fill=(0x00, 0xff, 0x00, 0x20), outline=(0x00, 0xff, 0x00))
                
            #
            # Case 2: False positive
            #
            elif expected == 0 and found > 0:
                draw.rectangle (rect.as_tuple (), fill=(0x00, 0x00, 0xff, 0x20), outline=(0x00, 0x00, 0xff))
                
            #
            # Case 3: False negative
            #
            elif expected > 0 and found == 0:
                draw.rectangle (rect.as_tuple (), fill=(0xff, 0x00, 0x00, 0x20), outline=(0xff, 0x00, 0x00))
            
            #
            # Add overlay with the cluster id
            #    
            if found > 0:
                draw.text (rect.p0.as_tuple (), str (int (found)))
                    
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

    image = create_test_image (test_image)
    overlay = create_result_overlay (test_image, test_image.labels)
    
    image.paste (overlay, mask=overlay)    
    image.show ()
