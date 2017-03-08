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

from test_image_generator import TestImage

import PIL.Image

#--------------------------------------------------------------------------
# MAIN
#

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

#
# Paste samples into displayable image
#
image = PIL.Image.new ('RGB', (args.width, args.height))
sample_size = args.sample_size

draw = PIL.ImageDraw.Draw (image, 'RGBA')

for y in range (0, int (math.floor (args.height / sample_size))):
    for x in range (0, int (math.floor (args.width / sample_size))):
        sample, flag = test_image.get_sample (x * sample_size, y * sample_size, sample_size)

        data = np.array ([int (round (d * 255)) for d in sample], np.uint8)
        sample_image = PIL.Image.frombuffer ('L', (sample_size, sample_size), data.tostring (), 'raw', 'L', 0, 1)
        sample_image = sample_image.convert ('RGBA')
        
        rect = (x * sample_size, y * sample_size, (x + 1) * sample_size, (y + 1) * sample_size)
        
        image.paste (sample_image, rect)
        
        #
        # Add overlay showing the 'border flag' status
        #
        if flag:
            r = (rect[0], rect[1], rect[2] - 1, rect[3] - 1)
            draw.rectangle (r, fill=(0x00, 0xff, 0x00, 0x20), outline=(0x00, 0xff, 0x00))



#
# Show generated image
#
image.show () 
