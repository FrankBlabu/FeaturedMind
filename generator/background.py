#!/usr/bin/python3
#
# background.py - Background generators
#
# Frank Blankenburg, May 2017
#

import argparse
import imghdr
import random
import math
import numpy as np
import os
import common.utils as utils

import skimage.color
import skimage.exposure
import skimage.filters
import skimage.io
import skimage.transform
import skimage.util

from common.geometry import Point2d, Size2d, Rect2d

#--------------------------------------------------------------------------
# Add args for background generator configuration to argument parser
#
def add_to_args_definition (parser):
    parser.add_argument ('-m', '--background_mode',      action='store', choices=['rects', 'imagedb'], default='rects', help='Background creation mode')
    parser.add_argument ('-d', '--background_directory', type=str, default=None, help='Directory for database based background generation')


#--------------------------------------------------------------------------
# Create background generator matching the command line arguments
#
def create_from_args (args):
    if args.background_mode == 'rects':
        background_generator = NoisyRectBackgroundGenerator (Size2d (args.width, args.height))
    elif args.background_mode == 'imagedb':
        if args.background_directory is None:
            raise RuntimeError ('Directory must be specified when using mode \'imagedb\' (option -d)')

        background_generator = ImageBackgroundGenerator (args.background_directory, Size2d (args.width, args.height))

    else:
        raise RuntimeError ('Unknown background generator name  \'{0}\''.arg (args.background_mode))

    return background_generator



#--------------------------------------------------------------------------
# CLASS NoisyRectBackgroundGenerator
#
# Generate pattern based background
#
# This class will generate an image background consisting of random, blurred,
# noisy and variated rectangles
#
class NoisyRectBackgroundGenerator:

    #
    # Constructor
    #
    # @param size Desired size
    #
    def __init__ (self, size):
        self.size = size

    #
    # Generate single image
    #
    def generate (self):
        shape = (int (self.size.height), int (self.size.width), 3)

        image = np.zeros (shape, dtype=np.float32)
        image = skimage.util.random_noise (image, mode='gaussian', seed=None, clip=True, mean=0.2, var=0.0001)

        number_of_shapes = random.randint (30, 80)
        for _ in range (number_of_shapes):

            rect_image = np.zeros (shape, dtype=np.float32)

            rect = Rect2d (Point2d (2 * self.size.width / 10, 4.5 * self.size.height / 10),
                           Point2d (8 * self.size.width / 10, 5.5 * self.size.height / 10))

            rect = rect.move_to (Point2d (random.randint (int (-1 * self.size.width / 10), int (9 * self.size.width / 10)),
                                          random.randint (int (-1 * self.size.height / 10), int (9 * self.size.height / 10))))

            color = (random.uniform (0.1, 0.7), random.uniform (0.1, 0.7), random.uniform (0.1, 0.7))
            rect.draw (rect_image, color, fill=True)

            if random.randint (0, 3) == 0:
                rect_image = skimage.util.random_noise (rect_image, mode='gaussian', seed=None, clip=True, mean=color, var=0.005)

            rect_image = utils.transform (rect_image, self.size,
                                          scale=(random.uniform (0.3, 1.4), random.uniform (0.8, 1.2)),
                                          shear=random.uniform (0.0, 0.3),
                                          rotation=random.uniform (0.0, 2 * math.pi))

            mask =  rect_image[:,:,0] >= color[0]
            mask |= rect_image[:,:,1] >= color[1]
            mask |= rect_image[:,:,2] >= color[2]
            image[mask] = rect_image[mask]

        return skimage.filters.gaussian (image, sigma=3, multichannel=True)


#--------------------------------------------------------------------------
# CLASS ImageBackgroundGenerator
#
# Background generator using a file based image database
#
class ImageBackgroundGenerator:

    #
    # Constructor
    #
    # @param path Path containing a set of images
    #
    def __init__ (self, path, size):

        path = os.path.abspath (path)

        self.files = [os.path.join (path, file) for file in os.listdir (path) if self.is_image (os.path.join (path, file))]
        self.width = int (size.width)
        self.height = int (size.height)

    #
    # Generate single inage
    #
    def generate (self):

        #
        # Read image and convert it into grayscale
        #
        image = skimage.io.imread (random.choice (self.files))

        #
        # If possible (image larger than desired size), crop a reasonable part
        #
        shrink = max (int (self.height / image.shape[0]), int (self.width / image.shape[1]))
        if shrink < 1.0:
            factor = random.uniform (max (shrink, 0.5), 1.0)

            crop_width = int (image.shape[1] * factor)
            crop_height = int (image.shape[0] * factor)

            crop_offset_x = random.randint (0, image.shape[1] - crop_width)
            crop_offset_y = random.randint (0, image.shape[0] - crop_height)

            image = image[crop_offset_y:crop_offset_y + crop_height, crop_offset_x:crop_offset_x + crop_width,:]

        image = skimage.transform.resize (image, (self.height, self.width, image.shape[2]), mode='reflect')

        #
        # Randomly rotate image
        #
        flip_mode = random.randint (0, 3)
        if flip_mode == 1:
            image = skimage.transform.rotate (image, 90)
        elif flip_mode == 2:
            image = skimage.transform.rotate (image, 180)
        elif flip_mode == 3:
            image = skimage.transform.rotate (image, 270)

        #
        # Make some noise
        #
        noise = random.randint (1, 6)
        if noise < 3:
            image = skimage.util.random_noise (image, mode='gaussian', seed=None, clip=True, mean=0.5, var=0.0001 * noise)

        #
        # Blur image
        #
        blur = random.randint (0, 4)
        if blur < 2:
            image = skimage.filters.gaussian (image, sigma=blur, multichannel=True)

        return np.reshape (image, (image.shape[0], image.shape[1], image.shape[2]))


    #
    # Check if the given file is a valid image file
    #
    def is_image (self, file):
        if not os.path.isfile (file):
            return False

        file_type = imghdr.what (file)

        if file_type != 'png' and file_type != 'jpeg' and file_type != 'bmp' and file_type != 'tiff':
            return False

        return True



#--------------------------------------------------------------------------
# MAIN
#
if __name__ == '__main__':

    random.seed ()

    #
    # Parse command line arguments
    #
    parser = argparse.ArgumentParser ()

    parser.add_argument ('-x', '--width', type=int, default=512,  help='Width of the generated image')
    parser.add_argument ('-y', '--height', type=int, default=512, help='Height of the generated image')

    add_to_args_definition (parser)

    args = parser.parse_args ()

    generator = create_from_args (args)
    image = generator.generate ()

    utils.show_image ([utils.to_rgb (image), 'Generated background'])
