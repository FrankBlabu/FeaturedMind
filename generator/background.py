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
import os
import cProfile
import pstats
import sys

import common.utils as utils

import skimage.color
import skimage.exposure
import skimage.filters
import skimage.io
import skimage.transform
import skimage.util

from abc import ABC, abstractmethod
from common.geometry import Point2d, Size2d, Rect2d


#--------------------------------------------------------------------------
# CLASS BackgroundGenerator
#
# Abstract base class for background generators
#
class BackgroundGenerator (ABC):

    @abstractmethod
    def generate (self):
        pass

    #--------------------------------------------------------------------------
    # Add args for background generator configuration to argument parser
    #
    @staticmethod
    def add_to_args_definition (parser):
        parser.add_argument ('-m', '--background_mode', action='store', choices=[NoisyRectBackgroundGenerator.TYPE, ImageBackgroundGenerator.TYPE],
                             default=NoisyRectBackgroundGenerator.TYPE, help='Background creation mode')
        parser.add_argument ('-d', '--background_directory', type=str, default=None, help='Directory for database based background generation')


    #--------------------------------------------------------------------------
    # Create background generator matching the command line arguments
    #
    @staticmethod
    def create (args):

        if args.background_mode == NoisyRectBackgroundGenerator.TYPE:
            background_generator = NoisyRectBackgroundGenerator (args)

        elif args.background_mode == ImageBackgroundGenerator.TYPE:
            background_generator = ImageBackgroundGenerator (args)

        else:
            raise RuntimeError ('Unknown background generator name \'{name}\''.arg (name=args.background_mode))

        return background_generator



#--------------------------------------------------------------------------
# CLASS NoisyRectBackgroundGenerator
#
# Generate pattern based background
#
# This class will generate an image background consisting of random, blurred,
# noisy and variated rectangles
#
class NoisyRectBackgroundGenerator (BackgroundGenerator):

    TYPE = 'rects'

    #
    # Constructor
    #
    # @param args Command line arguments
    #
    def __init__ (self, args):
        self.width = args.width
        self.height = args.height

    #
    # Generate single image
    #
    def generate (self):

        shape = (int (self.height), int (self.width), 3)

        image = np.zeros (shape, dtype=np.float32)
        image = skimage.util.random_noise (image, mode='gaussian', seed=None, clip=True, mean=0.2, var=0.0001)

        for _ in range (random.randint (30, 80)):

            rect_image = np.zeros (shape, dtype=np.float32)

            rect = Rect2d (Point2d (.2 * self.width, .45 * self.height),
                           Point2d (.8 * self.width, .55 * self.height))

            rect = rect.move_to (Point2d (random.randint (int (-.1 * self.width), int (.9 * self.width)),
                                          random.randint (int (-.1 * self.height), int (.9 * self.height))))

            color = (random.uniform (0.1, 0.7), random.uniform (0.1, 0.7), random.uniform (0.1, 0.7))
            rect.draw (rect_image, color, fill=True)

            if random.randint (0, 3) == 0:
                rect_image = skimage.util.random_noise (rect_image, mode='gaussian', seed=None, clip=True, mean=color, var=0.005)

            rect_image = utils.transform (rect_image, Size2d (self.width, self.height),
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
class ImageBackgroundGenerator (BackgroundGenerator):

    TYPE = 'imagedb'

    #
    # Constructor
    #
    # @param args Command line arguments
    #
    def __init__ (self, args):

        if args.background_directory is None:
            raise RuntimeError ('Directory must be specified when using mode \'imagedb\' (option -d)')

        path = os.path.abspath (args.background_directory)

        self.files = [os.path.join (path, file) for file in os.listdir (path) if utils.is_image (os.path.join (path, file))]
        self.width = args.width
        self.height = args.height

    #
    # Generate single inage
    #
    def generate (self):

        #
        # Read image and convert it into grayscale
        #
        image = skimage.io.imread (random.choice (self.files))

        #
        # Randomly rotate image
        #
        rotation = random.choice ([0, 90, 180, 270])
        if rotation > 0:
            image = skimage.transform.rotate (image, angle=rotation, resize=True)

        #
        # Randomly flip image
        #
        flip = random.choice (['none', 'horizontal', 'vertical'])
        if flip == 'horizontal':
            image = np.fliplr (image)
        elif flip == 'vertical':
            image = np.flipud (image)

        #
        # Scale desired target image size so that it fits into the source image in all cases.
        #
        scale = min (image.shape[0] / float (self.height), image.shape[1] / float (self.width))

        #
        # If the target image size is smaller than the source image, scale it up randomly so that
        # we do not get small image fragments only.
        #
        if scale > 1.0:
            scale = random.uniform (max (1.0, scale / 2), scale)

        crop_size = (int (self.height * scale), int (self.width * scale))

        crop_offset_y = random.randint (0, image.shape[0] - crop_size[0])
        crop_offset_x = random.randint (0, image.shape[1] - crop_size[1])

        image = image[crop_offset_y:crop_offset_y + crop_size[0], crop_offset_x:crop_offset_x + crop_size[1],:]

        image = skimage.transform.resize (image, (self.height, self.width, image.shape[2]), mode='reflect')

        return np.reshape (image, (image.shape[0], image.shape[1], image.shape[2]))




#--------------------------------------------------------------------------
# MAIN
#
if __name__ == '__main__':

    random.seed ()

    #
    # Parse command line arguments
    #
    parser = argparse.ArgumentParser ()

    parser.add_argument ('-x', '--width',   type=int,            default=512,   help='Width of the generated image')
    parser.add_argument ('-y', '--height',  type=int,            default=512,   help='Height of the generated image')
    parser.add_argument ('-p', '--profile', action='store_true', default=False, help='Profile run')

    BackgroundGenerator.add_to_args_definition (parser)

    args = parser.parse_args ()

    #
    # Instantiate generator class from command line arguments
    #
    generator = BackgroundGenerator.create (args)

    if args.profile:
        pr = cProfile.Profile ()
        pr.enable ()

    #
    # Generate image probe
    #
    image = generator.generate ()

    if args.profile:
        pr.disable ()

        stats = pstats.Stats (pr, stream=sys.stdout).sort_stats ('cumulative')
        stats.print_stats ()

    utils.show_image ([utils.to_rgb (image), 'Generated background'])
