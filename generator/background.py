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

from common.geometry import Point2d, Size2d, Rect2d
from generator.generator import Generator


#----------------------------------------------------------------------------------------------------------------------
# CLASS BackgroundGenerator
#
# Abstract base class for background generators
#
class BackgroundGenerator (Generator):

    TYPE = 'background'

    def __init__ (self, args):
        super ().__init__ (args)

    #--------------------------------------------------------------------------
    # Return if this generator creates an active layer which must be detected as a separate image segmentation class
    #
    def is_active_layer (self):
        return False


    #--------------------------------------------------------------------------
    # Add args for background generator configuration to argument parser
    #
    @staticmethod
    def add_to_args_definition (parser):
        parser.add_argument ('-m', '--background_mode', action='store', choices=[NoisyRectBackgroundGenerator.TYPE,
                                                                                 ImageBackgroundGenerator.TYPE,
                                                                                 EmptyBackgroundGenerator.TYPE],
                             default=NoisyRectBackgroundGenerator.TYPE, help='Background creation mode')
        parser.add_argument ('-d', '--background_directory', type=str, default=None, help='Directory for database based background generation')


    #--------------------------------------------------------------------------
    # Create background generator matching the command line arguments
    #
    @staticmethod
    def create (args):

        if args.background_mode == EmptyBackgroundGenerator.TYPE:
            generator = EmptyBackgroundGenerator (args)

        elif args.background_mode == NoisyRectBackgroundGenerator.TYPE:
            generator = NoisyRectBackgroundGenerator (args)

        elif args.background_mode == ImageBackgroundGenerator.TYPE:
            generator = ImageBackgroundGenerator (args)

        else:
            raise RuntimeError ('Unknown background generator name \'{name}\''.arg (name=args.background_mode))

        return generator



#----------------------------------------------------------------------------------------------------------------------
# CLASS EmptyBackgroundGenerator
#
# Generate empty background
#
class EmptyBackgroundGenerator (BackgroundGenerator):

    TYPE = 'none'

    #
    # Constructor
    #
    # @param args Command line arguments
    #
    def __init__ (self, args):
        super ().__init__ (args)

    #
    # Generate single image
    #
    def generate (self):
        return np.zeros ((self.height, self.width, 3), dtype=np.float32), None


#----------------------------------------------------------------------------------------------------------------------
# CLASS NoisyRectBackgroundGenerator
#
# Generate pattern based background
#
# This class will generate an image background consisting of random, blurred, noisy and variated rectangles
#
class NoisyRectBackgroundGenerator (BackgroundGenerator):

    TYPE = 'rects'

    #
    # Constructor
    #
    # @param args Command line arguments
    #
    def __init__ (self, args):
        super ().__init__ (args)

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

            mask = rect_image[:,:,0] >= color[0]
            mask |= rect_image[:,:,1] >= color[1]
            mask |= rect_image[:,:,2] >= color[2]
            image[mask] = rect_image[mask]

        return skimage.filters.gaussian (image, sigma=3, multichannel=True), None


#----------------------------------------------------------------------------------------------------------------------
# CLASS ImageBackgroundGenerator
#
# Background generator using a file based image database
#
# This generator class reads images from a database directory, selects appropriate parts matching the desired target
# resolution and transforms the results to get a larger base for the training set.
#
class ImageBackgroundGenerator (BackgroundGenerator):

    #
    # Type id for command line arguments
    #
    TYPE = 'imagedb'

    #
    # Constructor
    #
    # @param args Command line arguments
    #
    def __init__ (self, args):
        super ().__init__ (args)

        if args.background_directory is None:
            raise RuntimeError ('Directory must be specified when using mode \'imagedb\' (option -d)')

        path = os.path.abspath (args.background_directory)

        self.files = [os.path.join (path, file) for file in os.listdir (path) if utils.is_image (os.path.join (path, file))]

    #
    # Generate single image matching the given command line argument parameters
    #
    def generate (self):

        #
        # Read image
        #
        image = skimage.io.imread (random.choice (self.files))

        #
        # Randomly rotate image in 90 degree steps
        #
        rotation = random.choice ([0, 90, 180, 270])
        if rotation > 0:
            image = skimage.transform.rotate (image, angle=rotation, resize=True)

        #
        # Randomly flip image around the horizontal/vertical axis
        #
        flip = random.choice (['none', 'horizontal', 'vertical'])
        if flip == 'horizontal':
            image = np.fliplr (image)
        elif flip == 'vertical':
            image = np.flipud (image)

        #
        # Compute scale factor needed to squeeze / enlarge the source image to the target image size
        #
        scale = min (image.shape[0] / float (self.height), image.shape[1] / float (self.width))

        #
        # If the source image is larger than the target image, scale it down randomly so that we do not get small
        # image fragments only.
        #
        if scale > 1.0:
            scale = random.uniform (max (1.0, scale / 2), scale)

        #
        # Compute crop area and extract image part
        #
        crop_size = (int (self.height * scale), int (self.width * scale))

        crop_offset_y = random.randint (0, image.shape[0] - crop_size[0])
        crop_offset_x = random.randint (0, image.shape[1] - crop_size[1])

        image = image[crop_offset_y:crop_offset_y + crop_size[0], crop_offset_x:crop_offset_x + crop_size[1], :]

        #
        # The cropped image part can still be smaller that the target image and has to be scaled up accordingly
        #
        image = skimage.transform.resize (image, (self.height, self.width, image.shape[2]), mode='reflect')

        return np.reshape (image, (image.shape[0], image.shape[1], image.shape[2])), None




#----------------------------------------------------------------------------------------------------------------------
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
    image, mask = generator.generate ()
    assert mask is None

    if args.profile:
        pr.disable ()

        stats = pstats.Stats (pr, stream=sys.stdout).sort_stats ('cumulative')
        stats.print_stats ()

    utils.show_image ([image, 'Generated background'])
