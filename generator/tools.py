#!/usr/bin/python3
#
# tools.py - Generator tools
#
# Frank Blankenburg, Nov. 2017
#

import argparse
import random
import math
import numpy as np
import time

import common.utils as utils

import skimage.filters
import skimage.util

#----------------------------------------------------------------------------------------------------------------------
# DECORATOR @timeit
#
# Decorator used to measure execution time of a method
def timeit (func):

    def timed (*args, **kw):
        start_time = time.time ()
        result = func (*args, **kw)
        end_time = time.time ()

        print ('Execution time of \'{funcname}\': {time} ms'.format (funcname=func.__name__, time=int ((end_time - start_time) * 1000)))

        return result

    return timed

#----------------------------------------------------------------------------------------------------------------------
# CLASS MetalTextureCreator
#
# Generator for metal textures
#
# The textures are cached for faster reusing.
#
class MetalTextureCreator:

    def __init__ (self, width, height, color, shine):
        self.width = width
        self.height = height
        self.color = color
        self.shine = shine

        self.images = []
        for _ in range (3):
            image = self.create_single_image ()
            self.images.append (np.dstack ((image, image, image)))

            flipped = image[:, ::-1]
            self.images.append (np.dstack ((flipped, flipped, flipped)))

    #------------------------------------------------------------------------------------------------------------------
    # Create a metal texture
    #
    # See http://www.jhlabs.com/ip/brushed_metal.html for the algorithm
    #
    def create (self):
        return random.choice (self.images)

    def create_single_image (self):

        rotation = np.random.uniform (0, 90.0 / 15.0) * 15.0
        offset = np.random.uniform (0, 1 / 0.25) * 0.25 * math.pi / 2

        image = np.zeros ((self.height * 2, self.width * 2, 1), dtype=np.float32)
        image[:,:] = self.color

        image = skimage.util.random_noise (image, mode='gaussian', mean=0, var=0.005)
        image = skimage.filters.gaussian (image, sigma=[0, 12, 0], mode='nearest')

        offset = np.random.uniform ()

        for y in range (image.shape[0]):
            for x in range (image.shape[1]):
                factor = self.shine * math.sin (x * math.pi / self.width - offset)
                image[y,x] += factor

        image = skimage.transform.rotate (image, rotation, resize=False)
        image = image[self.height - int (self.height / 2) : self.height + int (self.height / 2),
                      self.width  - int (self.width / 2)  : self.width  + int (self.width / 2),
                      :]

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

    parser.add_argument ('-x', '--width',   type=int,   default=640,  help='Width of the generated images')
    parser.add_argument ('-y', '--height',  type=int,   default=480,  help='Height of the generated images')
    parser.add_argument ('-c', '--color',   type=float, default=0.3,  help='Color seed')
    parser.add_argument ('-s', '--shine',   type=float, default=0.1,  help='Shine seed')

    args = parser.parse_args ()

    texture_creator = MetalTextureCreator (args.width, args.height, args.color, args.shine)
    image = texture_creator.create ()
    utils.show_image ([image, 'Metal texture'])
