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
import skimage.filters
import skimage.io
import skimage.transform
import skimage.util

from common.geometry import Point2d, Size2d, Rect2d


#--------------------------------------------------------------------------
# Generate pattern based background
#
# This class will generate an image background consisting of random, blurred,
# noisy and variated rectangles
#
def generate_noisy_rects (size): 

    shape = (int (size.height), int (size.width))

    image = np.zeros (shape, dtype=np.float32)
    image = skimage.util.random_noise (image, mode='gaussian', seed=None, clip=True, mean=0.2, var=0.0001)
    
    number_of_shapes = random.randint (30, 80)
    for _ in range (number_of_shapes):

        rect_image = np.zeros (shape, dtype=np.float32)
        
        rect = Rect2d (Point2d (2 * size.width / 10, 4.5 * size.height / 10),
                       Point2d (8 * size.width / 10, 5.5 * size.height / 10))

        rect = rect.move_to (Point2d (random.randint (int (-1 * size.width / 10), int (9 * size.width / 10)),
                                      random.randint (int (-1 * size.height / 10), int (9 * size.height / 10))))

        color = random.uniform (0.02, 0.1)
        rect.draw (rect_image, color, fill=True)
        
        if random.randint (0, 3) == 0:
            rect_image = skimage.util.random_noise (rect_image, mode='gaussian', seed=None, clip=True, mean=color, var=0.005)

        rect_image = utils.transform (rect_image, size,
                                      scale=(random.uniform (0.3, 1.4), random.uniform (0.8, 1.2)), 
                                      shear=random.uniform (0.0, 0.3), 
                                      rotation=random.uniform (0.0, 2 * math.pi))
        
        image[rect_image >= color] = rect_image[rect_image >= color]
        
    return skimage.filters.gaussian (image, sigma=3)


#--------------------------------------------------------------------------
# CLASS ImageBackground
#
# Background generator using a file based image database
#
class ImageBackground:
    
    #
    # Constructor
    #
    # @param path Path containing a set of images
    #
    def __init__ (self, path, size):
        self.files = [os.path.join (path, file) for file in os.listdir (path) if self.is_image (os.path.join (path, file))]
        self.size = size

    #
    # Get random file
    #
    def get (self):
        
        #
        # Read image and resize it matching the desired background size
        #
        image = skimage.io.imread (random.choice (self.files))
        image = skimage.color.rgb2gray (image)
        image = np.reshape (image, (image.shape[0], image.shape[1], 1))
        
        image = skimage.transform.resize (image, (int (self.size.height), int (self.size.width), 1), mode='reflect')
        
        #
        # Randomly flip image        
        #
        if random.randint (0, 1) == 1:
            image = image[:,::-1,:]
        
        return image
        
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
    
    parser.add_argument ('-x', '--width', type=int, default=512,            
                         help='Width of the generated image')
    parser.add_argument ('-y', '--height', type=int, default=512,                    
                         help='Height of the generated image')
    parser.add_argument ('-m', '--mode', action='store', choices=['rects', 'imagedb'], default='rects', 
                         help='Background creation mode')
    parser.add_argument ('-d', '--directory', type=str, default=None, 
                         help='Directory for database based background generation')

    args = parser.parse_args ()

    print ('Background mode: {0}'.format (args.mode))

    if args.mode == 'rects':
        image = generate_noisy_rects (Size2d (args.width, args.height))
    elif args.mode == 'imagedb':
        assert args.directory is not None
        source = ImageBackground (args.directory, Size2d (args.width, args.height))
        image = source.get ()

    utils.show_image ([utils.to_rgb (image), 'Generated background'])
