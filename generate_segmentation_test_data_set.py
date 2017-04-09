#!/usr/bin/python3
#
# generate_segmentation_test_data_set.py - Generate test data set for semantic segmentation
#
# Frank Blankenburg, Apr. 2017
#

import argparse
import random
import h5py
import numpy as np

from common.geometry import Polygon2d, Rect2d, Ellipse2d
from test_image_generator import TestImage

#
# Convert image into TensorFlow compatible numpy array
#
def to_tf_format (args, image):
    return np.asarray ([float (d) / 255 for d in image.getdata ()], dtype=np.float32).reshape ((args.height, args.width, 1))


#--------------------------------------------------------------------------
# MAIN
#

random.seed ()

#
# Parse command line arguments
#
parser = argparse.ArgumentParser ()

parser.add_argument ('file',                     type=str,               help='Output file name')
parser.add_argument ('-x', '--width',            type=int, default=640,  help='Width of the generated images')
parser.add_argument ('-y', '--height',           type=int, default=480,  help='Height of the generated images')
parser.add_argument ('-n', '--number-of-images', type=int, default=5000, help='Number of samples to generate')

args = parser.parse_args ()

assert args.width <= 4096 and 'Training image width is too large.'
assert args.height <= 4096 and 'Training image height is too large.'

#
# Create test data sets
#
print ("Generating {0} images of size {1}x{2}...".format (args.number_of_images, args.width, args.height))

file = h5py.File (args.file, 'w')

file.attrs['version']          = 1
file.attrs['image_size']       = (args.width, args.height)
file.attrs['number_of_images'] = args.number_of_images
file.attrs['HDF5_Version']     = h5py.version.hdf5_version
file.attrs['h5py_version']     = h5py.version.version

args.sample_size = 16

images = file.create_dataset ('images', (args.number_of_images, args.height, args.width, 1), dtype='f', compression='lzf')

mask_borders = file.create_dataset ('masks_borders', (args.number_of_images, args.height, args.width, 1), 
                                    dtype='f', compression='lzf')
mask_rects  = file.create_dataset ('masks_rects', (0, args.height, args.width, 1), 
                                    maxshape=(args.number_of_images, args.height, args.width, 1), 
                                    dtype='f', compression='lzf')
mask_ellipses = file.create_dataset ('masks_ellipses', (0, args.height, args.width, 1), 
                                     maxshape=(args.number_of_images, args.height, args.width, 1), 
                                     dtype='f', compression='lzf')

for i in range (args.number_of_images):

    image = TestImage (args)
    
    print ('{0}/{1}'.format (i + 1, args.number_of_images))
    
    #
    # Image
    #
    images[i] = to_tf_format (args, image.image)
    
    #
    # Borders
    #
    mask, valid = image.get_cluster_mask (Polygon2d)
    assert valid

    mask_borders[i] = to_tf_format (args, mask)     
    
    #
    # Rectangles
    #
    mask, valid = image.get_cluster_mask (Rect2d)

    if valid:
        count = mask_rects.shape[0]
        mask_rects.resize (count + 1, axis=0)
        mask_rects[count] = to_tf_format (args, mask)     

    #
    # Ellipses
    #
    mask, valid = image.get_cluster_mask (Ellipse2d)

    if valid:
        count = mask_ellipses.shape[0]
        mask_ellipses.resize (count + 1, axis=0)
        mask_ellipses[count] = to_tf_format (args, mask)     

print ("Number of rectangle masks:", mask_rects.shape[0])
print ("Number of ellipse masks  :", mask_ellipses.shape[0])

file.flush ()
file.close ()

print ("Done")
