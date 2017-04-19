#!/usr/bin/python3
#
# generate_border_test_data_set.py - Generate test data set for border detection
#
# Frank Blankenburg, Feb. 2017
#

import argparse
import random
import h5py
import common.utils as utils

from common.geometry import Point2d, Size2d, Rect2d
from test_image_generator import TestImage
from skimage.color import gray2rgb

#--------------------------------------------------------------------------
# Local functions
#
def cutout (image, area):
    r = area.as_tuple ()
    result = image[r[1]:r[3]+1,r[0]:r[2]+1]    
    return result.reshape ((result.shape[0], result.shape[1], 1))


#--------------------------------------------------------------------------
# MAIN
#

random.seed ()

#
# Parse command line arguments
#
parser = argparse.ArgumentParser ()

parser.add_argument ('file',                      type=str,               help='Output file name')
parser.add_argument ('-x', '--width',             type=int, default=1024, help='Width of the generated images')
parser.add_argument ('-y', '--height',            type=int, default=768,  help='Height of the generated images')
parser.add_argument ('-n', '--number-of-samples', type=int, default=5000, help='Number of samples to generate')
parser.add_argument ('-s', '--sample-size',       type=int, default=64,   help='Edge size of each sample in pixels')

args = parser.parse_args ()

assert args.width <= 4096 and 'Training image width is too large.'
assert args.height <= 4096 and 'Training image height is too large.'

#
# Create test data sets
#
print ("Generating {0} border samples of size {1}x{1}...".format (args.number_of_samples, args.sample_size))

file = h5py.File (args.file, 'w')

file.attrs['version']           = 1
file.attrs['sample_size']       = args.sample_size
file.attrs['image_size']        = (args.width, args.height)
file.attrs['number_of_samples'] = args.number_of_samples
file.attrs['HDF5_Version']      = h5py.version.hdf5_version
file.attrs['h5py_version']      = h5py.version.version

data  = file.create_dataset ('data',         (args.number_of_samples, args.sample_size, args.sample_size, 1), dtype='f', compression='lzf')
truth = file.create_dataset ('ground_truth', (args.number_of_samples, args.sample_size, args.sample_size, 1), dtype='f', compression='lzf')

displayed_images = []

count = 0
while count < args.number_of_samples:
    
    test_image = TestImage (args)
    
    image    = test_image.image
    mask, _  = test_image.get_feature_mask ()

    y_offset = 0
    while y_offset + args.sample_size < args.height and count < args.number_of_samples:
        
        x_offset = 0
        while x_offset + args.sample_size < args.width and count < args.number_of_samples:
            
            rect = Rect2d (Point2d (x_offset, y_offset), Size2d (args.sample_size, args.sample_size))            

            image_sample = cutout (image, rect)
            mask_sample = cutout (mask, rect)
            
            data[count]  = utils.mean_center (image_sample)
            truth[count] = utils.mean_center (mask_sample)

            displayed_images.append ((gray2rgb (image_sample.reshape (args.sample_size, args.sample_size)), 'Image {0}'.format (count)))
            displayed_images.append ((gray2rgb (mask_sample.reshape (args.sample_size, args.sample_size)), 'Mask {0}'.format (count)))

            count += 1
            x_offset += args.sample_size / 2
        
        print (count)
                
        y_offset += args.sample_size / 2


#utils.show_image (displayed_images[40], displayed_images[41], displayed_images[42], displayed_images[43])

file.flush ()
file.close ()

