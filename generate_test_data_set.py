#!/usr/bin/python3
#
# generate_test_data_set.py - Generate test data set
#
# Frank Blankenburg, Feb. 2017
#

import argparse
import random
import h5py
import common.utils as utils

from test_image_generator import TestImage


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
parser.add_argument ('-n', '--number-of-images', type=int, default=1000, help='Number of images to generate')
parser.add_argument ('-l', '--log',              type=str,               help='Directory used for generating log data (HTML file and images)')

args = parser.parse_args ()

assert args.width <= 4096 and 'Training image width is too large.'
assert args.height <= 4096 and 'Training image height is too large.'

#
# Create test data sets
#
print ("Generating {0} images of size {1}x{2}...".format (args.number_of_images, args.width, args.height))

with h5py.File (args.file, 'w') as file:

    file.attrs['version']          = 1
    file.attrs['image_size']       = (args.width, args.height)
    file.attrs['number_of_images'] = args.number_of_images
    file.attrs['HDF5_Version']     = h5py.version.hdf5_version
    file.attrs['h5py_version']     = h5py.version.version
    
    data  = file.create_dataset ('data', (args.number_of_images, args.height, args.width, 1), dtype='f', compression='lzf')
    truth = file.create_dataset ('mask', (args.number_of_images, args.height, args.width, 1), dtype='f', compression='lzf')

    args.sample_size = 64

    for count in range (args.number_of_images):
        test_image = TestImage (args)
            
        image = utils.mean_center (test_image.image)
        image = image.reshape (args.height, args.width, 1)
        
        mask = test_image.get_specimen_mask ()
        mask = mask.reshape (args.height, args.width, 1)
            
        data[count] = image
        truth[count] = mask

        count += 1
        print ('{0} / {1}'.format (count, args.number_of_images))
            