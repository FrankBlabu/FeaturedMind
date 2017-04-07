#!/usr/bin/python3
#
# generate_segmentation_test_data_set.py - Generate test data set for semantic segmentation
#
# Frank Blankenburg, Apr. 2017
#

import argparse
import random
import h5py

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
parser.add_argument ('-x', '--width',            type=int, default=1024, help='Width of the generated images')
parser.add_argument ('-y', '--height',           type=int, default=768,  help='Height of the generated images')
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

data   = file.create_dataset ('data',   (args.number_of_images, args.width * args.height), dtype='f', compression='lzf')
labels = file.create_dataset ('labels', (args.number_of_images, args.width * args.height), dtype='f', compression='lzf')

for i in range (args.number_of_images):

    image = TestImage (args)

    data[i]   = image.get_image_data ()
    labels[i] = image.get_clustering_data () 

file.flush ()
file.close ()

print ("Done")
