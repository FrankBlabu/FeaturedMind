#!/usr/bin/python3
#
# generate_test_data_set.py - Generate test data set
#
# Frank Blankenburg, Feb. 2017
#

import argparse
import math
import random
import h5py

from common.geometry import Point2d, Size2d, Rect2d
from test_image_generator import TestImage

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
parser.add_argument ('-s', '--sample-size',       type=int, default=16,   help='Edge size of each sample in pixels')

args = parser.parse_args ()

assert args.width <= 4096 and 'Training image width is too large.'
assert args.height <= 4096 and 'Training image height is too large.'
assert args.sample_size <= 64 and 'Sample size is unusual large.'

#
# Create test data sets
#
print ("Generating {0} samples of size {1}x{1}...".format (args.number_of_samples, args.sample_size))

file = h5py.File (args.file, 'w')

file.attrs['version']           = 1
file.attrs['sample_size']       = args.sample_size
file.attrs['image_size']        = (args.width, args.height)
file.attrs['number_of_samples'] = args.number_of_samples
file.attrs['HDF5_Version']      = h5py.version.hdf5_version
file.attrs['h5py_version']      = h5py.version.version

data    = file.create_dataset ('data',    (args.number_of_samples, args.sample_size * args.sample_size), dtype='f', compression='lzf')
labels  = file.create_dataset ('labels',  (args.number_of_samples,), dtype='i', compression='lzf')
classes = file.create_dataset ('classes', (args.number_of_samples,), dtype='i', compression='lzf')

count = 0
while count < args.number_of_samples:
    
    image = TestImage (args)

    positive_samples = []
    negative_samples = []
     
    for y in range (image.samples.shape[0]):
        for x in range (image.samples.shape[1]):
            
            sample = image.samples[y][x]
            label = image.labels[y][x]
            
            if label > 0:
                positive_samples.append ((sample, label))
            else:
                negative_samples.append ((sample, label))
            
    random.shuffle (positive_samples)
    random.shuffle (negative_samples)
    
    if len (negative_samples) > len (positive_samples):
        negative_samples = negative_samples[0:len (positive_samples)]

    samples = positive_samples + negative_samples 
    random.shuffle (samples)
    
    for sample in samples:

        data[count]    = sample[0]
        labels[count]  = 1 if sample[1] > 0 else 0 
        classes[count] = sample[1]

        count += 1
        if count == args.number_of_samples:
            break

    print (len (samples), count)

file.flush ()
file.close ()

print ("Done")
