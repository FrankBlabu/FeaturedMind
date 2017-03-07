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

from test_image_generator import TestImage

import PIL.Image

#--------------------------------------------------------------------------
# MAIN
#

random.seed ()

#
# Parse command line arguments
#
parser = argparse.ArgumentParser ()

parser.add_argument ('file', type=str, help='Output file name')
parser.add_argument ('-x', '--width', type=int, default=640,
                     help='Width of the generated images')
parser.add_argument ('-y', '--height', type=int, default=480,
                     help='Height of the generated images')
parser.add_argument ('-n', '--number-of-samples', type=int, default=1000,
                     help='Number of samples to generate')
parser.add_argument ('-s', '--sample-size', type=int, default=32,
                     help='Edge size of each sample in pixels')

args = parser.parse_args ()

if args.width > 2048:
    assert 'Training image width is too large.'
if args.height > 2048:
    assert 'Training image height is too large.'

#
# Create test data sets
#
print ("Generating samples...")

file = h5py.File (args.file, 'w')

file.attrs['version']           = 1
file.attrs['sample_size']       = args.sample_size
file.attrs['image_size']        = (args.width, args.height)
file.attrs['number_of_samples'] = args.number_of_samples
file.attrs['HDF5_Version']      = h5py.version.hdf5_version
file.attrs['h5py_version']      = h5py.version.version

data = file.create_dataset ('data', (args.number_of_samples, args.sample_size * args.sample_size), 
                            dtype='f', compression='lzf')
labels = file.create_dataset ('labels', (args.number_of_samples, 2), dtype='f', compression='lzf')

count = 0
while count < args.number_of_samples:
    
    image = TestImage (args.width, args.height)

    positive_samples = []
    negative_samples = []
     
    for y in range (0, int (math.floor (args.height / args.sample_size))):
        for x in range (0, int (math.floor (args.width / args.sample_size))):
            sample, flag = image.get_sample (x * args.sample_size, y * args.sample_size, args.sample_size)
            
            if flag:
                positive_samples.append ((sample, True))
            else:
                negative_samples.append ((sample, False))
            
    random.shuffle (positive_samples)
    random.shuffle (negative_samples)
    
    if len (negative_samples) > len (positive_samples):
        negative_samples = negative_samples[0:len (positive_samples)]

    samples = positive_samples + negative_samples 
    random.shuffle (samples)
    
    for sample in samples:

        data[count] = sample[0]
        labels[count] = [0 if sample[1] else 1, 1 if sample[1] else 0]

        count += 1
        if count == args.number_of_samples:
            break

    print (len (samples), count)

file.flush ()
file.close ()

print ("Done")
