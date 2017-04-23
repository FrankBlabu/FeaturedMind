#!/usr/bin/python3
#
# generate_border_test_data_set.py - Generate test data set for border detection
#
# Frank Blankenburg, Feb. 2017
#

import argparse
import random
import h5py
import common.log
import common.utils as utils

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
parser.add_argument ('-s', '--sample-size',       type=int, default=64,   help='Edge size of each sample in pixels')
parser.add_argument ('-l', '--log',               type=str,               help='Directory used for generating log data (HTML file and images)')

args = parser.parse_args ()

assert args.width <= 4096 and 'Training image width is too large.'
assert args.height <= 4096 and 'Training image height is too large.'

#
# Create test data sets
#
print ("Generating {0} border samples of size {1}x{1}...".format (args.number_of_samples, args.sample_size))

with h5py.File (args.file, 'w') as file:

    file.attrs['version']           = 1
    file.attrs['sample_size']       = args.sample_size
    file.attrs['image_size']        = (args.width, args.height)
    file.attrs['number_of_samples'] = args.number_of_samples
    file.attrs['HDF5_Version']      = h5py.version.hdf5_version
    file.attrs['h5py_version']      = h5py.version.version
    
    data  = file.create_dataset ('data',         (args.number_of_samples, args.sample_size, args.sample_size, 1), dtype='f', compression='lzf')
    truth = file.create_dataset ('ground_truth', (args.number_of_samples, args.sample_size, args.sample_size, 1), dtype='f', compression='lzf')
    
    logger = common.log.NoLogger ()
    if args.log:
        logger = common.log.HTMLLogger (args.log, 'Border test data set')
    
    with logger as log:

        #
        # Generate samples and masks
        #
        count_samples = 0
        count_images = 0
        
        while count_samples < args.number_of_samples:
            
            test_image = TestImage (args)
            
            image    = test_image.image
            mask, _  = test_image.get_feature_mask ()

            log.add_caption ('Image #{0}'.format (count_images))
            log.add_image (image)
            
            count_images += 1
        
            positives = []
            negatives = []
                
            log_rows = []
        
            y_offset = 0
            while y_offset + args.sample_size < args.height:
                
                log_columns_image = []
                log_columns_mask  = []
                
                x_offset = 0
                while x_offset + args.sample_size < args.width:
                    
                    rect = Rect2d (Point2d (x_offset, y_offset), Size2d (args.sample_size, args.sample_size))            
        
                    image_sample = utils.cutout (image, rect)
                    mask_sample = utils.cutout (mask, rect)
                    
                    if mask_sample.max () > 0.5:
                        positives.append ((utils.mean_center (image_sample), mask_sample))
                    else:
                        negatives.append ((utils.mean_center (image_sample), mask_sample))
                    
                    log_columns_image.append (image_sample)
                    log_columns_mask.append (mask_sample)
        
                    x_offset += args.sample_size / 2
                                        
                log_rows.append (log_columns_image)
                log_rows.append (log_columns_mask)
                                        
                y_offset += args.sample_size / 2
        
            if len (negatives) > len (positives) / 10:                
                negatives = negatives[:int (len(positives) / 10)]
                
            samples = positives
            samples.extend (negatives)
            
            random.shuffle (samples)
            
            for sample in samples:
                
                if count_samples < args.number_of_samples:
                    data[count_samples]  = sample[0]
                    truth[count_samples] = sample[1]
                
                    count_samples += 1
                    
            print (count_samples)
        
            log.add_caption ('Samples')
            log.add_table (log_rows)
            
            