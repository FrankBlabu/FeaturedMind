#!/usr/bin/python3
#
# train_border_detection.py - Train net to recognize border structures
#                             in feature images
#
# Frank Blankenburg, Mar. 2017
#

import argparse
import gc
import h5py
import os
import subprocess
import webbrowser

import keras

from keras import optimizers
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model
from keras.callbacks import TensorBoard

import common.metrics


#--------------------------------------------------------------------------
# Generate model
#
# @param sample_size Size of a single sample in pixels
#
def create_model (sample_size):
    
    inputs = Input ((sample_size, sample_size, 1))
    
    conv1 = Conv2D (32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D (32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D (pool_size=(2, 2))(conv1)

    conv2 = Conv2D (64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D (64, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D (pool_size=(2, 2))(conv2)

    conv3 = Conv2D (128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D (128, kernel_size=(3, 3), activation='relu', padding='same')(conv3)    
    pool3 = MaxPooling2D (pool_size=(2, 2))(conv3)

    conv4 = Conv2D (256, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D (256, kernel_size=(3, 3), activation='relu', padding='same')(conv4)

    up5 = concatenate ([UpSampling2D (size=(2, 2))(conv4), conv3], axis=3)
    conv5 = Conv2D (128, kernel_size=(3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv2D (128, kernel_size=(3, 3), activation='relu', padding='same')(conv5)
    
    up6 = concatenate ([UpSampling2D (size=(2, 2))(conv5), conv2], axis=3)
    conv6 = Conv2D (64, kernel_size=(3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D (64, kernel_size=(3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate ([UpSampling2D (size=(2, 2))(conv6), conv1], axis=3)
    conv7 = Conv2D (32, kernel_size=(3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D (32, kernel_size=(3, 3), activation='relu', padding='same')(conv7)

    conv8 = Conv2D (1, kernel_size=(1, 1), activation='sigmoid')(conv7)

    model = Model (inputs=[inputs], outputs=[conv8])
    model.compile (optimizer=optimizers.Adam (lr=1e-5), loss=keras.losses.mean_squared_error, 
                   metrics=['accuracy', common.metrics.precision, common.metrics.recall])

    return model


        
#--------------------------------------------------------------------------
# MAIN
#

#
# Parse command line arguments
#
parser = argparse.ArgumentParser ()

parser.add_argument ('file',                type=str,               help='Test dataset file name')
parser.add_argument ('-e', '--epochs',      type=int, default=50,   help='Number of epochs')
parser.add_argument ('-l', '--log',         type=str, default=None, help='Log file directory')
parser.add_argument ('-o', '--output',      type=str, default=None, help='Model output file name')
parser.add_argument ('-b', '--batchsize',   type=int, default=128,  help='Number of samples per training batch')
parser.add_argument ('-t', '--tensorboard', action='store_true', default=False, help='Open log in tensorboard')
parser.add_argument ('-v', '--verbose',     action='store_true', default=False, help='Verbose output')

args = parser.parse_args ()

assert not args.tensorboard or args.log

#
# Delete old log files
#
if args.log:
    for root, dirs, files in os.walk (args.log, topdown=False):
        for name in files:
            if name.startswith ('events'):
                os.remove (os.path.join (root, name))

#
# Load sample data
#
file = h5py.File (args.file, 'r')

data         = file['data']
ground_truth = file['ground_truth']

sample_size = file.attrs['sample_size']

assert len (data) == len (ground_truth)

print ("Training model...")
print ("  Number of samples: ", data.shape[0])
print ("  Sample size      : ", sample_size)

loggers = []
if args.log != None:
    loggers.append (TensorBoard (os.path.abspath (args.log), histogram_freq=1, write_graph=True, write_images=False))
    
model = create_model (sample_size)

model.fit (data, ground_truth, batch_size=args.batchsize, epochs=args.epochs, 
           verbose=args.verbose, shuffle=True, validation_split=0.2,
           callbacks=loggers)

if args.output != None:
    model.save (os.path.abspath (args.output))

#
# Display result in tensorboard / browser
#
if args.tensorboard:
    process = subprocess.Popen (['tensorboard', '--logdir={0}'.format (args.log)])
    webbrowser.open ('http://localhost:6006', new=2)
    input ("Press [Enter] to continue...")
    process.terminate ()

file.close ()

#
# Tensorflow termination bug workaround
#
gc.collect ()

