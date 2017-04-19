#!/usr/bin/python3
#
# train_semantic_segmentation.py - Train deep learning network for finding segments
#
# Frank Blankenburg, Apr. 2017
# Based on: https://github.com/nicolov/segmentation_keras
#

import argparse
import gc
import h5py
import os
import random
import subprocess
import webbrowser

import common.metrics

from keras import optimizers
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model
from keras.callbacks import TensorBoard


def dice_coef_loss (y_true, y_pred):
    return -common.metrics.dice_coef (y_true, y_pred)

def create_model (rows, cols):
    
    inputs = Input ((rows, cols, 1))
    
    conv1 = Conv2D (32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D (32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D (pool_size=(2, 2))(conv1)

    conv2 = Conv2D (64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D (64, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D (pool_size=(2, 2))(conv2)

    conv3 = Conv2D (128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D (128, kernel_size=(3, 3), activation='relu', padding='same')(conv3)

    up4 = concatenate ([UpSampling2D (size=(2, 2))(conv3), conv2], axis=3)
    conv4 = Conv2D (64, kernel_size=(3, 3), activation='relu', padding='same')(up4)
    conv4 = Conv2D (64, kernel_size=(3, 3), activation='relu', padding='same')(conv4)

    up5 = concatenate ([UpSampling2D (size=(2, 2))(conv4), conv1], axis=3)
    conv5 = Conv2D (32, kernel_size=(3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv2D (32, kernel_size=(3, 3), activation='relu', padding='same')(conv5)

    conv6 = Conv2D (1, kernel_size=(1, 1), activation='sigmoid')(conv5)

    model = Model (inputs=[inputs], outputs=[conv6])
    model.compile (optimizer=optimizers.Adam (lr=1e-5), loss=dice_coef_loss, 
                   metrics=['accuracy', common.metrics.precision, common.metrics.recall, common.metrics.dice_coef])

    return model



#--------------------------------------------------------------------------
# MAIN
#

#
# Parse command line arguments
#
parser = argparse.ArgumentParser ()

parser.add_argument ('file',                type=str,               help='Test dataset file name')
parser.add_argument ('-e', '--epochs',      type=int, default=20,   help='Number of epochs')
parser.add_argument ('-l', '--log',         type=str, default=None, help='Log file directory')
parser.add_argument ('-o', '--output',      type=str, default=None, help='Model output file name')
parser.add_argument ('-b', '--batchsize',   type=int, default=5,    help='Number of samples per training batch')
parser.add_argument ('-t', '--tensorboard', action='store_true', default=False, help='Open log in tensorboard')
parser.add_argument ('-v', '--verbose',     action='store_true', default=False, help='Verbose output')
parser.add_argument ('-f', '--features',    type=str, default='borders', choices=['borders', 'rects', 'ellipses'], help='Feature types to detect')

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
        
images = file['images']
features = None

if args.features == 'borders':
    features = file['masks/borders']
elif args.features == 'rects':
    features = file['masks/rects']
elif args.features == 'ellipses':
    features = file['masks/ellipses']
else:
    assert False and 'Unknown feature type'
        
if images.shape[0] > features.shape[0]:
    images = images[:features.shape[0],:,:,:]

size = file.attrs['image_size'] 
        
print ("Training model...")
print ("  Number of images: ", images.shape[0])
print ("  Image size: {0}x{1}".format (size[0], size[1]))
print ("  Training for: {0}".format (args.features)) 

loggers = []
if args.log != None:
    loggers.append (TensorBoard (os.path.abspath (args.log), histogram_freq=1, write_graph=True, write_images=False))
    
model = create_model (size[1], size[0])

model.fit (images, features, batch_size=args.batchsize, epochs=args.epochs, 
           verbose=args.verbose, shuffle=True, validation_split=0.2,
           callbacks=loggers)

if args.output != None:
    model.save (os.path.abspath (args.output))

mask = model.predict (images[random.randint (0, images.shape[0]),:,:,:], verbose=args.verbose)

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

