#!/usr/bin/python3
#
# specimen_detection.py - Train net to recognize sheet metals in different environments
#
# Frank Blankenburg, Mar. 2017
#
# See https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pd
#     https://github.com/fchollet/keras/issues/2526
#

import argparse
import gc
import os
import subprocess
import webbrowser
import numpy as np

from keras import optimizers
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

import common.losses
import common.metrics
import common.utils as utils

from generator.sheetmetal import SheetMetalGenerator


#--------------------------------------------------------------------------
# Generate model
#
# @param width  Width of the image in pixels
# @param height Height of the image in pixels
#
def create_model (width, height):
    
    inputs = Input ((height, width, 1))
    
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
    model.compile (optimizer=optimizers.Adam (lr=1e-5),
                   loss=common.losses.dice_coef, 
                   metrics=['accuracy', 
                            common.metrics.precision, 
                            common.metrics.recall, 
                            common.metrics.f1_score,
                            common.metrics.dice_coef])

    return model


#--------------------------------------------------------------------------
# Generator
#
def sheet_metal_generator (width, height, batch_size):
    while True:
        
        images = np.zeros ((batch_size, height, width, 1), dtype=np.float32)
        masks  = np.zeros ((batch_size, height, width, 1), dtype=np.float32)
        
        for i in range (batch_size):
            sheet = SheetMetalGenerator (width, height)
            
            image = utils.mean_center (sheet.image)
            image = image.reshape ((image.shape[0], image.shape[1], 1))
            images[i] = image
            
            mask = sheet.mask
            mask = mask.reshape ((mask.shape[0], mask.shape[1], 1))
            masks[i] = mask
        
        yield (images, masks)
    
    

        
#--------------------------------------------------------------------------
# MAIN
#

#
# Parse command line arguments
#
parser = argparse.ArgumentParser ()

parser.add_argument ('-x', '--width',               type=int, default=640,  help='Image width')
parser.add_argument ('-y', '--height',              type=int, default=480,  help='Image height')
parser.add_argument ('-s', '--steps',               type=int, default=1000, help='Steps per epoch')
parser.add_argument ('-e', '--epochs',              type=int, default=10,   help='Number of epochs')
parser.add_argument ('-b', '--batchsize',           type=int, default=5  ,  help='Number of samples per training batch')
parser.add_argument ('-o', '--output',              type=str, default=None, help='Model output file name')
parser.add_argument ('-l', '--log',                 type=str, default=None, help='Log file directory')
parser.add_argument ('-t', '--tensorboard',         action='store_true', default=False, help='Open log in tensorboard')
parser.add_argument ('-v', '--verbose',             action='store_true', default=False, help='Verbose output')
parser.add_argument ('-i', '--intermediate-saving', action='store_true', default=False, help='Save intermediate model after each epoch')

args = parser.parse_args ()

assert not args.tensorboard or args.log
assert not args.intermediate_saving or args.output != None

#
# Delete old log files
#
if args.log:
    for root, dirs, files in os.walk (args.log, topdown=False):
        for name in files:
            if name.startswith ('events'):
                os.remove (os.path.join (root, name))

print ('Training model...')
print ('  Image size: {0}x{1}'.format (args.width, args.height))
print ('  Steps     : {0}'.format (args.steps))
print ('  Epochs    : {0}'.format (args.epochs))
print ('  Batchsize : {0}'.format (args.batchsize))

#
# Setup callbacks
#
callbacks = []

if args.log != None:
    callbacks.append (TensorBoard (os.path.abspath (args.log), histogram_freq=1, write_graph=True, write_images=False))

if args.intermediate_saving:
    file, ext = os.path.splitext (args.output)
    callbacks.append (ModelCheckpoint (file + '.{epoch:02d}' + ext, 
                                       monitor='dice_coef', 
                                       verbose=0, 
                                       save_best_only=True, 
                                       save_weights_only=False, 
                                       mode='max'))

callbacks.append (EarlyStopping (monitor='val_loss', min_delta=0, patience=1, verbose=args.verbose, mode='min'))

#
# Generate model and start fitting
#    
model = create_model (args.width, args.height)

model.fit_generator (generator=sheet_metal_generator (args.width, args.height, args.batchsize),
                     steps_per_epoch=args.steps, 
                     epochs=args.epochs, 
                     validation_data=sheet_metal_generator (args.width, args.height, args.batchsize),
                     validation_steps=int (args.steps / 10),
                     verbose=1 if args.verbose else 0, 
                     callbacks=callbacks)

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

#
# Tensorflow termination bug workaround
#
gc.collect ()

