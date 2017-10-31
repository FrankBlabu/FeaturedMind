#!/usr/bin/python3
#
# specimen_detection.py - Train net to recognize sheet metals in different environments
#
# Frank Blankenburg, Mar. 2017
#
# See https://github.com/aurora95/Keras-FCN/blob/master/models.py
#     https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pd
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
from keras.models import load_model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

import common.losses
import common.metrics

import generator.background
import generator.fixture
import generator.generator
import generator.sheetmetal


#----------------------------------------------------------------------------------------------------------------------
# Generate model
#
# This function generates a CNN keras model for pixel wise segmentation of an image of a given width / height
#
# @param generator Generator object creating the training/validation/test batches
#
def create_model (shape):

    inputs = Input (shape=shape, name='input')

    #
    # Downsampling
    #
    conv1 = Conv2D (32, (3, 3), activation='relu', padding='same') (inputs)
    conv1 = Conv2D (32, (3, 3), activation='relu', padding='same') (conv1)
    pool1 = MaxPooling2D (pool_size=(2, 2)) (conv1)

    conv2 = Conv2D (64, (3, 3), activation='relu', padding='same') (pool1)
    conv2 = Conv2D (64, (3, 3), activation='relu', padding='same') (conv2)
    pool2 = MaxPooling2D (pool_size=(2, 2)) (conv2)

    conv3 = Conv2D (128, (3, 3), activation='relu', padding='same') (pool2)
    conv3 = Conv2D (128, (3, 3), activation='relu', padding='same') (conv3)
    pool3 = MaxPooling2D (pool_size=(2, 2)) (conv3)

    conv4 = Conv2D (256, (3, 3), activation='relu', padding='same') (pool3)
    conv4 = Conv2D (256, (3, 3), activation='relu', padding='same') (conv4)
    pool4 = MaxPooling2D (pool_size=(2, 2)) (conv4)

    conv5 = Conv2D (512, (3, 3), activation='relu', padding='same') (pool4)
    conv5 = Conv2D (512, (3, 3), activation='relu', padding='same') (conv5)

    #
    # Upsampling
    #
    up6 = concatenate ([UpSampling2D (size=(2, 2)) (conv5), conv4], axis=3)
    conv6 = Conv2D (256, (3, 3), activation='relu', padding='same') (up6)
    conv6 = Conv2D (256, (3, 3), activation='relu', padding='same') (conv6)

    up7 = concatenate ([UpSampling2D (size=(2, 2)) (conv6), conv3], axis=3)
    conv7 = Conv2D (128, (3, 3), activation='relu', padding='same') (up7)
    conv7 = Conv2D (128, (3, 3), activation='relu', padding='same') (conv7)

    up8 = concatenate ([UpSampling2D (size=(2, 2)) (conv7), conv2], axis=3)
    conv8 = Conv2D (64, (3, 3), activation='relu', padding='same') (up8)
    conv8 = Conv2D (64, (3, 3), activation='relu', padding='same') (conv8)

    up9 = concatenate ([UpSampling2D (size=(2, 2)) (conv8), conv1], axis=3)
    conv9 = Conv2D (32, (3, 3), activation='relu', padding='same') (up9)
    conv9 = Conv2D (32, (3, 3), activation='relu', padding='same') (conv9)

    conv10 = Conv2D (generator.get_number_of_classes (), (1, 1), activation='sigmoid') (conv9)

    model = Model (inputs=[inputs], outputs=[conv10])
    model.compile (optimizer=optimizers.Adam (lr=1e-5),
                   loss=common.losses.dice_coef,
                   metrics=['accuracy',
                            common.metrics.precision,
                            common.metrics.recall,
                            common.metrics.f1_score,
                            common.metrics.dice_coef])

    return model


#----------------------------------------------------------------------------------------------------------------------
# Generator for training batches
#
def batch_generator (generator, batch_size):

    batch_x = np.zeros ((batch_size, generator.height, generator.width, 3))
    batch_y = np.zeros ((batch_size, generator.height, generator.width, 1))

    while True:

        for i in range (batch_size):
            image, mask = generator.generate ()
            mask = mask.reshape ((mask.shape[0], mask.shape[1], 1))

            batch_x[i] = image
            batch_y[i] = mask

        yield batch_x, batch_y


#----------------------------------------------------------------------------------------------------------------------
# Train specimen detection CNN
#
def train ():

    #
    # Parse command line arguments
    #
    parser = argparse.ArgumentParser ()

    parser.add_argument ('-x', '--width',                type=int, default=640,  help='Image width')
    parser.add_argument ('-y', '--height',               type=int, default=480,  help='Image height')
    parser.add_argument ('-s', '--steps',                type=int, default=1000, help='Steps per epoch')
    parser.add_argument ('-e', '--epochs',               type=int, default=10,   help='Number of epochs')
    parser.add_argument ('-b', '--batchsize',            type=int, default=5,    help='Number of samples per training batch')
    parser.add_argument ('-o', '--output',               type=str, default=None, help='Model output file name')
    parser.add_argument ('-l', '--log',                  type=str, default=None, help='Log file directory')
    parser.add_argument ('-t', '--tensorboard',          action='store_true', default=False, help='Open log in tensorboard')
    parser.add_argument ('-v', '--verbose',              action='store_true', default=False, help='Verbose output')
    parser.add_argument ('-i', '--intermediate-saving',  action='store_true', default=False, help='Save intermediate model after each epoch')
    parser.add_argument ('-c', '--continue-training',    type=str, default=None, help='Continue training of existing model')

    generator.background.BackgroundGenerator.add_to_args_definition (parser)

    args = parser.parse_args ()

    if args.tensorboard and not args.log:
        raise RuntimeError ('Tensorboard activated but no \'-l\' option given to specify log directory')
    if args.intermediate_saving and not args.output:
        raise RuntimeError ('Intermediate saving activated but no \'-o\' option given to specify the output file name')

    #
    # Delete old log files
    #
    if args.log:
        for root, dirs, files in os.walk (args.log, topdown=False):
            for name in files:
                if name.startswith ('events'):
                    os.remove (os.path.join (root, name))

    #
    # Setup training callbacks
    #
    callbacks = []

    if args.log is not None:
        callbacks.append (TensorBoard (os.path.abspath (args.log), histogram_freq=1, write_graph=True, write_images=True))

    if args.intermediate_saving:
        callbacks.append (ModelCheckpoint (args.output,
                                           monitor='dice_coef',
                                           verbose=0,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode='max'))

    callbacks.append (EarlyStopping (monitor='val_loss', min_delta=0, patience=1, verbose=args.verbose, mode='min'))

    #
    # Setup generator
    #
    generators = [ generator.background.BackgroundGenerator.create (args),
                   generator.sheetmetal.SheetMetalGenerator (args.width, args.height),
                   generator.fixture.FixtureGenerator (args.width, args.height) ]

    data = generator.generator.StackedGenerator (args.width, args.height, 3, generators)


    #
    # Print some startup information
    #
    print ('Training model...')
    print ('  Image size       : {width}x{height}'.format (width=args.width, height=args.height))
    print ('  Steps            : {steps}'.format (steps=args.steps))
    print ('  Epochs           : {epochs}'.format (epochs=args.epochs))
    print ('  Batchsize        : {batches}'.format (batches=args.batchsize))
    print ('  Number of classes: {classes}'.format (classes=data.get_number_of_classes ()))

    if args.background_directory:
        print ('  Background mode  : {mode} ({directory})'.format (mode=args.background_mode, directory=os.path.abspath (args.background_directory)))
    else:
        print ('  Background mode  : {mode}'.format (mode=args.background_mode))

    if args.continue_training:
        print ('    Continue training of model \'{model}\''.format (model=args.continue_training))

    #
    # Generate model and start fitting
    #
    if args.continue_training:
        model = load_model (args.continue_training, custom_objects={'dice_coef': common.losses.dice_coef,
                                                                    'precision': common.metrics.precision,
                                                                    'recall'   : common.metrics.recall,
                                                                    'f1_score' : common.metrics.f1_score})
    else:
        model = create_model (shape=(args.batchsize, data.height, data.width, data.depth))

    model.fit_generator (generator=batch_generator (data, args.batchsize),
                         steps_per_epoch=args.steps,
                         epochs=args.epochs,
                         validation_data=batch_generator (data, args.batchsize),
                         validation_steps=int (args.steps / 10),
                         verbose=1 if args.verbose else 0,
                         callbacks=callbacks)

    if args.output is not None:
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

if __name__ == '__main__':
    train ()
