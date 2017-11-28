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
import sys
import webbrowser

import tensorflow as tf

import common.constants
import common.losses
import common.metrics

import generator.generator
import generator.background
import generator.fixture
import generator.generator
import generator.sheetmetal


#----------------------------------------------------------------------------------------------------------------------
# Generate model
#
# This function generates a tensorflow CNN model for pixel wise segmentation of an image of a given width / height
#
# @param generator Generator object creating the training/validation/test batches
#
def create_model (generator, learning_rate, number_of_classes):

    inputs = tf.keras.layers.Input (shape=(generator.height, generator.width, generator.depth), name='input')

    #
    # Create VGG-16 model with pretrained weights (see https://keras.io/applications/#vgg16)
    #
    # Example see: https://github.com/divamgupta/image-segmentation-keras/blob/master/Models/VGGSegnet.py
    #
    x = tf.keras.applications.vgg16.VGG16 (include_top=False,
                                           weights='imagenet',
                                           input_tensor=inputs,
                                           input_shape=(generator.height, generator.width, generator.depth),
                                           pooling='None')

    #
    # The VGG 16 part of the model has pretrained, fixed weights and must not be trained again here.
    #
    for layer in x.layers:
        layer.trainable = False

    #
    # Upsampling to 'convert' object detection values into image segmentation results
    #
    x = tf.keras.layers.ZeroPadding2D ((1, 1)) (x.output)
    x = tf.keras.layers.Conv2D (512, (3, 3), padding='valid') (x)
    x = tf.keras.layers.BatchNormalization () (x)

    x = tf.keras.layers.UpSampling2D ((2,2)) (x)
    x = tf.keras.layers.ZeroPadding2D ((1,1)) (x)
    x = tf.keras.layers.Conv2D (256, (3, 3), padding='valid') (x)
    x = tf.keras.layers.BatchNormalization () (x)

    x = tf.keras.layers.UpSampling2D ((2,2)) (x)
    x = tf.keras.layers.ZeroPadding2D ((1,1)) (x)
    x = tf.keras.layers.Conv2D (128 , (3, 3), padding='valid') (x)
    x = tf.keras.layers.BatchNormalization () (x)

    x = tf.keras.layers.UpSampling2D ((2,2)) (x)
    x = tf.keras.layers.ZeroPadding2D ((1,1)) (x)
    x = tf.keras.layers.Conv2D (64, (3, 3), padding='valid',) (x)
    x = tf.keras.layers.BatchNormalization () (x)

    x = tf.keras.layers.Conv2D (number_of_classes, (3, 3) , padding='same') (x)
    x = tf.keras.layers.Activation ('softmax') (x)

    model = tf.keras.models.Model (inputs=[inputs], outputs=[x])

    model.summary ()

    model.compile (optimizer=tf.keras.optimizers.Adam (lr=learning_rate),
                   loss=common.losses.dice_coef,
                   metrics=['accuracy',
                            common.metrics.precision,
                            common.metrics.recall,
                            common.metrics.f1_score,
                            common.metrics.dice_coef])

    return model



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
        callbacks.append (tf.keras.callbacks.TensorBoard (os.path.abspath (args.log),
                                                          batch_size=args.batchsize,
                                                          write_grads=True,
                                                          write_graph=True,
                                                          write_images=True))

    if args.intermediate_saving:
        callbacks.append (tf.keras.callbacks.ModelCheckpoint (args.output,
                                                              monitor='dice_coef',
                                                              verbose=0,
                                                              save_best_only=True,
                                                              save_weights_only=False,
                                                              mode='max'))

    #callbacks.append (tf.keras.callbacks.EarlyStopping (monitor='val_loss', min_delta=0, patience=1, verbose=True, mode='min'))

    #
    # Setup generator
    #
    generators = [ generator.background.BackgroundGenerator.create (args),
                   generator.sheetmetal.SheetMetalGenerator (args),
                   generator.fixture.FixtureGenerator (args) ]

    data = generator.generator.StackedGenerator (args, generators)
    data.set_use_threading (False)

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
        print ('  Continue training of model \'{model}\''.format (model=args.continue_training))

    if args.log:
        print ('  Log directory    : {log}'.format (log=args.log))

    #
    # Generate model and start fitting
    #
    if args.continue_training:
        model = tf.keras.models.load_model (args.continue_training, custom_objects={'dice_coef': common.losses.dice_coef,
                                                                                    'precision': common.metrics.precision,
                                                                                    'recall'   : common.metrics.recall,
                                                                                    'f1_score' : common.metrics.f1_score})
    else:
        model = create_model (generator=data, learning_rate=1e-5, number_of_classes=data.get_number_of_classes ())

    print (model.input_shape, model.output_shape)

    model.fit_generator (generator=generator.generator.batch_generator (data, args.batchsize),
                         steps_per_epoch=args.steps,
                         epochs=args.epochs,
                         validation_data=generator.generator.batch_generator (data, args.batchsize),
                         validation_steps=int (args.steps / 10) if args.steps >= 10 else 1,
                         verbose=True,
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
