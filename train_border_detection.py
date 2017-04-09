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

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras import backend as K

import common.metrics
from common.training_data import TrainingData


#--------------------------------------------------------------------------
# Train border detection with a manually constructed multilayer 
# convolutional network
#
def train_cnn (args, data):
        
    x_train = data.get_training_data (TrainingData.Field.DATA)
    y_train = data.get_training_data (TrainingData.Field.LABELS)

    x_test = data.get_test_data (TrainingData.Field.DATA)
    y_test = data.get_test_data (TrainingData.Field.LABELS)
        
    input_shape = (data.sample_size, data.sample_size, 1)
    
    y_train = keras.utils.to_categorical (y_train, 2)
    y_test = keras.utils.to_categorical (y_test, 2)
    
    model = Sequential ()

    model.add (Conv2D (32, kernel_size=(5, 5),
                       activation='relu',
                       input_shape=input_shape))
    model.add (MaxPooling2D (pool_size=(2, 2)))

    
    model.add (Conv2D (64, kernel_size=(5, 5),
                       activation='relu',
                       input_shape=(int (data.sample_size / 2),
                                    int (data.sample_size / 2),
                                    1)))
    model.add (MaxPooling2D (pool_size=(2, 2)))
    
    model.add (Flatten ())
    model.add (Dense (1024, activation='relu'))
    model.add (Dropout (0.5))
    model.add (Dense (2, activation='softmax'))
            
    model.compile (loss=keras.losses.categorical_crossentropy,
                   optimizer=keras.optimizers.Adadelta (),
                   metrics=['accuracy', common.metrics.precision, common.metrics.recall])
    
    logger = []
    if args.log != None:
        logger.append (TensorBoard (os.path.abspath (args.log), histogram_freq=1, write_graph=True, write_images=False))
    
    model.fit (x_train, y_train, 
               batch_size=args.batchsize, 
               epochs=args.epochs, 
               verbose=1,
               validation_split=0.2,
               callbacks=logger)
    
    score = model.evaluate (x_test, y_test, verbose=0)

    print ('Test loss:', score[0])
    print ('Test accuracy:', score[1])

    if args.output != None:
        model.save (os.path.abspath (args.output))


        
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
data = TrainingData (file)
    
print ("Training model...")
print ("  Number of samples: ", data.size ())
print ("  Sample size      : ", data.sample_size)

    
train_cnn (args, data)

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
