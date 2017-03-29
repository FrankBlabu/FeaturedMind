#!/usr/bin/python3
#
# models.cnn_keras.py - CNN model setup using keras
#
# Frank Blankenburg, Mar. 2017
#

import os
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras import backend as K

import models.metrics


#--------------------------------------------------------------------------
# Train border detection with a manually constructed multilayer 
# convolutional network
#
def train (args, data):
        
    x_train = data.get_training_data ()[0]
    y_train = data.get_training_data ()[1]

    x_test = data.get_test_data ()[0]
    y_test = data.get_test_data ()[1]
        
    if K.image_data_format () == 'channels_first':
        x_train = x_train.reshape (x_train.shape[0], 1, data.sample_size, data.sample_size)
        x_test = x_test.reshape (x_test.shape[0], 1, data.sample_size, data.sample_size)
        input_shape = (1, data.sample_size, data.sample_size)
    else:
        x_train = x_train.reshape (x_train.shape[0], data.sample_size, data.sample_size, 1)
        x_test = x_test.reshape (x_test.shape[0], data.sample_size, data.sample_size, 1)
        input_shape = (data.sample_size, data.sample_size, 1)
    
    x_train = x_train.astype ('float32')
    x_test = x_test.astype ('float32')

    y_train = keras.utils.to_categorical (y_train, data.classes)
    y_test = keras.utils.to_categorical (y_test, data.classes)
    
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
    model.add (Dense (data.classes, activation='softmax'))
            
    model.compile (loss=keras.losses.categorical_crossentropy,
                   optimizer=keras.optimizers.Adadelta (),
                   metrics=['accuracy', models.metrics.precision, models.metrics.recall])
    
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

    
