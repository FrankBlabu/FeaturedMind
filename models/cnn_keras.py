#!/usr/bin/python3
#
# models.cnn_keras.py - CNN model setup using keras
#
# Frank Blankenburg, Mar. 2017
#

import os
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input
from keras import backend as K

from models.training_data import TrainingData


#--------------------------------------------------------------------------
# Compute precision metrics
#
# Precision := True positives / All positives guesses
#
# Meaning: When we found a feature, how many times was is really aa feature ?
#
def precision (y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum (K.round (K.clip (y_true * y_pred, 0, 1)))
    predicted_positives = K.sum (K.round (K.clip (y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon ())


#--------------------------------------------------------------------------
# Compute recall metrics
#
# Recall := True positives / All positives in dataset
#
# Meaning: How many of the actual present features did we find ?
#
def recall (y_true, y_pred):
    true_positives = K.sum (K.round (K.clip (y_true * y_pred, 0, 1)))
    all_positives = K.sum (K.round (K.clip (y_true, 0, 1)))
    return true_positives / (all_positives + K.epsilon ())


#--------------------------------------------------------------------------
# Train border detection with a manually constructed multilayer 
# convolutional network
#
def train (args, data):
    
    num_classes = data.segments + 2
    epochs = 10
    
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

    print ('x_train shape:', x_train.shape)
    print (x_train.shape[0], 'train samples')
    print (x_test.shape[0], 'test samples')
    
    y_train = keras.utils.to_categorical (y_train, num_classes)
    y_test = keras.utils.to_categorical (y_test, num_classes)
                
    model = Sequential ()
    model.add (Conv2D (32, kernel_size=(5, 5),
                       activation='relu',
                       input_shape=input_shape))
    model.add (MaxPooling2D (pool_size=(2, 2)))
    model.add (Dropout (0.25))
    model.add (Flatten ())
    model.add (Dense (1024, activation='relu'))
    model.add (Dropout (0.5))
    model.add (Dense (num_classes, activation='softmax'))
            
    model.compile (loss=keras.losses.categorical_crossentropy,
                   optimizer=keras.optimizers.Adadelta (),
                   metrics=['accuracy', precision, recall])
    
    model.fit (x_train, y_train, batch_size=args.batchsize, epochs=epochs,
              verbose=1, validation_data=(x_test, y_test))
    
    score = model.evaluate (x_test, y_test, verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
