#!/usr/bin/python3
#
# losses.py - Custom loss functions
#
# Frank Blankenburg, Apr. 2017
#

from keras import backend as K

smooth = 1.0

def dice_coef (y_true, y_pred):
    '''
    Sorensen-Dice coefficient computing (see https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
    
    This is a method to measure similarities between two samples
    '''
    y_true_f = K.flatten (y_true)
    y_pred_f = K.flatten (y_pred)
    intersection = K.sum (y_true_f * y_pred_f)
    return -(2. * intersection + smooth) / (K.sum (y_true_f) + K.sum (y_pred_f) + smooth)
