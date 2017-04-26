#!/usr/bin/python3
#
# losses.py - Custom loss functions
#
# Frank Blankenburg, Apr. 2017
#

import common.metrics

smooth = 1.0

def dice_coef (y_true, y_pred):
    '''
    Sorensen-Dice coefficient computing (see https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
    
    This is a method to measure similarities between two samples
    '''
    return -common.metrics.dice_coef (y_true, y_pred)
