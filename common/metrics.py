#!/usr/bin/python3
#
# metrics.py - Custom metric functions
#
# Frank Blankenburg, Mar. 2017
#

from keras import backend as K


#--------------------------------------------------------------------------
# Compute precision metrics
#
# Precision := True positives / All positives guesses
#
# Meaning: When we found a feature, how many times was is really a feature ?
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

def dice_coef (y_true, y_pred):
    smooth = 1.0
    
    y_true_f = K.flatten (y_true)
    y_pred_f = K.flatten (y_pred)
    intersection = K.sum (y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum (y_true_f) + K.sum (y_pred_f) + smooth)

