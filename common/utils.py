#!/usr/bin/python3
#
# utils.py - General utilities
#
# Frank Blankenburg, Apr. 2017
#

import imghdr
import math
import numpy as np
import os
import skimage.transform
import time

from matplotlib import pyplot as plt
from skimage.color import gray2rgb

#----------------------------------------------------------------------------------------------------------------------
# DECORATOR @timeit
#
# Decorator used to measure execution time of a method
#
def timeit (func):

    def timed (*args, **kw):
        start_time = time.time ()
        result = func (*args, **kw)
        end_time = time.time ()

        print ('Execution time of \'{funcname}\': {time} ms'.format (funcname=func.__name__, time=int ((end_time - start_time) * 1000)))

        return result

    return timed

#----------------------------------------------------------------------------
# Convert scikit image into TensorFlow compatible numpy array
#
# @param image Image (PIL) to convert
#
def image_to_tf (image):
    return image.reshape ((image.shape[0], image.shape[1], 3))

#
# Check if the given file is a valid image file
#
def is_image (file):

    if not os.path.isfile (file):
        return False

    file_type = imghdr.what (file)

    if file_type != 'png' and file_type != 'jpeg' and file_type != 'bmp' and file_type != 'tiff':
        return False

    return True


#----------------------------------------------------------------------------
# Add overlay with alpha channel to image
#
# @param image   Background image
# @param overlay Overlay in RGBA format
#
def add_overlay_to_image (image, overlay):

    assert len (image.shape) == 3
    assert len (overlay.shape) == 3
    assert image.shape[2] == 3
    assert overlay.shape[2] == 4
    assert image.shape[0] == overlay.shape[0]
    assert image.shape[1] == overlay.shape[1]

    overlay_alpha = np.zeros ((overlay.shape[0], overlay.shape[1], 3))
    overlay_alpha[:,:,0] = overlay[:,:,3]
    overlay_alpha[:,:,1] = overlay[:,:,3]
    overlay_alpha[:,:,2] = overlay[:,:,3]

    overlay_image = overlay[:,:,0:3]

    return (np.ones (image.shape) - overlay_alpha) * image + overlay_alpha * overlay_image


#----------------------------------------------------------------------------
# Show images in dialog
#
# @param images Images to show
# @param titles Image titles
#
def show_image (*args, show_legend=False):

    fig = plt.figure ()

    if len (args) == 1:
        partitions = [ (1, 1, 1) ]
    elif len (args) == 2:
        partitions = [ (1, 2, 1), (1, 2, 2) ]
    elif len (args) == 3:
        partitions = [ (1,3,1), (1,3,2), (1,3,3) ]
    elif len (args) == 4:
        partitions = [ (2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4) ]
    else:
        raise '{0} image partitions are not supported'.format (len (args))

    assert len (args) == len (partitions)

    for arg, partition in zip (args, partitions):
        part = fig.add_subplot (partition[0], partition[1], partition[2])
        part.set_title (arg[1])

        image = arg[0]

        colormap = None

        if len (image.shape) < 3:
            image = image.reshape ((image.shape[0], image.shape[1], 1))
            colormap = 'CMRmap'

        plt.imshow (arg[0], cmap=colormap)

        if colormap is not None and show_legend:
            plt.colorbar ()

    fig.tight_layout ()

    def onresize (event):
        plt.tight_layout ()

    fig.canvas.mpl_connect ('resize_event', onresize)

    plt.show ()


def mean_center (image):
    '''
    Mean center image data
    '''

    assert len (image.shape) == 3

    for channel in range (image.shape[-1]):
        std = image[:, channel].std ()
        if not math.isclose (std, 0):
            image[:, channel] = (image[:, channel] - image[:,channel].mean ()) / std

    return image

def mean_uncenter (image):
    '''
    Transpose (probably mean centered) image data into the interval [0, 1]
    '''
    if math.isclose (image.max (), image.min ()):
        return np.clip (image - image.min (), 0, 1)

    return np.clip ((image - image.min ()) / (image.max () - image.min ()), 0, 1)

def to_rgb (image):
    '''
    Convert numpy array representing an image into a RGB image of the right shape
    '''

    image = mean_uncenter (image)

    return gray2rgb (image)


#--------------------------------------------------------------------------
# Cut area out of image
#
def cutout (image, area):
    r = area.as_tuple ()
    result = image[r[1]:r[3] + 1,r[0]:r[2] + 1, image.shape[2]]
    return result.reshape ((result.shape[0], result.shape[1], result.shape[2]))


#--------------------------------------------------------------------------
# Transform image
#
# The image must have a float compatible data type (np.float64, usually !). Otherwise
# the value range is destroyed. The skimage parameter 'preserve_range' would prevent that,
# but generate visible object borders.
#
def transform (image, size, scale, shear, rotation):

    scale_trans = skimage.transform.AffineTransform (scale=scale)
    image = skimage.transform.warp (image, scale_trans.inverse, output_shape=image.shape)

    center_trans = skimage.transform.AffineTransform (translation=((1 - scale[0]) * size.width / 2, (1 - scale[1]) * size.height / 2))
    image = skimage.transform.warp (image, center_trans.inverse, output_shape=image.shape)

    image = skimage.transform.rotate (image, angle=rotation * 180.0 / math.pi, resize=False, center=None)

    shear_trans = skimage.transform.AffineTransform (shear=shear)
    image = skimage.transform.warp (image, shear_trans.inverse, output_shape=image.shape)

    return image
