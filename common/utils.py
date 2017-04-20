#!/usr/bin/python3
#
# utils.py - General utilities
#
# Frank Blankenburg, Apr. 2017
#

import math
import numpy as np

from matplotlib import pyplot as plt

#----------------------------------------------------------------------------
# Convert scikit image into TensorFlow compatible numpy array
#
# @param image Image (PIL) to convert
#
def image_to_tf (image):
    return image.reshape ((image.shape[0], image.shape[1], 1))


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
def show_image (*args):
    
    fig = plt.figure ()

    if len (args) == 1:
        partitions = [ (1, 1, 1) ]
    elif len (args) == 2:
        partitions = [ (2, 1, 1), (2, 1, 2) ]
    elif len (args) == 3:
        partitions = [ (2,3,(1,3)), (2,3,4), (2,3,5), (2,3,6) ]
    elif len (args) == 4:
        partitions = [ (2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4) ]
    else:
        raise '{0} image partitions are not supported'.format (len (args))

    assert len (args) == len (partitions)

    for arg, partition in zip (args, partitions):  
        part = fig.add_subplot (partition[0], partition[1], partition[2])
        part.set_title (arg[1])
        plt.imshow (arg[0])
    
    fig.tight_layout ()
    
    def onresize (event):
        plt.tight_layout ()
        
    fig.canvas.mpl_connect ('resize_event', onresize)
    
    plt.show ()

#----------------------------------------------------------------------------
# Mean center image data
#
def mean_center (image):
    if math.isclose (image.max (), image.min ()):
        return image

    return 2 * (image - image.mean ()) / (image.max () - image.min ())
    
    
