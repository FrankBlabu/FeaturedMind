#!/usr/bin/python3
#
# utils.py - General utilities
#
# Frank Blankenburg, Apr. 2017
#

import math
import numpy as np
import skimage.transform

from matplotlib import pyplot as plt
from skimage.color import gray2rgb

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
        partitions = [ (1, 2, 1), (1, 2, 2) ]
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
        
        image = arg[0]
        
        colormap = None
        if len (image.shape) < 3 or image.shape[2] != 3:
            image = image.reshape ((image.shape[0], image.shape[1]))
            colormap = 'CMRmap'
        
        plt.imshow (arg[0], cmap=colormap)
        
        if not colormap is None:
            plt.colorbar ()
    
    fig.tight_layout ()
    
    def onresize (event):
        plt.tight_layout ()
        
    fig.canvas.mpl_connect ('resize_event', onresize)
    
    plt.show ()


def mean_center (image):
    '''
    Mean center image data in the interval [-1, 1]
    '''
    if math.isclose (image.max (), image.min ()):
        return np.clip (image - image.min (), -1, 1)

    return np.clip (2 * (image - image.mean ()) / (image.max () - image.min ()), -1, 1)

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
    
    if len (image.shape) > 2:
        image = image.reshape ((image.shape[0], image.shape[1]))
        
    return gray2rgb (image)


#--------------------------------------------------------------------------
# Cut area out of image
#
def cutout (image, area):
    r = area.as_tuple ()
    result = image[r[1]:r[3]+1,r[0]:r[2]+1]    
    return result.reshape ((result.shape[0], result.shape[1], 1))


#--------------------------------------------------------------------------
# Transform image
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

