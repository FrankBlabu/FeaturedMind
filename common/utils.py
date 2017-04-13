#!/usr/bin/python3
#
# utils.py - General utilities
#
# Frank Blankenburg, Apr. 2017
#

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
def show_image (*arg):
    
    fig = plt.figure ()

    for i in range (len (arg)):                    
        part = fig.add_subplot (1, len (arg), i+1)
        part.set_title (arg[i][1])            
        plt.imshow (arg[i][0])
    
    plt.show ()
