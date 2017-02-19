#
# generate_images.py - Generate a set of training/test images
#
# Frank Blankenburg, Feb. 2017
#

import random
import PIL
import PIL.Image
import PIL.ImageDraw

#--------------------------------------------------------------------------
# Create an elliptical feature
#
# @param image    Image to draw the ellipse into
# @param center   Center of the feature
# @param size     Size of the bounding box of the feature
# @param rotation Rotation of the feature around its center in degrees
#
def create_elliptical_feature (image, center, size, rotation):
    overlay = PIL.Image.new ('RGBA', size)
    
    draw = PIL.ImageDraw.Draw (overlay)
    draw.ellipse ((0, 0, size[0] - 1, size[1] - 1),
                  outline='#ffffff', fill=None)

    rotated = overlay.rotate (rotation, expand=True,
                              resample=PIL.Image.BILINEAR)
    
    image.paste (rotated, (int (center[0] - rotated.size[0] / 2),
                           int (center[1] - rotated.size[1] / 2)), rotated)


#--------------------------------------------------------------------------
# MAIN
#

random.seed ()

#
# Create 8 bit grayscale image
#
image = PIL.Image.new ('RGBA', (800, 600))

for i in range (10):
    create_elliptical_feature (image, (400, 300), (100, 200), i * 36)

image.show ()
