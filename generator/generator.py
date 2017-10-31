#!/usr/bin/python3
#
# generator.py - Abstract base class for all generator classes
#
# Frank Blankenburg, Oct. 2017
#

import numpy as np

from abc import ABC, abstractmethod


#----------------------------------------------------------------------------------------------------------------------
# CLASS Generator
#
# Abstract base class for all generator classes
#
class Generator (ABC):

    def __init__ (self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    #
    # Return if this generator creates an active layer which must be detected as a separate image segmentation class
    #
    @abstractmethod
    def is_active_layer (self):
        pass

    #
    # Generate a new image instance
    #
    # @return Tuple consisting of the image itself and a mask marking all relevant image points. The second
    #         return parameter can be 'None'  if there is no relevant mask and all pixels are valid, which is
    #         especially the case for background images.
    #
    @abstractmethod
    def generate (self):
        pass


#----------------------------------------------------------------------------------------------------------------------
# CLASS StackedGenerator
#
# Generator class using multiple other generators to generate a combined image and mask output
#
class StackedGenerator (Generator):

    def __init__ (self, width, height, depth, generators):
        super ().__init__ (width, height, depth)
        self.generators = generators

    #
    # Return if this generator creates an active layer which must be detected as a separate image segmentation class
    #
    def is_active_layer (self):
        raise RuntimeError ('Stacked generator class cannot be used as a layer generator')

    def generate (self):

        image = np.zeros ((self.height, self.width, self.depth), dtype=np.float32)
        mask  = np.zeros ((self.height, self.width), dtype=np.int32)
        step = 0

        for generator in self.generators:

            step_image, step_mask = generator.generate ()

            assert step_image.shape[0] == self.height
            assert step_image.shape[1] == self.width
            assert step_image.shape[2] == self.depth

            assert step_mask is None or step_mask.shape[0] == self.height
            assert step_mask is None or step_mask.shape[1] == self.width
            assert step_mask is None or len (step_mask.shape) == 2

            if step_mask is None:
                image = step_image
            else:
                image[step_mask > 0] = step_image[step_mask > 0]
                mask[step_mask > 0] = step

            step += 1

        return image, mask

    #
    # Return the number of classes generated
    #
    def get_number_of_classes (self):

        n = 0
        for generator in self.generators:
            if generator.is_active_layer ():
                n += 1

        return n
