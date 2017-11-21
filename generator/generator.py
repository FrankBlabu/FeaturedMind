#!/usr/bin/python3
#
# generator.py - Abstract base class for all generator classes
#
# Frank Blankenburg, Oct. 2017
#

import numpy as np
import queue
import threading

import skimage.filters

from abc import ABC, abstractmethod

#----------------------------------------------------------------------------------------------------------------------
# CLASS GeneratorThread
#
class GeneratorThread (threading.Thread):

    def __init__ (self, generator, number, result):
        super ().__init__ ()
        self.generator = generator
        self.number = number
        self.result = result

    def run (self):

        for _ in range (self.number):
            (image, mask) = self.generator.generate ()
            self.result.put ((image, mask))



#----------------------------------------------------------------------------------------------------------------------
# CLASS Generator
#
# Abstract base class for all generator classes
#
class Generator (ABC):

    #
    # Create generator
    #
    # @param args Arguments used to create the generator
    #
    def __init__ (self, args):
        self.width = args.width
        self.height = args.height
        self.depth = 3

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

    def __init__ (self, args, generators):
        super ().__init__ (args)
        self.generators = generators
        self.use_threading = True

    #
    # Return if this generator creates an active layer which must be detected as a separate image segmentation class
    #
    def is_active_layer (self):
        raise RuntimeError ('Stacked generator class cannot be used as a layer generator')

    def set_use_threading (self, state):
        self.use_threading = state

    def generate (self):

        image = np.zeros ((self.height, self.width, self.depth), dtype=np.float32)
        mask  = np.zeros ((self.height, self.width), dtype=np.float32)
        step = 0

        results = []
        threads = []

        if self.use_threading:
            for generator in self.generators:
                results.append (queue.Queue ())
                threads.append (GeneratorThread (generator, 1, results[-1]))
                threads[-1].start ()

            for result in results:
                step_image, step_mask = result.get ()

                assert step_image.shape[0] == self.height
                assert step_image.shape[1] == self.width
                assert step_image.shape[2] == self.depth

                assert step_mask is None or step_mask.shape[0] == self.height
                assert step_mask is None or step_mask.shape[1] == self.width
                assert step_mask is None or len (step_mask.shape) == 2

                if step_mask is None:
                    image = step_image
                else:
                    copy_mask = skimage.filters.gaussian (step_mask, sigma=0.5, mode='nearest')
                    copy_mask = np.dstack ((copy_mask, copy_mask, copy_mask))

                    image = (1 - copy_mask) * image + copy_mask * step_image
                    mask[step_mask > 0] = step

                step += 1

            for thread in threads:
                thread.join ()

        else:
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
                    copy_mask = skimage.filters.gaussian (step_mask, sigma=0.5, mode='nearest')
                    copy_mask = np.dstack ((copy_mask, copy_mask, copy_mask))

                    image = (1 - copy_mask) * image + copy_mask * step_image
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


#----------------------------------------------------------------------------------------------------------------------
# Generator for training batches
#
def batch_generator (generator, batch_size):

    classes = generator.get_number_of_classes ()

    batch_x = np.zeros ((batch_size, generator.height, generator.width, 3))
    batch_y = np.zeros ((batch_size, generator.height, generator.width, classes))

    while True:

        for i in range (batch_size):
            image, mask = generator.generate ()

            assert len (image.shape) == 3
            assert image.shape[0] == generator.height
            assert image.shape[1] == generator.width
            assert image.shape[2] == generator.depth

            assert len (mask.shape) == 2
            assert mask.shape[0] == image.shape[0]
            assert mask.shape[1] == image.shape[1]

            batch_x[i] = image

            for layer in range (classes):
                batch_y[i,:,:,layer][mask == layer + 1] = 1

        yield batch_x, batch_y
