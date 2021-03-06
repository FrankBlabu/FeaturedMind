#!/usr/bin/python3
#
# generator.py - Abstract base class for all generator classes
#
# Frank Blankenburg, Oct. 2017
#

import numpy as np
import queue
import threading

import common.utils

from abc import ABC, abstractmethod


#----------------------------------------------------------------------------------------------------------------------
# CLASS GeneratorThread
#
class GeneratorThread (threading.Thread):
    """
    Thread for threaded sample generation

    This generator thread runs a given number of sample generating instances in parallel and forwards the results
    via a queue
    """

    def __init__ (self, generator, number, result):
        """
        Constructor

        :param generator: Generator producing a single sample. The generator object must be reentrant.
        :param number:    Number of threads running in parallel.
        :param result:    Queue used to forward the results to the calling instance.
        """
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
class Generator (ABC):
    """
    Abstract base class for all generator classes

    It is assumed that all generator are producing RGB images with 3 channels.
    """

    def __init__ (self, args):
        """
        Constructor

        :param args: Command line arguments configuring the generator. Must at least contains the parameters 'width' and 'height'.
        """
        self.width = args.width
        self.height = args.height
        self.depth = 3

    @abstractmethod
    def is_active_layer (self):
        """
        Return if this generator creates an active layer which must be detected as a separate image segmentation class
        """
        pass

    @abstractmethod
    def generate (self):
        """
        Generate a new image instance

        :return: Tuple consisting of the image itself and a mask marking all relevant image points. The second
                 return parameter can be 'None'  if there is no relevant mask and all pixels are valid, which is
                 especially the case for background images.
        """
        pass


#----------------------------------------------------------------------------------------------------------------------
# CLASS StackedGenerator
#
class StackedGenerator (Generator):
    """
    Generator class using multiple other generators to generate a combined image and mask output

    Each generator returns a (image, mask) tuple. The resulting stacked image is generated by just pasting each new
    layer onto the existing combined image if the corresponding mask is set. The mask can contain values in the range
    [0:1] and represents the alpha channel. So a '1' pixel in the mask will use the incoming images pixel only, while a
    '0.5' will combine both the result images pixel and the incoming images pixel with the same factor.

    The masks of the active generators are stacked as channels.
    """

    def __init__ (self, args, generators):
        """
        Constructor

        :param args:       Command line parameters used to set up the generators
        :param generators: Sorted list of generators for creating the image overlays and associated masks
        """
        super ().__init__ (args)
        self.generators = generators
        self.use_threading = True

    def is_active_layer (self):
        """
        Return if this generator creates an active layer which must be detected as a separate image segmentation class

        This is never the case for the stacked generator
        """
        raise RuntimeError ('Stacked generator class cannot be used as a layer generator')

    def set_use_threading (self, state):
        """
        Set if threading should be used for image generation.

        :warining: This might result in collisions with the keras threading system !
        """
        self.use_threading = state

    def generate (self):

        image = np.zeros ((self.height, self.width, self.depth), dtype=np.float64)
        mask  = None

        results = []
        threads = []

        if self.use_threading:
            for generator in self.generators:
                results.append (queue.Queue ())
                threads.append (GeneratorThread (generator, 1, results[-1]))
                threads[-1].start ()

            for result in results:
                step_image, step_mask = result.get ()
                image, mask = self.add_result_step (image, mask, step_image, step_mask)

            for thread in threads:
                thread.join ()

        else:
            for generator in self.generators:
                step_image, step_mask = generator.generate ()
                image, mask = self.add_result_step (image, mask, step_image, step_mask)

        return image, mask

    def add_result_step (self, image, mask, step_image, step_mask):
        """
        Add a generated (image, mask) tuple to the result image and mask arrays

        :param image: Reference to the result image array
        :param mask:  Reference to the result mask array
        :param step_image: Image generated in a single step
        :param step_mask:  Mask generated in a single step. Can be 'None' for inactive layers.
        :return: Modified (image, mask)
        """
        assert step_image.shape == image.shape

        assert step_mask is None or step_mask.shape[0:2] == image.shape[0:2]
        assert step_mask is None or len (step_mask.shape) == 2

        if step_mask is None:
            image = step_image
            mask = None
        else:
            copy_mask = np.expand_dims (step_mask, axis=2)
            image = (1 - copy_mask) * image + copy_mask * step_image

            if mask is None:
                mask = copy_mask
            else:
                assert mask.shape[0:2] == step_mask.shape[0:2]
                mask = np.dstack ((mask, step_mask))

        return image, mask


    #
    # Return the number of classes generated
    #
    def get_number_of_classes (self):
        return sum (1 for i in self.generators if i.is_active_layer ())



def generate_sample (generator, mask_width, mask_height):
    """

    Produces a map describing a single sample with this content:

    - 'image'         : Full size and mean centered RGB image with one float in the range of [0:1] per channel
    - 'mask'          : Quantified mask with one channel per generator class. Each channel contains either
                        0.0 is no object of the associated class present or 1.0 if the is one in this cell.
    - 'original_image': Full size image without mean centering, otherwise like 'image'
    - 'original_mask' : Full size mask without quantization, otherwise like 'mask'

    :param generator:   Generator used to create a set of image and pixel wise mask
    :param mask_width:  Width of the (quantified) mask in segments
    :param mask_height: Height of the (quantified) mask in segments
    :param testing:     If 'true', additional data for testing this function is added.
    :return:            Map describing a single sample
    """
    assert mask_width > 0
    assert mask_height > 0
    assert mask_height <= generator.height
    assert mask_width <= generator.width

    full_image, full_mask = generator.generate ()

    assert len (full_image.shape) == 3
    assert full_image.shape[0] == generator.height
    assert full_image.shape[1] == generator.width
    assert full_image.shape[2] == generator.depth
    assert full_image.dtype == np.float64

    assert len (full_mask.shape) == 3
    assert full_mask.shape[0] == full_image.shape[0]
    assert full_mask.shape[1] == full_image.shape[1]
    assert full_mask.dtype == np.float64

    #
    # Square areas of the mask are combined into a more coarse version to match the
    # required network output format.
    #
    mask = np.zeros ((mask_height, mask_width, full_mask.shape[2]), dtype=full_mask.dtype)

    step_height = int (full_image.shape[0] / mask_height)
    step_width = int (full_image.shape[1] / mask_width)
    step_area = step_width * step_height

    for h in range (mask_height):
        for w in range (mask_width):

            sample = full_mask[h * step_height:h * step_height + step_height,  w * step_width:w * step_width + step_width]
            mask[h,w] = np.round (sample.sum (axis=(0, 1)) / step_area + 0.4)

    sample = {}
    sample['image']          = common.utils.mean_center (full_image)
    sample['mask']           = mask
    sample['original_image'] = full_image
    sample['original_mask']  = full_mask

    assert sample['image'] is not sample['original_image']

    return sample


def batch_generator (generator, batch_size, mask_width, mask_height):
    """

    Yielding generator which produces a batch of (image, mask) pairs ready to be fed into the tensorflow graph.

    :param sample_generator: Sample generator producing one sample per call
    :param batch_size:       Size of a single batch generated per 'next ()' call
    :param mask_width:       Width of the (quantified) mask in segments
    :param mask_height:      Height of the (quantified) mask in segments
    :return     :            A batch of tuples. The content depends on the configured generator mode.
    """
    while True:

        batch_x = None
        batch_y = None

        for i in range (batch_size):

            sample = generate_sample (generator, mask_width, mask_height)

            image =  sample['image']
            mask = sample['mask']

            if i == 0:
                batch_x = np.zeros ((batch_size, image.shape[0], image.shape[1], image.shape[2]), dtype=image.dtype)
                batch_y = np.zeros ((batch_size, mask.shape[0], mask.shape[1], mask.shape[2]), dtype=mask.dtype)

            batch_x[i] = image
            batch_y[i] = mask

        yield batch_x, batch_y
