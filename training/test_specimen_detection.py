#!/usr/bin/python3
#
# test_specimen_detection.py - Test for the specimen detection
#
# Frank Blankenburg, Oct. 2017
#

import unittest
import numpy as np
import generator.generator as generator

import training.specimen_detection

#----------------------------------------------------------------------------------------------------------------------
# CLASS TestSpecimenDetection
#
# Unittest for the specimen detection functions
class TestGenerator (generator.Generator):

    def __init__ (self, width, height, depth):
        super ().__init__ (width, height, depth)
        self.classes = 2

        assert width >= 2
        assert height >= 2

    def is_active_layer (self):
        return False

    def generate (self):
        image = np.random.rand (self.height, self.width, self.depth)
        mask = np.zeros ((self.height, self.width), dtype=np.float64)
        mask[0, 0] = 1
        mask[1, 1] = 2

        return image, mask

    def get_number_of_classes (self):
        return self.classes


#----------------------------------------------------------------------------------------------------------------------
# CLASS TestSpecimenDetection
#
# Unittest for the specimen detection functions
#
class TestSpecimenDetection (unittest.TestCase):

    def test_batch_generator (self):

        #
        # Single element batch
        #
        generator = TestGenerator (width=4, height=2, depth=3)
        batches = training.specimen_detection.batch_generator (generator, 1)

        for _ in range (10):
            images, masks = next (batches)

            self.assertEqual (images.shape, (1, 2, 4, 3))
            self.assertEqual (masks.shape, (1, 2, 4, generator.get_number_of_classes ()))

            self.assertEqual ((masks[:,0,0,0] == 1).all (), True)
            self.assertEqual ((masks[:,0,0,1] == 1).all (), False)
            self.assertEqual ((masks[:,1,1,0] == 1).all (), False)
            self.assertEqual ((masks[:,1,1,1] == 1).all (), True)

        #
        # Multi element batch
        #
        generator = TestGenerator (width=5, height=3, depth=3)
        batches = training.specimen_detection.batch_generator (generator, 8)

        for _ in range (1):
            images, masks = next (batches)

            self.assertEqual (images.shape, (8, 3, 5, 3))
            self.assertEqual (masks.shape, (8, 3, 5, generator.get_number_of_classes ()))

            self.assertEqual ((masks[:,0,0,0] == 1).all (), True)
            self.assertEqual ((masks[:,0,0,1] == 1).all (), False)
            self.assertEqual ((masks[:,1,1,0] == 1).all (), False)
            self.assertEqual ((masks[:,1,1,1] == 1).all (), True)
