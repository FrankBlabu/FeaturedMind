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
    def __init__ (self, width, height):
        super ().__init__ (width, height)

    def is_active_layer (self):
        return False

    def generate (self):
        image = np.random.rand (self.height, self.width, 3)
        mask = np.ones ((self.height, self.width), dtype=np.int32)

        return image, mask

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
        generator = TestGenerator (4, 2)
        batches = training.specimen_detection.batch_generator (generator, 1)

        for _ in range (10):
            b1, b2 = next (batches)

            self.assertEqual (b1.shape, (1, 2, 4, 3))
            self.assertEqual (b2.shape, (1, 2, 4, 1))

        #
        # Multi element batch
        #
        generator = TestGenerator (5, 3)
        batches = training.specimen_detection.batch_generator (generator, 8)

        for _ in range (10):
            b1, b2 = next (batches)

            self.assertEqual (b1.shape, (8, 3, 5, 3))
            self.assertEqual (b2.shape, (8, 3, 5, 1))
