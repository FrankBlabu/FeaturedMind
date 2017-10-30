#!/usr/bin/python3
#
# fixture.py - Generator for images resembling a fixture setup
#
# Frank Blankenburg, Oct. 2017
#

import argparse
import random
import math
import numpy as np
import common.utils as utils

import skimage.util

from common.geometry import Point2d, Size2d, Polygon2d
from generator.generator import Generator


#----------------------------------------------------------------------------------------------------------------------
# CLASS FixtureGenerator
#
# This class will generate an image containing a simulated fixture like object together with a mask marking
# its pixel positions.
#
class FixtureGenerator (Generator):

    #--------------------------------------------------------------------------
    # Constructor
    #
    # @param width  Overall image width
    # @param height Overall image height
    #
    def __init__ (self, width, height):
        super ().__init__ (width, height)
        self.size = Size2d (width, height)

    #
    # Generate single image containing a fixture like setup
    #
    def generate (self):

        #
        # Generale image as RGB with some background noise
        #
        image = np.zeros ((self.height, self.width, 3), dtype=np.float32)
        mask  = np.zeros ((self.height, self.width), dtype=np.int32)

        #
        # We are adding 1-2 fixtures per image
        #
        for _ in range (random.choice ([1, 2])):
            self.add_fixture (image, mask)

        return image, mask

    #
    # Add single fixture structure to image
    #
    def add_fixture (self, image, mask):

        #
        # Randomly determine fixture size in relation to the target image size
        #
        fixture_size = Size2d (
            width  = random.uniform  (self.size.width * 0.5, self.size.width * 0.8),
            height = random.uniform (self.size.height * 0.5, self.size.height * 0.8)
        )

        fixture_thickness = Size2d (
            width  = max (10, random.uniform (fixture_size.width * 0.01, fixture_size.width * 0.1)),
            height = max (10, random.uniform (fixture_size.height * 0.01, fixture_size.height * 0.1))
        )

        #
        # Draw fixture edges. The fixture can have from 1 edge (just a single element) up to
        # 4 edges (closed frame). The top left cornet point is always at (0, 0)
        #
        number_of_edges = random.choice ([1,1,1,1,2,2,2,3,3,3,4])

        if number_of_edges == 1:
            points = []
            points.append (Point2d (0,                  0))
            points.append (Point2d (fixture_size.width, 0))
            points.append (Point2d (fixture_size.width, fixture_thickness.height))
            points.append (Point2d (0,                  fixture_thickness.height))

        elif number_of_edges == 2:
            points = []
            points.append (Point2d (0,                       0))
            points.append (Point2d (fixture_size.width,      0))
            points.append (Point2d (fixture_size.width,      fixture_thickness.height))
            points.append (Point2d (fixture_thickness.width, fixture_thickness.height))
            points.append (Point2d (fixture_thickness.width, fixture_size.height))
            points.append (Point2d (0,                       fixture_size.height))

        elif number_of_edges == 3:
            points = []
            points.append (Point2d (0,                       0))
            points.append (Point2d (fixture_size.width,      0))
            points.append (Point2d (fixture_size.width,      fixture_thickness.height))
            points.append (Point2d (fixture_thickness.width, fixture_thickness.height))
            points.append (Point2d (fixture_thickness.width, fixture_size.height - fixture_thickness.height))
            points.append (Point2d (fixture_size.width,      fixture_size.height - fixture_thickness.height))
            points.append (Point2d (fixture_size.width,      fixture_size.height))
            points.append (Point2d (0,                       fixture_size.height))

        elif number_of_edges == 4:
            points = []
            points.append (Point2d (0,                                            0))
            points.append (Point2d (fixture_size.width,                           0))
            points.append (Point2d (fixture_size.width,                           fixture_thickness.height))
            points.append (Point2d (fixture_thickness.width,                      fixture_thickness.height))
            points.append (Point2d (fixture_thickness.width,                      fixture_size.height - fixture_thickness.height))
            points.append (Point2d (fixture_size.width - fixture_thickness.width, fixture_size.height - fixture_thickness.height))
            points.append (Point2d (fixture_size.width - fixture_thickness.width, fixture_thickness.height))
            points.append (Point2d (fixture_size.width,                           fixture_thickness.height))
            points.append (Point2d (fixture_size.width,                           fixture_size.height))
            points.append (Point2d (0,                                            fixture_size.height))

        polygon = Polygon2d (points)

        #
        # Move polygon to image center
        #
        center = Point2d (self.size.width / 2, self.size.height / 2)
        polygon.move (center - Point2d (fixture_size.width / 2, fixture_size.height / 2))

        #
        # Randomly rotate polygon round image center
        #
        polygon.rotate (center, random.uniform (0, 2 * math.pi))

        #
        # Move whole polygon randomly some pixels
        #
        delta = Point2d (random.uniform (-self.size.width / 5, self.size.width / 5),
                         random.uniform (-self.size.height / 5, self.size.height / 5))

        polygon.move (delta)

        #
        # Create color source with some random noise and copy the part determined by the polygon into
        # the target image
        #
        source_image = np.zeros (image.shape, dtype=np.float32)
        source_image.fill (0.1)
        source_image = skimage.util.random_noise (source_image, mode='speckle', seed=None, clip=True, mean=0.0, var=0.005)

        source_mask = np.zeros ((image.shape[0], image.shape[1]), dtype=np.int32)
        polygon.draw (source_mask, 1.0, True)

        image[source_mask > 0] = source_image[source_mask > 0]
        mask[source_mask > 0] = 1


#--------------------------------------------------------------------------
# MAIN
#
if __name__ == '__main__':

    random.seed ()

    #
    # Parse command line arguments
    #
    parser = argparse.ArgumentParser ()

    parser.add_argument ('-x', '--width',   type=int, default=640,  help='Width of the generated images')
    parser.add_argument ('-y', '--height',  type=int, default=480,  help='Height of the generated images')

    args = parser.parse_args ()

    generator = FixtureGenerator (args.width, args.height)
    image, mask = generator.generate ()

    utils.show_image ([image, 'Fixture'], [mask, 'Mask'])
