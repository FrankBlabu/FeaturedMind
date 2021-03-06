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

import skimage.filters
import skimage.util

import generator.background
import generator.generator
import generator.fixture
import generator.tools

from common.geometry import Point2d, Size2d, Polygon2d


#----------------------------------------------------------------------------------------------------------------------
# CLASS FixtureGenerator
#
# This class will generate an image containing a simulated fixture like object together with a mask marking
# its pixel positions.
#
class FixtureGenerator (generator.generator.Generator):

    TYPE = 'fixture'

    #--------------------------------------------------------------------------
    # Constructor
    #
    def __init__ (self, args):
        super ().__init__ (args)

        self.size = Size2d (args.width, args.height)
        self.texture_creator = generator.tools.MetalTextureCreator (width=args.width, height=args.height, color=0.2, shine=0.1)

    #--------------------------------------------------------------------------
    # Return if this generator creates an active layer which must be detected as a separate image segmentation class
    #
    def is_active_layer (self):
        return True

    #--------------------------------------------------------------------------
    # Generate single image containing a fixture like setup
    #
    def generate (self):

        #
        # Generale image as RGB with some background noise
        #
        image = np.zeros ((self.height, self.width, self.depth), dtype=np.float64)
        mask  = np.zeros ((self.height, self.width), dtype=np.float64)

        #
        # We are adding 1-2 fixtures per image
        #
        for _ in range (random.choice ([1, 2])):
            image, mask = self.add_fixture (image, mask)

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
        # Create metal texture color source and copy the part determined by the polygon into the target image
        #
        source_image = self.texture_creator.create ()

        source_mask = np.zeros ((source_image.shape[0], source_image.shape[1]), dtype=source_image.dtype)
        polygon.draw (source_mask, 1.0, True)

        copy_mask = skimage.filters.gaussian (source_mask, sigma=1, mode='nearest')
        copy_mask = np.dstack ((copy_mask, copy_mask, copy_mask))

        blended = (1 - copy_mask) * image + copy_mask * source_image
        image[source_image > 0] = blended[source_image > 0]
        mask[source_mask > 0] = source_mask[source_mask > 0]

        return image, mask


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

    generator.background.BackgroundGenerator.add_to_args_definition (parser)

    args = parser.parse_args ()

    parts = [generator.background.BackgroundGenerator.create (args),
             generator.fixture.FixtureGenerator (args)]

    source = generator.generator.StackedGenerator (args, parts)
    image, mask = source.generate ()

    assert len (mask.shape) == 3
    assert image.shape[0] == args.height
    assert image.shape[1] == args.width
    assert image.shape[2] == 3
    assert image.dtype == np.float64

    assert len (mask.shape) == 2
    assert mask.shape[0] == args.height
    assert mask.shape[1] == args.width
    assert mask.dtype == np.float64

    utils.show_image ([image, 'Fixture'], [mask, 'Mask'])
