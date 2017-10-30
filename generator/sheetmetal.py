#!/usr/bin/python3
#
# sheetmetal.py - Generator for images resembling a sheet metal part
#
# Frank Blankenburg, Mar. 2017
#

import argparse
import random
import numpy as np
import common.utils as utils
import generator.background as background
import generator.fixture as fixture

import skimage.util

from common.geometry import Point2d, Size2d, Rect2d, Ellipse2d, Polygon2d
from generator.generator import Generator


#----------------------------------------------------------------------------------------------------------------------
# CLASS SheetMetalGenerator
#
# This class will generate an image containing a simulated sheet metal like part inclucing some features like drilled
# or punched holes together with a mask marking the location of the pixels belonging to the sheet metal part.
#
class SheetMetalGenerator (Generator):

    #--------------------------------------------------------------------------
    # Configuration
    #


    #
    # Spacing used for distance to border, distance of the generated features, ...
    #
    spacing = Size2d (32, 32)


    #--------------------------------------------------------------------------
    # Constructor
    #
    # @param width  Overall image width
    # @param height Overall image height
    #
    def __init__ (self, width, height):

        self.size = Size2d (width, height)
        self.width = width
        self.height = height


    #--------------------------------------------------------------------------
    # Generate image
    #
    def generate (self):

        #
        # Generale image as RGB with some background noise
        #
        specimen = np.zeros ((self.height, self.width, 3), dtype=np.float32)

        #
        # Mask marking the area covered by the sheet
        #
        mask = np.zeros ((self.height, self.width), dtype=np.float32)

        #
        # Step 1: Compute area used for the specimen border
        #
        # The border will be rectangular in principle but in later steps segments of the
        # specimen are cut out to simulate a more random layout.
        #
        outer_border_limit = [Point2d (SheetMetalGenerator.spacing), Point2d (self.size - SheetMetalGenerator.spacing)]
        inner_border_limit = [Point2d (2 * SheetMetalGenerator.spacing), Point2d (self.size - 2 * SheetMetalGenerator.spacing)]

        border_rect = Rect2d (Point2d (random.randint (outer_border_limit[0].x,
                                                       inner_border_limit[0].x),
                                       random.randint (outer_border_limit[0].y,
                                                       inner_border_limit[0].y)),
                              Point2d (random.randint (inner_border_limit[1].x,
                                                       outer_border_limit[1].x),
                                       random.randint (inner_border_limit[1].y,
                                                       outer_border_limit[1].y)))



        #
        # Compute segments of the specimen used. There are 3x3 segments available.
        #
        available = [[True, True, True], [True, True, True], [True, True, True]]
        border = []

        segment_mode = random.randint (0, 2)

        #
        # Case 1.1: Corner segments might be missing
        #
        if segment_mode == 0:
            available[0][0] = random.randint (0, 9) > 5
            available[2][0] = random.randint (0, 9) > 5
            available[0][2] = random.randint (0, 9) > 5
            available[2][2] = random.randint (0, 9) > 5

            segment = self.get_segment_rect (border_rect, (0, 0))
            used = available[0][0]

            border.append (segment.p3)
            border.append (segment.p0 if used else segment.p2)
            border.append (segment.p1)

            segment = self.get_segment_rect (border_rect, (2, 0))
            used = available[2][0]

            border.append (segment.p0)
            border.append (segment.p1 if used else segment.p3)
            border.append (segment.p2)

            segment = self.get_segment_rect (border_rect, (2, 2))
            used = available[2][2]

            border.append (segment.p1)
            border.append (segment.p2 if used else segment.p0)
            border.append (segment.p3)

            segment = self.get_segment_rect (border_rect, (0, 2))
            used = available[0][2]

            border.append (segment.p2)
            border.append (segment.p3 if used else segment.p1)
            border.append (segment.p0)

        #
        # Case 1.2: Top/down edge segments might be missing
        #
        elif segment_mode == 1:
            available[1][0] = random.randint (0, 9) > 5
            available[1][2] = random.randint (0, 9) > 5

            segment = self.get_segment_rect (border_rect, (0, 0))
            border.append (segment.p0)

            segment = self.get_segment_rect (border_rect, (1, 0))
            used = available[1][0]

            border.append (segment.p0)
            if not used:
                border.append (segment.p3)
                border.append (segment.p2)
            border.append (segment.p1)

            segment = self.get_segment_rect (border_rect, (2, 0))
            border.append (segment.p1)

            segment = self.get_segment_rect (border_rect, (2, 2))
            border.append (segment.p2)

            segment = self.get_segment_rect (border_rect, (1, 2))
            used = available[1][2]

            border.append (segment.p2)
            if not used:
                border.append (segment.p1)
                border.append (segment.p0)
            border.append (segment.p3)

            segment = self.get_segment_rect (border_rect, (0, 2))
            border.append (segment.p3)


        #
        # Case 1.3: Left/right edge segments might be missing
        #
        elif segment_mode == 2:
            available[0][1] = random.randint (0, 9) > 5
            available[2][1] = random.randint (0, 9) > 5

            segment = self.get_segment_rect (border_rect, (0, 0))
            border.append (segment.p0)

            segment = self.get_segment_rect (border_rect, (2, 0))
            border.append (segment.p1)

            segment = self.get_segment_rect (border_rect, (2, 1))
            used = available[2][1]

            border.append (segment.p1)
            if not used:
                border.append (segment.p0)
                border.append (segment.p3)
            border.append (segment.p2)

            segment = self.get_segment_rect (border_rect, (2, 2))
            border.append (segment.p2)

            segment = self.get_segment_rect (border_rect, (0, 2))
            border.append (segment.p3)

            segment = self.get_segment_rect (border_rect, (0, 1))
            used = available[0][1]

            border.append (segment.p3)
            if not used:
                border.append (segment.p2)
                border.append (segment.p1)
            border.append (segment.p0)

        #
        # Step 2: Draw border
        #
        border = Polygon2d (border)

        specimen_image = np.zeros (specimen.shape, dtype=np.float32)
        specimen_image.fill (0.7)
        specimen_image = skimage.util.random_noise (specimen_image, mode='speckle', seed=None, clip=True, mean=0.0, var=0.005)

        specimen_mask = np.zeros (specimen.shape, dtype=np.float32)
        border.draw (specimen_mask, 1.0, fill=True)

        specimen[specimen_mask != 0] = specimen_image[specimen_mask != 0]
        border.draw (mask, 1.0, fill=True)

        #
        # Step 2: Add some features to the available areas
        #
        for y in range (0, 3):
            for x in range (0, 3):
                if available[x][y] and random.randint (0, 9) < 7:
                    area = self.get_segment_rect (border_rect, (x, y))
                    available[x][y] = False

                    if x < 2 and available[x + 1][y] and random.randint (0, 9) < 5:
                        area = area.expanded (self.get_segment_rect (border_rect, (x + 1, y)))
                        available[x + 1][y] = False

                    elif y < 2 and available[x][y + 1] and random.randint (0, 9) < 5:
                        area = area.expanded (self.get_segment_rect (border_rect, (x, y + 1)))
                        available[x][y + 1] = False

                    self.create_feature_set (specimen, mask, area)

        #
        # Step 3: Apply some transformations to the specimen
        #
        scale = (random.uniform (0.5, 0.8), random.uniform (0.5, 0.8))
        shear = random.uniform (0.0, 0.2)
        rotation = random.uniform (-1.0, 1.0)

        specimen = utils.transform (specimen, self.size, scale=scale, shear=shear, rotation=rotation)
        mask = utils.transform (mask, self.size, scale=scale, shear=shear, rotation=rotation)

        return specimen, mask


    #--------------------------------------------------------------------------
    # Generate feature set in the given area
    #
    # @param area Area the feature may occupy
    #
    def create_feature_set (self, specimen, mask, area):

        split_scenarios = []
        split_scenarios.append ([area])
        split_scenarios.append ([area])
        split_scenarios.append ([area])
        split_scenarios.append ([area])
        split_scenarios.append ([area])
        split_scenarios.append ([area.split ((0, 1, 3), (0, 1, 3)),
                                 area.split ((2, 2, 3), (0, 1, 3)),
                                 area.split ((0, 2, 3), (2, 2, 3))])
        split_scenarios.append ([area.split ((0, 2, 3), (0, 0, 3)),
                                 area.split ((0, 2, 3), (1, 2, 3))])
        split_scenarios.append ([area.split ((0, 2, 3), (0, 0, 3)),
                                 area.split ((0, 2, 3), (1, 2, 3))])
        split_scenarios.append ([area.split ((0, 2, 3), (0, 1, 3)),
                                 area.split ((0, 2, 3), (2, 2, 3))])
        split_scenarios.append ([area.split ((0, 1, 3), (0, 2, 3)),
                                 area.split ((2, 2, 3), (0, 2, 3))])
        split_scenarios.append ([area.split ((0, 1, 3), (0, 1, 3)),
                                 area.split ((2, 2, 3), (0, 1, 3)),
                                 area.split ((0, 1, 3), (2, 2, 3)),
                                 area.split ((2, 2, 3), (2, 2, 3))])
        split_scenarios.append ([area.split ((0, 0, 3), (0, 0, 3)),
                                 area.split ((1, 1, 3), (0, 0, 3)),
                                 area.split ((2, 2, 3), (0, 0, 3)),
                                 area.split ((0, 0, 3), (1, 1, 3)),
                                 area.split ((1, 1, 3), (1, 1, 3)),
                                 area.split ((2, 2, 3), (1, 1, 3)),
                                 area.split ((0, 0, 3), (2, 2, 3)),
                                 area.split ((1, 1, 3), (2, 2, 3)),
                                 area.split ((2, 2, 3), (2, 2, 3))])

        split = split_scenarios[random.randint (0, len (split_scenarios) - 1)]

        for area in split:
            self.create_feature (specimen, mask, area)


    #--------------------------------------------------------------------------
    # Generate random feature
    #
    # @param image Image to draw into
    # @param area  Area the feature may occupy
    #
    def create_feature (self, specimen, mask, area):
        inner_rect = Rect2d (area.p0 + SheetMetalGenerator.spacing / 2, area.p2 - SheetMetalGenerator.spacing / 2)

        if inner_rect.size ().width > SheetMetalGenerator.spacing.width and inner_rect.size ().height > SheetMetalGenerator.spacing.height:

            offset = Size2d (random.randint (0, int (inner_rect.size ().width - SheetMetalGenerator.spacing.width)),
                             random.randint (0, int (inner_rect.size ().height - SheetMetalGenerator.spacing.height)))

            feature_rect = Rect2d (inner_rect.p0 + offset / 2, inner_rect.p2 - offset / 2)
            feature_type = random.randint (0, 1)

            #
            # Feature type 1: Rectangle
            #
            if feature_type == 0:
                feature_rect.draw (specimen, 0.0, fill=True)
                feature_rect.draw (mask, 0.0, fill=True)

            #
            # Feature type 2: Circle
            #
            elif feature_type == 1:
                circle = Ellipse2d (feature_rect).to_circle ()
                circle.draw (specimen, 0.0, fill=True)
                circle.draw (mask, 0.0, fill=True)


    #--------------------------------------------------------------------------
    # Return the segment rect in the (3, 3) raster of an image rect
    #
    # @param rect    Complete image rect
    # @param segment Segment (x, y) identifier in (3, 3) raster
    # @return Rectangle of the matching segment part
    #
    @staticmethod
    def get_segment_rect (rect, segment):
        x = segment[0]
        y = segment[1]

        assert x >= 0 and x <= 2
        assert y >= 0 and y <= 2

        return Rect2d (Point2d (rect.p0.x + x * rect.size ().width / 3,
                                rect.p0.y + y * rect.size ().height / 3),
                       Point2d (rect.p0.x + (x + 1) * rect.size ().width / 3,
                                rect.p0.y + (y + 1) * rect.size ().height / 3))


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
    parser.add_argument ('-f', '--fixture', action='store_true', default=False, help='Add fixture')

    background.BackgroundGenerator.add_to_args_definition (parser)
    args = parser.parse_args ()

    background_generator = background.BackgroundGenerator.create (args)
    image, mask = background_generator.generate ()

    sheet_generator = SheetMetalGenerator (args.width, args.height)
    sheet, mask = sheet_generator.generate ()
    image[mask > 0.5] = sheet[mask > 0.5]

    if args.fixture:
        fixture_generator = fixture.FixtureGenerator (args.width, args.height)
        fixture, mask = fixture_generator.generate ()
        image[mask > 0.5] = fixture[mask > 0.5]

    utils.show_image ([image, 'Sheet metal'],
                      [mask,  'Specimen mask'])
