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

import skimage.util

from common.geometry import Point2d, Size2d, Rect2d, Ellipse2d, Polygon2d


#--------------------------------------------------------------------------
# CLASS SheetMetalGenerator
#
# This class will generate an image containing a simulated sheet metal like
# part inclucing some features like drilled or pounded holes together
# with a set of masks marking the location of the involved elements.
#
class SheetMetalGenerator:

    #--------------------------------------------------------------------------
    # Configuration
    #
    spacing = Size2d (32, 32)


    #--------------------------------------------------------------------------
    # Constructor
    #
    # @param width                Overall image width
    # @param height               Overall image height
    # @param background_generator Background generator class
    #
    def __init__ (self, width, height, background_generator):

        self.size = Size2d (width, height)

        #
        # Generale image as RGB with some background noise
        #
        self.specimen = np.zeros ((height, width, 3), dtype=np.float32)

        #
        # Mask marking the area covered by the sheet
        #
        self.mask = np.zeros ((height, width), dtype=np.float32)

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

        specimen_image = np.zeros (self.specimen.shape, dtype=np.float32)
        specimen_image.fill (0.6)
        specimen_image = skimage.util.random_noise (specimen_image, mode='speckle', seed=None, clip=True, mean=0.0, var=0.005)

        specimen_mask = np.zeros (self.specimen.shape, dtype=np.float32)
        border.draw (specimen_mask, 1.0, fill=True)

        self.specimen[specimen_mask != 0] = specimen_image[specimen_mask != 0]
        border.draw (self.mask, 1.0, fill=True)

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

                    self.create_feature_set (area)

        #
        # Step 3: Apply some transformations to the specimen
        #
        scale = (random.uniform (0.5, 0.8), random.uniform (0.5, 0.8))
        shear = random.uniform (0.0, 0.2)
        rotation = random.uniform (-1.0, 1.0)

        self.specimen = utils.transform (self.specimen, self.size, scale=scale, shear=shear, rotation=rotation)
        self.mask = utils.transform (self.mask, self.size, scale=scale, shear=shear, rotation=rotation)

        #
        # Step 4: Setup random background pattern
        #
        self.image = background_generator.generate ()

        #
        # Step 5: Combine image and background
        #
        self.mask[self.mask <= 0.5] = 0.0
        self.mask[self.mask > 0.5] = 1.0
        self.image[self.mask > 0.5] = self.specimen[self.mask > 0.5]


    #--------------------------------------------------------------------------
    # Draw rectangular feature
    #
    # @param rect Rectangle used for the feature
    #
    def draw_rectangular_feature (self, rect):
        rect.draw (self.specimen, 0.0, fill=True)
        rect.draw (self.mask, 0.0, fill=True)


    #--------------------------------------------------------------------------
    # Draw circular feature
    #
    # @param ellipse Ellipse of the feature
    #
    def draw_circular_feature (self, ellipse):
        ellipse.draw (self.specimen, 0.0, fill=True)
        ellipse.draw (self.mask, 0.0, fill=True)


    #--------------------------------------------------------------------------
    # Generate feature set in the given area
    #
    # @param image Image to draw into
    # @param area  Area the feature may occupy
    #
    def create_feature_set (self, area):

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
            self.create_feature (area)


    #--------------------------------------------------------------------------
    # Generate random feature
    #
    # @param image Image to draw into
    # @param area  Area the feature may occupy
    #
    def create_feature (self, area):
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
                self.draw_rectangular_feature (feature_rect)

            #
            # Feature type 2: Circle
            #
            elif feature_type == 1:
                self.draw_circular_feature (Ellipse2d (feature_rect).to_circle ())


    #--------------------------------------------------------------------------
    # Generate a numpy array matching the given size
    #
    def create_array (self, size):
        return np.zeros ((int (round (size.height)), int (round (size.width))), dtype=np.float32)


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

    parser.add_argument ('-x', '--width',     type=int, default=640,  help='Width of the generated images')
    parser.add_argument ('-y', '--height',    type=int, default=480,  help='Height of the generated images')
    parser.add_argument ('-d', '--directory', type=str, default=None, help='Directory for database based background generation')
    parser.add_argument ('-m', '--mode',  action='store', choices=['rects', 'imagedb'], default='rects', help='Background creation mode')

    args = parser.parse_args ()

    if args.mode == 'rects':
        background_generator = background.NoisyRectBackgroundGenerator (Size2d (args.width, args.height))
    elif args.mode == 'imagedb':
        assert args.directory is not None
        background_generator = background.ImageBackgroundGenerator (args.directory, Size2d (args.width, args.height))

    image = SheetMetalGenerator (args.width, args.height, background_generator)

    utils.show_image ([utils.to_rgb (image.image), 'Sheet metal'],
                      [utils.to_rgb (image.mask),  'Specimen mask'])
