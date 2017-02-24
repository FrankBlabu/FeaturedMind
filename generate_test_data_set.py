#!/usr/bin/python3
#
# generate_test_data_set.py - Generate test data set
#
# Frank Blankenburg, Feb. 2017
#

import argparse
import random
import PIL
import PIL.Image
import PIL.ImageDraw

#--------------------------------------------------------------------------
# CLASS Configuration
#
# Container keeping the generator configuration

class Configuration:
    def __init__ (self):
        parser = argparse.ArgumentParser ()

        parser.add_argument ('file', type=str, help='Output file name')
        parser.add_argument ('-x', '--width', type=int, default=640,
                             help='Width of the generated images')
        parser.add_argument ('-y', '--height', type=int, default=480,
                             help='Height of the generated images')
        parser.add_argument ('-n', '--number-of-images', type=int, default=20,
                             help='Number of images generated')
        parser.add_argument ('-s', '--sample-size', type=int, default=32,
                             help='Edge size of each sample in pixels')

        args = parser.parse_args ()

        if args.width > 4096:
            assert 'Training image width is too large.'
        if args.height > 4096:
            assert 'Training image height is too large.'

        self.width            = args.width
        self.height           = args.height
        self.number_of_images = args.number_of_images
        self.sample_size      = args.sample_size

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
# Generate a random arc based test image
#
# @param size Size of the generated image
# @return Generated image
#
def generate_arc (config):
    image = PIL.Image.new ('RGBA', (config.width, config.height))
    draw = PIL.ImageDraw.Draw (image)

    draw.arc ((0, 0, config.width - 1, config.height - 1), 0, 90, fill=None) 

    return image

#--------------------------------------------------------------------------
# Return the segment rect of an image rect
# 
# @param rect    Image rect [(x0, y0), (x1, y1)]
# @param segment Segment (x, y) identifier
# @return Rectangle of the segment [(x0, y0), (x1, y1)]
#
def get_segment (rect, segment):
    x = segment[0]
    y = segment[1]
    
    assert x >= 0 and x <= 2
    assert y >= 0 and y <= 2

    size = (rect[1][0] - rect[0][0], rect[1][1] - rect[0][1])
    
    return [(rect[0][0] + int (round (x * size[0] / 3)),
             rect[0][1] + int (round (y * size[1] / 3))),
            (rect[0][0] + int (round ((x + 1) * size[0] / 3)),
             rect[0][1] + int (round ((y + 1) * size[1] / 3)))]
#--------------------------------------------------------------------------
# Return corner point of rectangle
#
# @param rect Rectangle
# @param n    Point number (from top left clockwise)
# @return Point coordinates
#
def get_rect_point (rect, n):
    if n == 0:
        return rect[0]
    elif n == 1:
        return (rect[1][0], rect[0][1])
    elif n == 2:
        return rect[1]
    elif n == 3:
        return (rect[0][0], rect[1][1])

    assert False and 'Illegal rectangle point index'


#--------------------------------------------------------------------------
# Generate random training image
#
# @param config Configuration
# @return Generated training image
#
def generate_training_image (config):

    #
    # Final image
    #
    image = PIL.Image.new ('RGBA', (config.width, config.height))
    #draw = PIL.ImageDraw.Draw (image)

    #
    # Image pattern overlay
    #
    pattern = PIL.Image.new ('RGBA', image.size)
    draw_pattern = PIL.ImageDraw.Draw (pattern)

    #
    # Compute area used for the specimen border
    #
    outer_border_offset = (int (round (5 * config.width / 100)),
                           int (round (5 * config.height / 100)))
    outer_border_limit = [(0 + outer_border_offset[0],
                           0 + outer_border_offset[1]),
                          (config.width - outer_border_offset[0],
                           config.height - outer_border_offset[1])]

    inner_border_offset = (int (round (25 * config.width / 100)),
                           int (round (25 * config.height / 100)))
    inner_border_limit = [(0 + inner_border_offset[0],
                           0 + inner_border_offset[1]),
                          (config.width - inner_border_offset[0],
                           config.height - inner_border_offset[1])]

    border_rect = [(random.randint (outer_border_limit[0][0],
                                    inner_border_limit[0][0]),
                    random.randint (outer_border_limit[0][1],
                                    inner_border_limit[0][1])),
                   (random.randint (inner_border_limit[1][0],
                                    outer_border_limit[1][0]),
                    random.randint (inner_border_limit[1][1],
                                    outer_border_limit[1][1]))]

    #
    # Compute segments of the specimen used. There are 3x3 segments available
    #
    used_segments = 3 * [3 * [True]]
    border = []

    segment_mode = 1#random.randint (0, 2)

    #
    # Corner segments might be missing 
    #
    if segment_mode == 0:
        used_segments[0][0] = random.randint (0, 9) > 5
        used_segments[2][0] = random.randint (0, 9) > 5
        used_segments[0][2] = random.randint (0, 9) > 5
        used_segments[2][2] = random.randint (0, 9) > 5

        segment = get_segment (border_rect, (0, 0))
        used = used_segments[0][0]
        
        border.append (get_rect_point (segment, 3))
        border.append (get_rect_point (segment, 0 if used else 2))
        border.append (get_rect_point (segment, 1))

        segment = get_segment (border_rect, (2, 0))
        used = used_segments[2][0]

        border.append (get_rect_point (segment, 0))
        border.append (get_rect_point (segment, 1 if used else 3))
        border.append (get_rect_point (segment, 2))

        segment = get_segment (border_rect, (2, 2))
        used = used_segments[2][2]

        border.append (get_rect_point (segment, 1))
        border.append (get_rect_point (segment, 2 if used else 0))
        border.append (get_rect_point (segment, 3))

        segment = get_segment (border_rect, (0, 2))
        used = used_segments[0][2]

        border.append (get_rect_point (segment, 2))
        border.append (get_rect_point (segment, 3 if used else 1))
        border.append (get_rect_point (segment, 0))

    #
    # Edge segments might be missing
    #
    elif segment_mode == 1:
        used_segments[0][1] = random.randint (0, 9) > 5
        used_segments[2][1] = random.randint (0, 9) > 5
        used_segments[1][0] = random.randint (0, 9) > 5
        used_segments[1][2] = random.randint (0, 9) > 5

        segment = get_segment (border_rect, (0, 0))
        border.append (get_rect_point (segment, 0))
        
        segment = get_segment (border_rect, (1, 0))
        used = used_segments[1][0]
        
        border.append (get_rect_point (segment, 0))
        if not used:
            border.append (get_rect_point (segment, 3))
            border.append (get_rect_point (segment, 2))
        border.append (get_rect_point (segment, 1))

        segment = get_segment (border_rect, (2, 0))
        border.append (get_rect_point (segment, 1))
        
        segment = get_segment (border_rect, (2, 1))
        used = used_segments[2][1]

        border.append (get_rect_point (segment, 1))
        if not used:        
            border.append (get_rect_point (segment, 0))
            border.append (get_rect_point (segment, 3))
        border.append (get_rect_point (segment, 2))

        segment = get_segment (border_rect, (2, 2))
        border.append (get_rect_point (segment, 2))

        segment = get_segment (border_rect, (1, 2))
        used = used_segments[1][2]
        
        border.append (get_rect_point (segment, 2))
        if not used:
            border.append (get_rect_point (segment, 1))
            border.append (get_rect_point (segment, 0))
        border.append (get_rect_point (segment, 3))

        segment = get_segment (border_rect, (0, 2))
        border.append (get_rect_point (segment, 3))

        segment = get_segment (border_rect, (0, 1))
        used = used_segments[0][1]
        
        border.append (get_rect_point (segment, 3))
        if not used:
            border.append (get_rect_point (segment, 2))
            border.append (get_rect_point (segment, 1))
        border.append (get_rect_point (segment, 0))
        
    #
    # Center segment might be missing
    #
    elif segment_mode == 2:
        pass
        
    draw_pattern.rectangle (border_rect, fill=None, outline='#555555')
    draw_pattern.polygon (border, fill=None, outline='#ffffff')

    image.paste (pattern, mask=pattern)
    
    return image
    

#--------------------------------------------------------------------------
# MAIN
#

random.seed ()

#
# Parse command line arguments
#
config = Configuration ()

#
# Create 8 bit grayscale image
#
image = generate_training_image (config)

image.show ()
