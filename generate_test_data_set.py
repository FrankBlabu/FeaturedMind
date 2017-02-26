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
# CLASS Vec2d
# 
# Two dimensional vector
class Vec2d:
    def __init__ (self, x, y=None):
        if type (x) == tuple and y == None:
            self.x = x[0]
            self.y = x[1]
        else:
            assert y != None
            self.x = x
            self.y = y
        
    def __add__ (a, b):
        assert type (a) == Vec2d
        if type (b) == tuple:
            b = Vec2d (b[0], b[1])
        
        return Vec2d (a.x + b.x, a.y + b.y)
            
    def __sub__ (a, b):
        assert type (a) == Vec2d
        if type (b) == tuple:
            b = Vec2d (b[0], b[1])

        return Vec2d (a.x - b.x, a.y - b.y)
    
    def __mul__ (a, b):
        assert type (a) == Vec2d
        assert type (b) == int or type (b) == float
        
        return Vec2d (a.x * b, a.y * b)
        
    def __rmul__ (a, b):
        assert type (a) == Vec2d
        assert type (b) == int or type (b) == float
        
        return Vec2d (a.x * b, a.y * b)

    def __truediv__ (a, b):
        assert type (a) == Vec2d
        assert type (b) == int or type (b) == float
        
        return Vec2d (a.x / b, a.y / b) 

    def __not__ (self):
        self.x = -self.x
        self.y = -self.y
            
    def __repr__ (self):
        return 'Vec2d ({0}, {1})'.format (self.x, self.y)

    def asTuple (self):
        return (int (round (self.x)), int (round (self.y)))


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

        self.size             = Vec2d (args.width, args.height)
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
    overlay = PIL.Image.new ('RGBA', size.asTuple ())
    
    draw = PIL.ImageDraw.Draw (overlay)
    draw.ellipse ([(0, 0), (size - (1, 1)).asTuple ()],
                  outline='#ffffff', fill=None)

    rotated = overlay.rotate (rotation, expand=True,
                              resample=PIL.Image.BILINEAR)
    
    image.paste (rotated, (int (center.x - rotated.size.x / 2),
                           int (center.y - rotated.size.y / 2)), rotated)

#--------------------------------------------------------------------------
# Generate a random arc based test image
#
# @param size Size of the generated image
# @return Generated image
#
def generate_arc (config):
    image = PIL.Image.new ('RGBA', config.size.asTuple ())
    draw = PIL.ImageDraw.Draw (image)

    draw.arc ([(0, 0), (config.size - (1, 1)).asTuple ()], 0, 90, fill=None) 

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

    size = Vec2d (rect[1].x - rect[0].x, rect[1].y - rect[0].y)
    
    return [Vec2d (rect[0].x + x * size.x / 3,
                   rect[0].y + y * size.y / 3),
            Vec2d (rect[0].x + (x + 1) * size.x / 3,
                   rect[0].y + (y + 1) * size.y / 3)]

#--------------------------------------------------------------------------
# Expand the segment to the bounding box of two segments
#
# @param rect1 Segment rect 1 [(x0, y0), (x1, y1)]
# @param rect2 Segment rect 2 [(x0, y0), (x1, y1)]
# @return Resulting segment covering the bounding box
#
def expand_segment (rect1, rect2):
    return [Vec2d (min (rect1[0].x, rect2[0].x), min (rect1[0].y, rect2[0].y)),
            Vec2d (max (rect1[1].x, rect2[1].x), max (rect1[1].y, rect2[1].y))]
    
    
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
        return Vec2d (rect[1].x, rect[0].y)
    elif n == 2:
        return rect[1]
    elif n == 3:
        return Vec2d (rect[0].x, rect[1].y)

    assert False and 'Illegal rectangle point index'

#--------------------------------------------------------------------------
# Return size of a rectangle
#
# @param rect Rectangle
# @return Size of the rectangle
def get_rect_size (rect):
    return Vec2d (rect[1].x - rect[0].x + 1, rect[1].y - rect[0].y + 1)

#--------------------------------------------------------------------------
# Convert a (origin, size) tuple into a rectangle representation
# 
# @param origin Origin of the rectangle
# @param size   Size of the rectangle
# @return Rectangle in [(x0, y0), (x1, y1]) representation
def to_rect (origin, size):
    return [origin.asTuple (), (origin + size - (1, 1)).asTuple ()]


#--------------------------------------------------------------------------
# Generate random feature
#
# @param config Configuration
# @param size   Size of the area the feature may occupy
def create_feature (config, size):
    feature_image = PIL.Image.new ('RGBA', size.asTuple ())
    draw = PIL.ImageDraw.Draw (feature_image)

    inner_offset = 0.05 * size
    inner_size = size - 2 * inner_offset

    assert inner_size.x > 10 and inner_size.y > 10

    feature_size = Vec2d (random.randint (10, int (round (inner_size.x))),
                          random.randint (10, int (round (inner_size.y))))
    
    feature_offset = Vec2d (inner_offset.x + random.randint (0, int (round (inner_size.x - feature_size.x))),
                            inner_offset.y + random.randint (0, int (round (inner_size.y - feature_size.y))))

    feature_type = random.randint (0, 1)
    
    #
    # Feature type 1: Rectangle
    #
    if feature_type == 0:
        draw.rectangle (to_rect (feature_offset, feature_size), fill=None, outline='#ffffff')
        
    #
    # Feature type 2: Circle
    #
    elif feature_type == 1:
        if feature_size.x > feature_size.y:
            feature_offset = feature_offset + ((feature_size.x - feature_size.y) / 2, 0)
            feature_size = Vec2d (feature_size.y, feature_size.y)
        else:
            feature_offset = feature_offset + (0, (feature_size.y - feature_size.x) / 2)
            feature_size = Vec2d (feature_size.x, feature_size.x)

        draw.ellipse (to_rect (feature_offset, feature_size), fill=None, outline='#ffffff')
    
    #
    # Feature type 3: Slotted hole
    #
    elif feature_type == 2:
        pass

    return feature_image

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
    image = PIL.Image.new ('RGBA', config.size.asTuple ())
    
    #
    # Compute area used for the specimen border
    #
    outer_border_offset = config.size * 5 / 100
    outer_border_limit = [outer_border_offset, config.size - outer_border_offset]

    inner_border_offset = config.size * 25 / 100
    inner_border_limit = [inner_border_offset, config.size - inner_border_offset]
    
    border_rect = [Vec2d (random.randint (outer_border_limit[0].x,
                                          inner_border_limit[0].x),
                          random.randint (outer_border_limit[0].y,
                                          inner_border_limit[0].y)),
                   Vec2d (random.randint (inner_border_limit[1].x,
                                          outer_border_limit[1].x),
                          random.randint (inner_border_limit[1].y,
                                          outer_border_limit[1].y))]

    #
    # Compute segments of the specimen used. There are 3x3 segments available.
    #
    available = [[True, True, True], [True, True, True], [True, True, True]]
    border = []

    segment_mode = random.randint (0, 2)

    #
    # Case 1: Corner segments might be missing 
    #
    if segment_mode == 0:
        available[0][0] = random.randint (0, 9) > 5
        available[2][0] = random.randint (0, 9) > 5
        available[0][2] = random.randint (0, 9) > 5
        available[2][2] = random.randint (0, 9) > 5

        segment = get_segment (border_rect, (0, 0))
        used = available[0][0]
        
        border.append (get_rect_point (segment, 3))
        border.append (get_rect_point (segment, 0 if used else 2))
        border.append (get_rect_point (segment, 1))

        segment = get_segment (border_rect, (2, 0))
        used = available[2][0]

        border.append (get_rect_point (segment, 0))
        border.append (get_rect_point (segment, 1 if used else 3))
        border.append (get_rect_point (segment, 2))

        segment = get_segment (border_rect, (2, 2))
        used = available[2][2]

        border.append (get_rect_point (segment, 1))
        border.append (get_rect_point (segment, 2 if used else 0))
        border.append (get_rect_point (segment, 3))

        segment = get_segment (border_rect, (0, 2))
        used = available[0][2]

        border.append (get_rect_point (segment, 2))
        border.append (get_rect_point (segment, 3 if used else 1))
        border.append (get_rect_point (segment, 0))

    #
    # Case 2: Top/down edge segments might be missing
    #
    elif segment_mode == 1:
        available[1][0] = random.randint (0, 9) > 5
        available[1][2] = random.randint (0, 9) > 5

        segment = get_segment (border_rect, (0, 0))
        border.append (get_rect_point (segment, 0))
        
        segment = get_segment (border_rect, (1, 0))
        used = available[1][0]
        
        border.append (get_rect_point (segment, 0))
        if not used:
            border.append (get_rect_point (segment, 3))
            border.append (get_rect_point (segment, 2))
        border.append (get_rect_point (segment, 1))

        segment = get_segment (border_rect, (2, 0))
        border.append (get_rect_point (segment, 1))
        
        segment = get_segment (border_rect, (2, 2))
        border.append (get_rect_point (segment, 2))

        segment = get_segment (border_rect, (1, 2))
        used = available[1][2]
        
        border.append (get_rect_point (segment, 2))
        if not used:
            border.append (get_rect_point (segment, 1))
            border.append (get_rect_point (segment, 0))
        border.append (get_rect_point (segment, 3))

        segment = get_segment (border_rect, (0, 2))
        border.append (get_rect_point (segment, 3))


    #
    # Case 3: Left/right edge segments might be missing
    #
    elif segment_mode == 2:
        available[0][1] = random.randint (0, 9) > 5
        available[2][1] = random.randint (0, 9) > 5

        segment = get_segment (border_rect, (0, 0))
        border.append (get_rect_point (segment, 0))
        
        segment = get_segment (border_rect, (2, 0))
        border.append (get_rect_point (segment, 1))
        
        segment = get_segment (border_rect, (2, 1))
        used = available[2][1]

        border.append (get_rect_point (segment, 1))
        if not used:        
            border.append (get_rect_point (segment, 0))
            border.append (get_rect_point (segment, 3))
        border.append (get_rect_point (segment, 2))

        segment = get_segment (border_rect, (2, 2))
        border.append (get_rect_point (segment, 2))

        segment = get_segment (border_rect, (0, 2))
        border.append (get_rect_point (segment, 3))

        segment = get_segment (border_rect, (0, 1))
        used = available[0][1]
        
        border.append (get_rect_point (segment, 3))
        if not used:
            border.append (get_rect_point (segment, 2))
            border.append (get_rect_point (segment, 1))
        border.append (get_rect_point (segment, 0))

    #
    # Copy border overlay into main image
    #
    border_image = PIL.Image.new ('RGBA', image.size)
    draw_features = PIL.ImageDraw.Draw (border_image)

    poly = [point.asTuple () for point in border]

    draw_features.polygon (poly, fill=None, outline='#ffffff')

    image.paste (border_image, mask=border_image)

    #
    # Add some features to the available areas
    #
    for y in range (0, 3):
        for x in range (0, 3):
            if available[x][y] and random.randint (0, 9) < 5:
                area = get_segment (border_rect, (x, y))
                available[x][y] = False
                
                if x < 2 and available[x+1][y] and random.randint (0, 9) < 5:
                    area = expand_segment (area, get_segment (border_rect, (x+1, y)))
                    available[x+1][y] = False

                if y < 2 and available[x][y+1] and random.randint (0, 9) < 5:
                    area = expand_segment (area, get_segment (border_rect, (x, y+1)))
                    available[x][y+1] = False
            
                feature_image = create_feature (config, get_rect_size (area))
                image.paste (feature_image, box=area[0].asTuple (), mask=feature_image)
                                
                
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
