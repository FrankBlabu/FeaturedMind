#!/usr/bin/python3
#
# generate_test_data_set.py - Generate test data set
#
# Frank Blankenburg, Feb. 2017
#

import argparse
import math
import pickle
import random

from enum import Enum

import PIL
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFilter
import PIL.ImageStat


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
#
class Configuration:
    def __init__ (self):
        parser = argparse.ArgumentParser ()

        parser.add_argument ('file', type=str, help='Output file name')
        parser.add_argument ('-x', '--width', type=int, default=640,
                             help='Width of the generated images')
        parser.add_argument ('-y', '--height', type=int, default=480,
                             help='Height of the generated images')
        parser.add_argument ('-n', '--number-of-images', type=int, default=5,
                             help='Number of images generated')
        parser.add_argument ('-s', '--sample-size', type=int, default=32,
                             help='Edge size of each sample in pixels')

        args = parser.parse_args ()

        if args.width > 4096:
            assert 'Training image width is too large.'
        if args.height > 4096:
            assert 'Training image height is too large.'

        self.file             = args.file
        self.size             = Vec2d (args.width, args.height)
        self.number_of_images = args.number_of_images
        self.sample_size      = args.sample_size



#--------------------------------------------------------------------------
# CLASS TestImage
#
# Class keeping the data of a single test set image
#
class TestImage:
    
    #
    # Constructor
    #
    # @param size Size of the image in pixels
    #
    def __init__ (self, size):
        #
        # The complete test image
        #
        self.image = self.add_background_noise (PIL.Image.new ('RGBA', size.asTuple ()))
        
        #
        # Mask marking the feature and border relevant pixels
        #
        self.mask  = PIL.Image.new ('1', size.asTuple ())
        
        #
        # List of samples
        #
        # Each sample is a tuple in the format (image, flag) where the flag
        # indicates if the sample show a border (true) or not (false)
        #
        self.samples = []
        
    
    def to_pickle (self, config):
        dict = {}
        dict['sample_size'] = config.sample_size
        dict['samples'] = []
        
        for sample in self.samples:
            dict['samples'].append ((sample[0].tobytes (), sample[1]))
            
        return dict
        
        


    #
    # Draw specimen border into image
    #
    # @param border Polygon defining the specimen border
    #
    def draw_border (self, border):
        border_image = PIL.Image.new ('RGBA', self.image.size)

        draw = PIL.ImageDraw.Draw (border_image)
        
        for y in range (border_image.size[1]):
            for x in range (border_image.size[0]):
                draw.point ((x, y), fill='#' + 3 * str (random.randint (5, 6)))
        
        draw.polygon (border, fill=None, outline='#ffffff')

        border_image = border_image.filter (PIL.ImageFilter.GaussianBlur (radius=1))

        border_mask = PIL.Image.new ('1', self.image.size)

        draw = PIL.ImageDraw.Draw (border_mask)
        draw.polygon (border, fill='#fff', outline='#fff')

        self.image.paste (border_image, mask=border_mask)

        draw = PIL.ImageDraw.Draw (self.mask)
        draw.polygon (border, fill=None, outline='#ffffff')

    #
    # Draw rectangular feature
    #
    def draw_rectangular_feature (self, offset, size):
        feature_image = self.add_background_noise (PIL.Image.new ('RGBA', size.asTuple ()))
        
        draw = PIL.ImageDraw.Draw (feature_image)
        draw.rectangle (self.to_native_rect (Vec2d (0, 0), size), fill=None, outline='#ffffff')
        feature_image = feature_image.filter (PIL.ImageFilter.GaussianBlur (radius=1))
        
        self.image.paste (feature_image, box=offset.asTuple ())
        
        draw = PIL.ImageDraw.Draw (self.mask)
        draw.rectangle (self.to_native_rect (offset, size), fill=None, outline='#ffffff')
            
    #
    # Draw circular feature
    #
    def draw_circular_feature (self, offset, size):
        feature_image = self.add_background_noise (PIL.Image.new ('RGBA', size.asTuple ()))

        draw = PIL.ImageDraw.Draw (feature_image)
        draw.ellipse (self.to_native_rect (Vec2d (0, 0), size), fill=None, outline='#ffffff')
        feature_image = feature_image.filter (PIL.ImageFilter.GaussianBlur (radius=1))

        mask_image = PIL.Image.new ('1', size.asTuple ())
        draw = PIL.ImageDraw.Draw (mask_image)
        draw.ellipse (self.to_native_rect (Vec2d (0, 0), size), fill='#fff', outline='#ffffff')

        self.image.paste (feature_image, box=offset.asTuple (), mask=mask_image)
        
        draw = PIL.ImageDraw.Draw (self.mask)
        draw.ellipse (self.to_native_rect (offset, size), fill=None, outline='#ffffff')

    #
    # Add background noise to an image
    #
    def add_background_noise (self, image):
        draw = PIL.ImageDraw.Draw (image)
        
        for y in range (image.size[1]):
            for x in range (image.size[0]):
                draw.point ((x, y), fill='#' + 3 * str (random.randint (0, 7)))

        return image.filter (PIL.ImageFilter.GaussianBlur (radius=2))


    #
    # Convert a (origin, size) tuple into a rectangle representation
    # 
    # @param origin Origin of the rectangle
    # @param size   Size of the rectangle
    # @return Rectangle in [(x0, y0), (x1, y1]) representation
    #
    def to_native_rect (self, origin, size):
        return [origin.asTuple (), (origin + size - (1, 1)).asTuple ()]
    
    #
    # Create sample from main image
    #
    # This function returns a single sample of the generated image which can
    # be used for training together with a boolean flag indicating if the
    # sample contains some part of a feature
    #
    # @param config Generator configuration
    # @param x      Horizontal offset in pixels
    # @param y      Vertical offset in pixels
    # @return (Sample image, feature flag)
    #
    def create_sample (self, config, x, y):
        pass


#--------------------------------------------------------------------------
# Return the segment rect of an image rect
# 
# @param rect    Image rect [Vec2d (x0, y0), Vec2d (x1, y1)]
# @param segment Segment (x, y) identifier
# @return Rectangle of the segment [Vec2d (x0, y0), Vec2d (x1, y1)]
#
def get_segment (rect, segment):
    x = segment[0]
    y = segment[1]
    
    assert x >= 0 and x <= 2
    assert y >= 0 and y <= 2

    size = Vec2d (rect[1].x - rect[0].x + 1, rect[1].y - rect[0].y + 1)
    
    return [Vec2d (rect[0].x + x * size.x / 3,
                   rect[0].y + y * size.y / 3),
            Vec2d (rect[0].x + (x + 1) * size.x / 3,
                   rect[0].y + (y + 1) * size.y / 3)]
    

#--------------------------------------------------------------------------
# Expand the segment to the bounding box of two segments
#
# @param rect1 Segment rect 1 [Vec2d (x0, y0), Vec2d (x1, y1)]
# @param rect2 Segment rect 2 [Vec2d (x0, y0), Vec2d (x1, y1)]
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
#
def get_rect_size (rect):
    return Vec2d (rect[1].x - rect[0].x + 1, rect[1].y - rect[0].y + 1)


#--------------------------------------------------------------------------
# Generate random feature
#
# @param config Configuration
# @param image  Image to draw into
# @param area   Area the feature may occupy
#
def create_feature (config, image, area):
    area_size = get_rect_size (area)
    
    inner_offset = 0.05 * area_size
    inner_size = area_size - 2 * inner_offset

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
        image.draw_rectangular_feature (area[0] + feature_offset, feature_size)
        
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

        image.draw_circular_feature (area[0] + feature_offset, feature_size)
    
    #
    # Feature type 3: Slotted hole
    #
    elif feature_type == 2:
        pass


#--------------------------------------------------------------------------
# Generate random training image
#
# @param config Configuration
# @return Generated training image
#
def generate_training_samples (config):

    image = TestImage (config.size)

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

    image.draw_border ([point.asTuple () for point in border])

    #
    # Add some features to the available areas
    #
    for y in range (0, 3):
        for x in range (0, 3):
            if available[x][y] and random.randint (0, 9) < 7:
                area = get_segment (border_rect, (x, y))
                available[x][y] = False
                
                if x < 2 and available[x+1][y] and random.randint (0, 9) < 5:
                    area = expand_segment (area, get_segment (border_rect, (x+1, y)))
                    available[x+1][y] = False

                elif y < 2 and available[x][y+1] and random.randint (0, 9) < 5:
                    area = expand_segment (area, get_segment (border_rect, (x, y+1)))
                    available[x][y+1] = False
            
                create_feature (config, image, area)
                
    #
    # Create set of samples and border presence vector
    #
    count = 0
    
    positive_samples = []
    negative_samples = []
     
    for y in range (0, int (math.floor (config.size.y / config.sample_size))):
        for x in range (0, int (math.floor (config.size.x / config.sample_size))):
            sample_area = [x * config.sample_size, 
                           y * config.sample_size,
                           (x + 1) * config.sample_size, 
                           (y + 1) * config.sample_size]

            sample = image.image.crop (sample_area).convert (mode='L')            
            sample_mask = image.mask.crop (sample_area)
            
            stat = PIL.ImageStat.Stat (sample_mask)
            
            if stat.extrema[0][1] > 0:
                positive_samples.append ((sample, True))
            else:
                negative_samples.append ((sample, False))
            
    random.shuffle (positive_samples)
    random.shuffle (negative_samples)
    
    if len (negative_samples) > len (positive_samples):
        negative_samples = negative_samples[0:len (positive_samples)]
        
    image.samples.extend (positive_samples)
    image.samples.extend (negative_samples)
    
    random.shuffle (image.samples)
                        
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
# Create test data sets
#
dict = {}
dict['version'] = 1
dict['sample_size'] = config.sample_size
dict['samples'] = []

for _ in range (config.number_of_images):
    samples = generate_training_samples (config)
    
    for sample in samples.samples:
        dict['samples'].append ((sample[0].tobytes (), sample[1]))
    
with open (config.file, 'wb') as file:
    pickle.dump (dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close ()
