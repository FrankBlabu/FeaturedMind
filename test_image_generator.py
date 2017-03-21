#!/usr/bin/python3
#
# TestImage.py - Randomly generated test data image
#
# Frank Blankenburg, Mar. 2017
#

import math
import random
import unittest

from common import Vec2d

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFilter

import numpy as np


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
# Convert a (origin, size) tuple into a rectangle representation
# 
# @param origin Origin of the rectangle
# @param size   Size of the rectangle
# @return Rectangle in [(x0, y0), (x1, y1]) representation
#
def to_native_rect (origin, size):
    return [origin.asTuple (), (origin + size - (1, 1)).asTuple ()]


#--------------------------------------------------------------------------
# CLASS TestImage
#
# Class keeping the data of a single test set image
#
class TestImage:
    
    #--------------------------------------------------------------------------
    # Constructor
    #
    # @param size Size of the image in pixels
    #
    def __init__ (self, width, height):

        self.width = width
        self.height = height

        self.direction_threshold = 0.7
        
        size = (width, height)

        #
        # Number of segments used for line direction marking and the
        # matching colors (plus 1 for unclassified segments)
        #
        self.arc_segments = 4
        self.arc_colors = [ (0xff, 0x00, 0x00),
                            (0x00, 0xff, 0x00),
                            (0x00, 0x00, 0xff),
                            (0xff, 0xff, 0x00),
                            (0x55, 0x55, 0x55) ]
        
        assert len (self.arc_colors) >= self.arc_segments
        
        self.arc_color_index = {}
        for i in range (len (self.arc_colors)):
            self.arc_color_index[self.arc_colors[i]] = i   
            
        #
        # The complete test image
        #
        self.image = self.add_background_noise (PIL.Image.new ('L', size))
        
        #
        # Mask marking the feature and border relevant pixels for detection of edges
        #
        self.mask  = PIL.Image.new ('RGB', size)
            
        #
        # Compute area used for the specimen border
        #
        outer_border_offset = Vec2d (size) * 5 / 100
        outer_border_limit = [outer_border_offset, Vec2d (size) - outer_border_offset]
    
        inner_border_offset = Vec2d (size) * 25 / 100
        inner_border_limit = [inner_border_offset, Vec2d (size) - inner_border_offset]
        
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
    
            segment = self.get_segment (border_rect, (0, 0))
            used = available[0][0]
            
            border.append (get_rect_point (segment, 3))
            border.append (get_rect_point (segment, 0 if used else 2))
            border.append (get_rect_point (segment, 1))
    
            segment = self.get_segment (border_rect, (2, 0))
            used = available[2][0]
    
            border.append (get_rect_point (segment, 0))
            border.append (get_rect_point (segment, 1 if used else 3))
            border.append (get_rect_point (segment, 2))
    
            segment = self.get_segment (border_rect, (2, 2))
            used = available[2][2]
    
            border.append (get_rect_point (segment, 1))
            border.append (get_rect_point (segment, 2 if used else 0))
            border.append (get_rect_point (segment, 3))
    
            segment = self.get_segment (border_rect, (0, 2))
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
    
            segment = self.get_segment (border_rect, (0, 0))
            border.append (get_rect_point (segment, 0))
            
            segment = self.get_segment (border_rect, (1, 0))
            used = available[1][0]
            
            border.append (get_rect_point (segment, 0))
            if not used:
                border.append (get_rect_point (segment, 3))
                border.append (get_rect_point (segment, 2))
            border.append (get_rect_point (segment, 1))
    
            segment = self.get_segment (border_rect, (2, 0))
            border.append (get_rect_point (segment, 1))
            
            segment = self.get_segment (border_rect, (2, 2))
            border.append (get_rect_point (segment, 2))
    
            segment = self.get_segment (border_rect, (1, 2))
            used = available[1][2]
            
            border.append (get_rect_point (segment, 2))
            if not used:
                border.append (get_rect_point (segment, 1))
                border.append (get_rect_point (segment, 0))
            border.append (get_rect_point (segment, 3))
    
            segment = self.get_segment (border_rect, (0, 2))
            border.append (get_rect_point (segment, 3))
    
    
        #
        # Case 3: Left/right edge segments might be missing
        #
        elif segment_mode == 2:
            available[0][1] = random.randint (0, 9) > 5
            available[2][1] = random.randint (0, 9) > 5
    
            segment = self.get_segment (border_rect, (0, 0))
            border.append (get_rect_point (segment, 0))
            
            segment = self.get_segment (border_rect, (2, 0))
            border.append (get_rect_point (segment, 1))
            
            segment = self.get_segment (border_rect, (2, 1))
            used = available[2][1]
    
            border.append (get_rect_point (segment, 1))
            if not used:        
                border.append (get_rect_point (segment, 0))
                border.append (get_rect_point (segment, 3))
            border.append (get_rect_point (segment, 2))
    
            segment = self.get_segment (border_rect, (2, 2))
            border.append (get_rect_point (segment, 2))
    
            segment = self.get_segment (border_rect, (0, 2))
            border.append (get_rect_point (segment, 3))
    
            segment = self.get_segment (border_rect, (0, 1))
            used = available[0][1]
            
            border.append (get_rect_point (segment, 3))
            if not used:
                border.append (get_rect_point (segment, 2))
                border.append (get_rect_point (segment, 1))
            border.append (get_rect_point (segment, 0))
    
        self.draw_border ([point.asTuple () for point in border])
        
        #
        # Add some features to the available areas
        #
        for y in range (0, 3):
            for x in range (0, 3):
                if available[x][y] and random.randint (0, 9) < 7:
                    area = self.get_segment (border_rect, (x, y))
                    available[x][y] = False
                    
                    if x < 2 and available[x+1][y] and random.randint (0, 9) < 5:
                        area = self.expand_segment (area, self.get_segment (border_rect, (x+1, y)))
                        available[x+1][y] = False
    
                    elif y < 2 and available[x][y+1] and random.randint (0, 9) < 5:
                        area = self.expand_segment (area, self.get_segment (border_rect, (x, y+1)))
                        available[x][y+1] = False
                
                    self.create_feature (area)
    
    
    #--------------------------------------------------------------------------
    # Draw specimen border into image
    #
    # @param border Polygon defining the specimen border
    #
    def draw_border (self, border):
        border_image = PIL.Image.new ('L', self.image.size)

        draw = PIL.ImageDraw.Draw (border_image)
        
        #
        # Draw specimen background (with some noise)
        #
        for y in range (border_image.size[1]):
            for x in range (border_image.size[0]):
                draw.point ((x, y), fill=random.randint (100, 120))
        
        draw.polygon (border, fill=None, outline=0xff)

        border_image = border_image.filter (PIL.ImageFilter.GaussianBlur (radius=1))
        
        border_mask = PIL.Image.new ('1', self.image.size)

        draw = PIL.ImageDraw.Draw (border_mask)
        draw.polygon (border, fill=0xff, outline=0xff)

        self.image.paste (border_image, mask=border_mask)

        draw = PIL.ImageDraw.Draw (self.mask)
        
        last_point = None
        
        for point in border:
            if last_point:
                self.draw_line (draw, last_point, point)
            last_point = point
            
        if len (border) > 1:
            self.draw_line (draw, last_point, border[0])
        

    #--------------------------------------------------------------------------
    # Draw single line
    #
    # The line is drawn with an appropriate direction color
    #
    # @param draw Drawing handle
    # @param p1   Starting point of the line
    # @param p2   Target point of the line
    #
    def draw_line (self, draw, p1, p2):
        draw.line ([p1, p2], fill=self.get_color_for_direction (p1, p2))
        
    
    #--------------------------------------------------------------------------
    # Draw rectangular feature
    #
    def draw_rectangular_feature (self, offset, size):
        feature_image = self.add_background_noise (PIL.Image.new ('L', size.asTuple ()))
        
        draw = PIL.ImageDraw.Draw (feature_image)
        draw.rectangle (to_native_rect (Vec2d (0, 0), size), fill=None, outline=0xff)
        feature_image = feature_image.filter (PIL.ImageFilter.GaussianBlur (radius=1))
        
        self.image.paste (feature_image, box=offset.asTuple ())
        
        draw = PIL.ImageDraw.Draw (self.mask)
        
        rect = [offset, offset + size]
        self.draw_line (draw, get_rect_point (rect, 0).asTuple (), get_rect_point (rect, 1).asTuple ())
        self.draw_line (draw, get_rect_point (rect, 1).asTuple (), get_rect_point (rect, 2).asTuple ())
        self.draw_line (draw, get_rect_point (rect, 2).asTuple (), get_rect_point (rect, 3).asTuple ())
        self.draw_line (draw, get_rect_point (rect, 3).asTuple (), get_rect_point (rect, 0).asTuple ())
        
            
    #--------------------------------------------------------------------------
    # Draw circular feature
    #
    def draw_circular_feature (self, offset, size):
        feature_image = self.add_background_noise (PIL.Image.new ('L', size.asTuple ()))

        draw = PIL.ImageDraw.Draw (feature_image)
        draw.ellipse (to_native_rect (Vec2d (0, 0), size), fill=None, outline=0xff)
        feature_image = feature_image.filter (PIL.ImageFilter.GaussianBlur (radius=1))

        mask_image = PIL.Image.new ('1', size.asTuple ())
        draw = PIL.ImageDraw.Draw (mask_image)
        draw.ellipse (to_native_rect (Vec2d (0, 0), size), fill=0xff, outline=0xff)

        self.image.paste (feature_image, box=offset.asTuple (), mask=mask_image)
        
        rect = to_native_rect (offset, size)
        color = 0xffffff
        
        draw = PIL.ImageDraw.Draw (self.mask)
        draw.ellipse (rect, fill=None, outline=color)
            
        center = (int (round (rect[0][0] + (rect[1][0] - rect[0][0]) / 2)),
                  int (round (rect[0][1] + (rect[1][1] - rect[0][1]) / 2)))
            
        for y in range (rect[0][1], rect[1][1] + 1):
            for x in range (rect[0][0], rect[1][0] + 1):
                r, g, b = self.mask.getpixel ((x, y))
                if r > 0 or g > 0 or b > 0:
                    color = self.get_color_for_direction ((0, 0), (-y + center[1], x - center[0]))
                    self.mask.putpixel ((x, y), color)

    #--------------------------------------------------------------------------
    # Generate random feature
    #
    # @param image Image to draw into
    # @param area  Area the feature may occupy
    #
    def create_feature (self, area):
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
            self.draw_rectangular_feature (area[0] + feature_offset, feature_size)
            
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
    
            self.draw_circular_feature (area[0] + feature_offset, feature_size)
        
        #
        # Feature type 3: Slotted hole
        #
        elif feature_type == 2:
            pass

    #--------------------------------------------------------------------------
    # Compute color for a line segment indicating the direction
    #
    # @param p1 First point
    # @param p2 Second point
    # @return Color matching the direction
    #
    def get_color_for_direction (self, p1, p2):        
        angle = (math.atan2 (p2[1] - p1[1], p2[0] - p1[0]) + math.pi) % math.pi
                
        segment = round (2 * self.arc_segments * angle / (2 * math.pi)) % self.arc_segments
                
        return self.arc_colors[segment]
    

    #--------------------------------------------------------------------------
    # Return sample area from image
    #
    # The image data is normalized in the interval [0...1]
    #
    # @param x    Sample x offset in pixels
    # @param y    Sample y offset in pixels
    # @param size Sample edge size in pixels
    # @return Tuple with sample in float array format / flag for border presence 
    #
    def get_sample (self, x, y, size):

        sample_area = [x, y, x + size, y + size]

        sample = self.image.crop (sample_area)            
        sample_mask = self.mask.crop (sample_area)
        
        #
        # Classify segment content
        #
        distribution = np.zeros ((len (self.arc_colors)))
        
        for y in range (sample_mask.height):
            for x in range (sample_mask.width):
                color = sample_mask.getpixel ((x, y))
                if color in self.arc_color_index:
                    distribution[self.arc_color_index[color]] += 1        

        index = 0

        if distribution.sum () > 0:
            distribution /= distribution.sum ()
            segment = np.argmax (distribution)
            if distribution[segment] > self.direction_threshold:
                index = segment + 1
            else:
                index = self.arc_segments + 1
        
        return ([float (d) / 255 for d in sample.getdata ()], index) 


    #--------------------------------------------------------------------------
    # Add background noise to an image
    #
    @staticmethod
    def add_background_noise (image):
        draw = PIL.ImageDraw.Draw (image)
        
        for y in range (image.size[1]):
            for x in range (image.size[0]):
                draw.point ((x, y), fill=random.randint (0, 80))

        return image.filter (PIL.ImageFilter.GaussianBlur (radius=2))


    
    #--------------------------------------------------------------------------
    # Return the segment rect of an image rect
    # 
    # @param rect    Image rect [Vec2d (x0, y0), Vec2d (x1, y1)]
    # @param segment Segment (x, y) identifier
    # @return Rectangle of the segment [Vec2d (x0, y0), Vec2d (x1, y1)]
    #
    @staticmethod
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
    @staticmethod
    def expand_segment (rect1, rect2):
        return [Vec2d (min (rect1[0].x, rect2[0].x), min (rect1[0].y, rect2[0].y)),
                Vec2d (max (rect1[1].x, rect2[1].x), max (rect1[1].y, rect2[1].y))]
        

#
# MAIN
#
if __name__ == '__main__':
    image = TestImage (640, 480)
    image.image.show (title='Generated image')
    image.mask.show (title='Pixel mask')
    
