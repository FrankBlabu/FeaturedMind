#!/usr/bin/python3
#
# TestImage.py - Randomly generated test data image
#
# Frank Blankenburg, Mar. 2017
#

import math
import random

from common import Vec2d
from enum import Enum

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFilter
import PIL.ImageEnhance

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
    
    class Direction (Enum):
        NONE    = 0
        UP      = 1
        DOWN    = 2
        LEFT    = 3
        RIGHT   = 4
        UNKNOWN = 5
    
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
        # Mask colors matching the directions
        #
        self.direction_colors = [ (0x00, 0x00, 0x00),
                                 (0xff, 0x00, 0x00),
                                 (0x00, 0xff, 0x00),
                                 (0x00, 0x00, 0xff),
                                 (0xff, 0xff, 0x00),
                                 (0x55, 0x55, 0x55) ]
    
        self.direction_color_index = {}
        for i in range (len (self.direction_colors)):
            self.direction_color_index[self.direction_colors[i]] = i
    
        
        assert len (self.direction_colors) == len (TestImage.Direction)
        
        #
        # The complete test image
        #
        self.image = self.add_background_noise (PIL.Image.new ('L', size))
        
        #
        # Mask marking the feature and border relevant pixels for detection of edges
        #
        self.direction_mask  = PIL.Image.new ('RGB', size)
        
        #
        # Mask used for sample clustering
        #
        self.cluster_mask = PIL.Image.new ('L', size)
            
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
    
        feature_id = 1
        
        self.draw_border ([point.asTuple () for point in border], feature_id)
        feature_id += 1
        
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
                
                    self.create_feature (area, feature_id)                    
                    feature_id += 1
    
    
    #--------------------------------------------------------------------------
    # Draw specimen border into image
    #
    # @param border Polygon defining the specimen border
    # @param id     Unique feature id
    #
    def draw_border (self, border, id):
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

        draw = PIL.ImageDraw.Draw (self.cluster_mask)
        draw.polygon (border, fill=None, outline=id)

        draw = PIL.ImageDraw.Draw (self.direction_mask)
        
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
    # @param draw  Drawing handle
    # @param p1    Starting point of the line
    # @param p2    Target point of the line
    # @param color Line color. If 'none', the direction colors is used instead.
    #
    def draw_line (self, draw, p1, p2, color=None):
        draw.line ([p1, p2], fill=color if color != None else self.get_color_for_line (p1, p2))
        
    
    #--------------------------------------------------------------------------
    # Draw rectangular feature
    #
    # @param offset Rectangle offset (top left point)
    # @param size   Rectangle size
    # @param id     Unique feature id
    #
    def draw_rectangular_feature (self, offset, size, id):
        
        assert id > 0x00 and id <= 0xff
        
        feature_image = self.add_background_noise (PIL.Image.new ('L', size.asTuple ()))
        
        draw = PIL.ImageDraw.Draw (feature_image)
        draw.rectangle (to_native_rect (Vec2d (0, 0), size), fill=None, outline=0xff)
        feature_image = feature_image.filter (PIL.ImageFilter.GaussianBlur (radius=1))
        
        self.image.paste (feature_image, box=offset.asTuple ())
        
        draw  = PIL.ImageDraw.Draw (self.cluster_mask)
        draw.rectangle (to_native_rect (offset, size), fill=None, outline=id)
        
        draw = PIL.ImageDraw.Draw (self.direction_mask)
        
        rect = [offset, offset + size]
        self.draw_line (draw, get_rect_point (rect, 0).asTuple (), get_rect_point (rect, 1).asTuple ())
        self.draw_line (draw, get_rect_point (rect, 1).asTuple (), get_rect_point (rect, 2).asTuple ())
        self.draw_line (draw, get_rect_point (rect, 2).asTuple (), get_rect_point (rect, 3).asTuple ())
        self.draw_line (draw, get_rect_point (rect, 3).asTuple (), get_rect_point (rect, 0).asTuple ())
        
        
            
    #--------------------------------------------------------------------------
    # Draw circular feature
    #
    # @param offset Rectangle offset (top left point)
    # @param size   Rectangle size
    # @param id     Unique feature id
    #
    def draw_circular_feature (self, offset, size, id):

        assert id > 0x00 and id <= 0xff
        
        feature_image = self.add_background_noise (PIL.Image.new ('L', size.asTuple ()))

        draw = PIL.ImageDraw.Draw (feature_image)
        draw.ellipse (to_native_rect (Vec2d (0, 0), size), fill=None, outline=0xff)
        feature_image = feature_image.filter (PIL.ImageFilter.GaussianBlur (radius=1))

        mask_image = PIL.Image.new ('1', size.asTuple ())
        draw = PIL.ImageDraw.Draw (mask_image)
        draw.ellipse (to_native_rect (Vec2d (0, 0), size), fill=0xff, outline=0xff)

        self.image.paste (feature_image, box=offset.asTuple (), mask=mask_image)
        
        draw = PIL.ImageDraw.Draw (self.cluster_mask)
        draw.ellipse (to_native_rect (offset, size), fill=None, outline=id)
        
        rect = to_native_rect (offset, size)
        color = 0xffffff
        
        draw = PIL.ImageDraw.Draw (self.direction_mask)
        draw.ellipse (rect, fill=None, outline=color)
            
        center = (int (round (rect[0][0] + (rect[1][0] - rect[0][0]) / 2)),
                  int (round (rect[0][1] + (rect[1][1] - rect[0][1]) / 2)))
            
        for y in range (rect[0][1], rect[1][1] + 1):
            for x in range (rect[0][0], rect[1][0] + 1):
                r, g, b = self.direction_mask.getpixel ((x, y))
                if r > 0 or g > 0 or b > 0:
                    color = self.get_color_for_line ((0, 0), (-y + center[1], x - center[0]))
                    self.direction_mask.putpixel ((x, y), color)

    #--------------------------------------------------------------------------
    # Generate random feature
    #
    # @param image Image to draw into
    # @param area  Area the feature may occupy
    # @param id    Unique feature id
    #
    def create_feature (self, area, id):
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
            self.draw_rectangular_feature (area[0] + feature_offset, feature_size, id)
            
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
    
            self.draw_circular_feature (area[0] + feature_offset, feature_size, id)
        
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
    def get_color_for_line (self, p1, p2):        
        angle = (math.atan2 (p2[1] - p1[1], p2[0] - p1[0]) + math.pi) % math.pi
                
        n = len (TestImage.Direction) - 2
        segment = round (2 * n * angle / (2 * math.pi)) % n
                
        return self.direction_colors[segment + 1]

    #--------------------------------------------------------------------------
    # Return color matching the given direction
    #
    # @param direction Direction (enum value)
    # @return Color matching the direction
    #
    def get_color_for_direction (self, direction):        
        return self.direction_colors[direction.value]
        

    #--------------------------------------------------------------------------
    # Return sample area from image
    #
    # The image data is normalized in the interval [0...1]
    #
    # @param x    Sample x offset in pixels
    # @param y    Sample y offset in pixels
    # @param size Sample edge size in pixels
    # @return Tuple with sample in float array format / direction / cluster id 
    #
    def get_sample (self, x, y, size):

        sample_area = [x, y, x + size, y + size]

        sample = self.image.crop (sample_area)            
        direction_mask = self.direction_mask.crop (sample_area)
        cluster_mask = self.cluster_mask.crop (sample_area)
        
        #
        # Classify segment content
        #
        direction_distribution = np.zeros ((len (TestImage.Direction)))
        cluster_distribution = np.zeros (0xff)
        
        assert direction_mask.width == cluster_mask.width
        assert direction_mask.height == cluster_mask.height
        
        for y in range (direction_mask.height):
            for x in range (direction_mask.width):
                
                direction_color = direction_mask.getpixel ((x, y))
                if direction_color in self.direction_color_index:
                    direction_distribution[self.direction_color_index[direction_color]] += 1
                    
                cluster_color = cluster_mask.getpixel ((x, y))
                assert cluster_color >= 0 and cluster_color <= 0xff
                
                if cluster_color > 0:
                    cluster_distribution[cluster_color] += 1

        direction_distribution[TestImage.Direction.NONE.value] = 0

        #
        # The label column will contain '0' for an empty segment,
        # '1..n' for the n segment types and 'n+1' for an unclassified segment
        # 
        direction = TestImage.Direction.NONE
        
        if direction_distribution.sum () > 0:
            direction_distribution /= direction_distribution.sum ()
            index = np.argmax (direction_distribution)
            if direction_distribution[index] > self.direction_threshold:
                direction = TestImage.Direction (index)
            else:
                direction = TestImage.Direction.UNKNOWN
        
        cluster = 0
        
        if cluster_distribution.sum () > 0:
            cluster_distribution /= cluster_distribution.sum ()
            cluster = np.argmax (cluster_distribution)
                
        return ([float (d) / 255 for d in sample.getdata ()], direction, cluster) 


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
    image.direction_mask.show (title='Direction mask')
    
    enhancer = PIL.ImageEnhance.Sharpness (image.cluster_mask)
    enhancer.enhance (100.0).show (title='Cluster mask')
    
