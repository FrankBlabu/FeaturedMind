#!/usr/bin/python3
#
# test_image_generator.py - Randomly generated test data image
#
# Class for generation of a single test image which can be later divided into
# multiple rectangluar samples for training.
#
# Frank Blankenburg, Mar. 2017
#

import math
import random

from common.geometry import Point2d, Size2d, Rect2d
from enum import Enum

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFilter
import PIL.ImageEnhance

import numpy as np



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
        
        size = Size2d (width, height)

        #
        # Mask colors matching the directions
        #
        # Each color matches a direction and is used in the image mask to
        # assign the right direction label to each sample area
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
        self.image = self.add_background_noise (PIL.Image.new ('L', size.as_tuple ()))
        
        #
        # Mask marking the feature and border relevant pixels for detection of edges
        #
        self.direction_mask  = PIL.Image.new ('RGB', size.as_tuple ())
        
        #
        # Mask used for sample clustering
        #
        self.cluster_mask = PIL.Image.new ('L', size.as_tuple ())
            
        #
        # Compute area used for the specimen border
        #
        outer_border_offset = size * 5 / 100
        outer_border_limit = [Point2d (outer_border_offset), Point2d (size - outer_border_offset)]
    
        inner_border_offset = Point2d (size) * 25 / 100
        inner_border_limit = [inner_border_offset, Point2d (size - inner_border_offset)]
        
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
        # Case 1: Corner segments might be missing 
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
        # Case 2: Top/down edge segments might be missing
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
        # Case 3: Left/right edge segments might be missing
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
    
        feature_id = 1
        
        self.draw_border (border, feature_id)
        feature_id += 1
        
        #
        # Add some features to the available areas
        #        
        for y in range (0, 3):
            for x in range (0, 3):
                if available[x][y] and random.randint (0, 9) < 7:
                    area = self.get_segment_rect (border_rect, (x, y))
                    available[x][y] = False
                    
                    if x < 2 and available[x+1][y] and random.randint (0, 9) < 5:
                        area = area.expanded (self.get_segment_rect (border_rect, (x+1, y)))
                        available[x+1][y] = False
    
                    elif y < 2 and available[x][y+1] and random.randint (0, 9) < 5:
                        area = area.expanded (self.get_segment_rect (border_rect, (x, y+1)))
                        available[x][y+1] = False
                
                    self.create_feature (area, feature_id)                    
                    feature_id += 1
    
    
    #--------------------------------------------------------------------------
    # Draw specimen border into image
    #
    # @param border     Polygon defining the specimen border
    # @param feature_id Unique feature id
    #
    def draw_border (self, border, feature_id):
        
        assert feature_id > 0x00 and feature_id <= 0xff
        
        border_image = PIL.Image.new ('L', self.image.size)

        draw = PIL.ImageDraw.Draw (border_image)
        
        #
        # Draw specimen background (with some noise)
        #
        for y in range (border_image.size[1]):
            for x in range (border_image.size[0]):
                draw.point ((x, y), fill=random.randint (100, 120))
        
        native_border = [point.as_tuple () for point in border]
        
        draw.polygon (native_border, fill=None, outline=0xff)

        border_image = border_image.filter (PIL.ImageFilter.GaussianBlur (radius=1))        
        border_mask = PIL.Image.new ('1', self.image.size)

        draw = PIL.ImageDraw.Draw (border_mask)
        draw.polygon (native_border, fill=0xff, outline=0xff)

        self.image.paste (border_image, mask=border_mask)

        draw = PIL.ImageDraw.Draw (self.cluster_mask)
        draw.polygon (native_border, fill=None, outline=feature_id)

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
        draw.line ([p1.as_tuple (), p2.as_tuple ()], fill=color if color != None else self.get_color_for_line (p1, p2))
        
    
    #--------------------------------------------------------------------------
    # Draw rectangular feature
    #
    # @param rect       Rectangle used for the feature
    # @param feature_id Unique feature id
    #
    def draw_rectangular_feature (self, rect, feature_id):
        
        assert feature_id > 0x00 and feature_id <= 0xff
        
        feature_image = self.add_background_noise (PIL.Image.new ('L', rect.size ().as_tuple ()))

        r = rect.move_to (Point2d (0, 0))
        
        draw = PIL.ImageDraw.Draw (feature_image)
        draw.rectangle (r.as_tuple (), fill=None, outline=0xff)
        feature_image = feature_image.filter (PIL.ImageFilter.GaussianBlur (radius=1))
        
        self.image.paste (feature_image, box=rect.p0.as_tuple ())
        
        draw  = PIL.ImageDraw.Draw (self.cluster_mask)
        draw.rectangle (rect.as_tuple (), fill=None, outline=feature_id)
        
        draw = PIL.ImageDraw.Draw (self.direction_mask)
        
        self.draw_line (draw, rect.p0, rect.p1)
        self.draw_line (draw, rect.p1, rect.p2)
        self.draw_line (draw, rect.p2, rect.p3)
        self.draw_line (draw, rect.p3, rect.p0)
        
        
            
    #--------------------------------------------------------------------------
    # Draw circular feature
    #
    # @param rect       Rectangle used for the circular feature
    # @param feature_id Unique feature id
    #
    def draw_circular_feature (self, rect, feature_id):

        assert feature_id > 0x00 and feature_id <= 0xff
        
        r = rect.move_to (Point2d (0, 0))
        
        feature_image = self.add_background_noise (PIL.Image.new ('L', rect.size ().as_tuple ()))

        draw = PIL.ImageDraw.Draw (feature_image)
        draw.ellipse (r.as_tuple (), fill=None, outline=0xff)
        feature_image = feature_image.filter (PIL.ImageFilter.GaussianBlur (radius=1))

        mask_image = PIL.Image.new ('1', rect.size ().as_tuple ())
        draw = PIL.ImageDraw.Draw (mask_image)
        draw.ellipse (r.as_tuple (), fill=0xff, outline=0xff)

        self.image.paste (feature_image, box=rect.p0.as_tuple (), mask=mask_image)
        
        draw = PIL.ImageDraw.Draw (self.cluster_mask)
        draw.ellipse (rect.as_tuple (), fill=None, outline=feature_id)
        
        color = 0xffffff
        
        draw = PIL.ImageDraw.Draw (self.direction_mask)
        draw.ellipse (rect.as_tuple (), fill=None, outline=color)
            
        for y in range (int (rect.p0.y), int (rect.p2.y)):
            for x in range (int (rect.p0.x), int (rect.p2.x)):
                r, g, b = self.direction_mask.getpixel ((x, y))
                if r > 0 or g > 0 or b > 0:
                    color = self.get_color_for_line (Point2d (0, 0), Point2d (-y + rect.center ().y, x -rect.center ().x))
                    self.direction_mask.putpixel ((x, y), color)


    #--------------------------------------------------------------------------
    # Generate random feature
    #
    # @param image      Image to draw into
    # @param area       Area the feature may occupy
    # @param feature_id Unique feature id
    #
    def create_feature (self, area, feature_id):
        inner_offset = 0.05 * area.size ()
        inner_size = area.size () - 2 * inner_offset
    
        assert inner_size.width > 10 and inner_size.height > 10
    
        feature_size = Size2d (random.randint (10, int (round (inner_size.width))),
                                random.randint (10, int (round (inner_size.height))))
        
        feature_offset = Point2d (inner_offset.width + random.randint (0, int (round (inner_size.width - feature_size.width))),
                                  inner_offset.height + random.randint (0, int (round (inner_size.height - feature_size.height))))
    
        feature_type = random.randint (0, 1)
        
        #
        # Feature type 1: Rectangle
        #
        if feature_type == 0:
            self.draw_rectangular_feature (Rect2d (area.p0 + feature_offset, feature_size), feature_id)
            
        #
        # Feature type 2: Circle
        #
        elif feature_type == 1:
            if feature_size.width > feature_size.height:
                feature_offset = feature_offset + Point2d ((feature_size.width - feature_size.height) / 2, 0)
                feature_size = Size2d (feature_size.height, feature_size.height)
            else:
                feature_offset = feature_offset + Point2d (0, (feature_size.height - feature_size.width) / 2)
                feature_size = Size2d (feature_size.width, feature_size.width)
    
            self.draw_circular_feature (Rect2d (area.p0 + feature_offset, feature_size), feature_id)
        
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
        angle = (math.atan2 (p2.y - p1.y, p2.x - p1.x) + math.pi) % math.pi
                
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
    # @param rect    Image rect
    # @param segment Segment (x, y) identifier
    # @return Rectangle of the segment
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
        

#
# MAIN
#
if __name__ == '__main__':
    image = TestImage (640, 480)
    image.image.show (title='Generated image')
    image.direction_mask.show (title='Direction mask')
    
    enhancer = PIL.ImageEnhance.Sharpness (image.cluster_mask)
    enhancer.enhance (100.0).show (title='Cluster mask')
    
