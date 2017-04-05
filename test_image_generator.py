#!/usr/bin/python3
#
# test_image_generator.py - Randomly generated test data image
#
# Class for generation of a single test image which can be later divided into
# multiple rectangluar samples for training.
#
# Frank Blankenburg, Mar. 2017
#

import argparse
import random

from common.geometry import Point2d, Size2d, Rect2d

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFilter
import PIL.ImageEnhance
import PIL.ImageStat



#--------------------------------------------------------------------------
# CLASS TestImage
#
# Class keeping the data of a single test set image
#
class TestImage:
    
    #--------------------------------------------------------------------------
    # Constructor
    #
    # @param args Image parameters
    #
    def __init__ (self, args):

        self.width = args.width
        self.height = args.height
        self.sample_size = Size2d (args.sample_size, args.sample_size)
        self.objects = []

        size = Size2d (self.width, self.height)

        #
        # The complete test image
        #
        self.image = self.add_background_noise (PIL.Image.new ('L', size.as_tuple ()))
        
        #
        # Mask marking the feature and border relevant pixels for detection of edges
        #
        self.label_mask  = PIL.Image.new ('L', size.as_tuple ())
        
        #
        # Compute area used for the specimen border
        #
        outer_border_offset = 2 * self.sample_size
        outer_border_limit = [Point2d (outer_border_offset), Point2d (size - outer_border_offset)]
    
        inner_border_offset = size * 25 / 100
        
        
        inner_border_limit = [Point2d (inner_border_offset), Point2d (size - inner_border_offset)]
        
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
        
        specimen_image = PIL.Image.new ('L', self.image.size)

        draw = PIL.ImageDraw.Draw (specimen_image)
        
        #
        # Draw specimen background (with some noise)
        #
        for y in range (specimen_image.size[1]):
            for x in range (specimen_image.size[0]):
                draw.point ((x, y), fill=random.randint (100, 120))
        
        native_border = [point.as_tuple () for point in border]
        
        draw.polygon (native_border, fill=None, outline=0xff)

        specimen_image = specimen_image.filter (PIL.ImageFilter.GaussianBlur (radius=1))        
        specimen_mask = PIL.Image.new ('1', self.image.size)

        draw = PIL.ImageDraw.Draw (specimen_mask)
        draw.polygon (native_border, fill=0xff, outline=0xff)

        self.image.paste (specimen_image, mask=specimen_mask)

        draw = PIL.ImageDraw.Draw (self.label_mask)
        draw.polygon (native_border, fill=None, outline=feature_id)
                

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
        
        draw  = PIL.ImageDraw.Draw (self.label_mask)
        draw.rectangle (rect.as_tuple (), fill=None, outline=feature_id)
        
        self.objects.append (rect)
        
        
            
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
        
        draw = PIL.ImageDraw.Draw (self.label_mask)
        draw.ellipse (rect.as_tuple (), fill=None, outline=feature_id)
        
        self.objects.append (rect)
        

    #--------------------------------------------------------------------------
    # Generate random feature
    #
    # @param image      Image to draw into
    # @param area       Area the feature may occupy
    # @param feature_id Unique feature id
    #
    def create_feature (self, area, feature_id):
        inner_rect = Rect2d (area.p0 + self.sample_size + Size2d (2, 2),
                             area.p2 - self.sample_size - Size2d (2, 2))
    
        if inner_rect.size ().width > self.sample_size.width and inner_rect.size ().height > self.sample_size.height:
    
            offset = Size2d (random.randint (0, int (inner_rect.size ().width - self.sample_size.width)),
                             random.randint (0, int (inner_rect.size ().height - self.sample_size.height)))
            
            feature_rect = Rect2d (inner_rect.p0 + offset / 2, inner_rect.p2 - offset / 2)
            feature_type = random.randint (0, 1)
            
            #
            # Feature type 1: Rectangle
            #
            if feature_type == 0:
                self.draw_rectangular_feature (feature_rect, feature_id)
                
            #
            # Feature type 2: Circle
            #
            elif feature_type == 1:
                if feature_rect.size ().width < feature_rect.size ().height:
                    feature_rect = feature_rect.resized (Size2d (feature_rect.size ().width, feature_rect.size ().width))             
                else:
                    feature_rect = feature_rect.resized (Size2d (feature_rect.size ().height, feature_rect.size ().height))             
                    
                self.draw_circular_feature (feature_rect, feature_id)


    #--------------------------------------------------------------------------
    # Return sample area from image
    #
    # The image data is normalized in the interval [0...1]
    # The label indicates the feature this sample belongs to (0 = no feature)
    #
    # @param area Area to sample
    # @return Tuple with sample in (float array format / label) 
    #
    def get_sample (self, area):

        assert type (area) is Rect2d

        crop_area = area + Size2d (1, 1)

        sample = self.image.crop (crop_area.as_tuple ())            
        label_mask = self.label_mask.crop (crop_area.as_tuple ())

        label_stat = PIL.ImageStat.Stat (label_mask)
        
        return ([float (d) / 255 for d in sample.getdata ()], label_stat.extrema[0][1]) 


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
        

#--------------------------------------------------------------------------
# MAIN
#
if __name__ == '__main__':

    random.seed ()

    #
    # Parse command line arguments
    #
    parser = argparse.ArgumentParser ()
    
    parser.add_argument ('-x', '--width',       type=int, default=1024, help='Width of the generated images')
    parser.add_argument ('-y', '--height',      type=int, default=768,  help='Height of the generated images')
    parser.add_argument ('-s', '--sample-size', type=int, default=16,   help='Edge size of each sample in pixels')

    args = parser.parse_args ()

    image = TestImage (args)
    image.image.show (title='Generated image')
    
    enhancer = PIL.ImageEnhance.Sharpness (image.label_mask)
    enhancer.enhance (100.0).show (title='Label mask')
    
