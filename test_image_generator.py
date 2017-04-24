#!/usr/bin/python3
#
# test_image_generator.py - Randomly generated test data image
#
# Class for generation of a single test image together with additional data
# needed for various training cases
#
# Frank Blankenburg, Mar. 2017
#

import argparse
import random
import numpy as np
import common.utils as utils 

import skimage.filters

from common.geometry import Point2d, Size2d, Rect2d, Ellipse2d, Polygon2d 


#--------------------------------------------------------------------------
# CLASS TestImage
#
# Class keeping the data of a single test set image
#
class TestImage:
        
    #--------------------------------------------------------------------------
    # Configuration
    #
    border_spacing = 10 
        
        
    #--------------------------------------------------------------------------
    # Constructor
    #
    # @param args Image parameters as parsed from the command line
    #
    def __init__ (self, args):

        self.size = Size2d (args.width, args.height)
        self.sample_size = Size2d (args.sample_size, args.sample_size)
        self.objects = []

        #
        # The complete test image as grayscale
        #
        self.image = self.add_background_noise (np.zeros ((args.height, args.width), dtype=np.float32), bias=0.2, delta=0.2)

        #
        # Mask marking the feature and border relevant pixels for detection of edges. The image
        # is grayscale and will contain the feature id as pixel value.
        #
        self.border_mask = np.zeros ((args.height, args.width), dtype=np.float32)
        
        #
        # Step 1: Compute area used for the specimen border
        #
        # The border will be rectangular in principle but in later steps segments of the
        # specimen are cut out to simulate a more random layout.
        #
        outer_border_offset = self.sample_size / 2
        outer_border_limit = [Point2d (outer_border_offset), Point2d (self.size - outer_border_offset)]
    
        inner_border_offset = self.sample_size * 2                
        inner_border_limit = [Point2d (inner_border_offset), Point2d (self.size - inner_border_offset)]
        
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
    
        feature_id = 1
        
        self.draw_border (Polygon2d (border), feature_id)
        feature_id += 1
        
        #
        # Step 2: Add some features to the available areas
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
                
                    self.create_feature_set (area, feature_id)                    
                    feature_id += 1
                    
    
    #--------------------------------------------------------------------------
    # Draw specimen border into image
    #
    # @param border     Polygon defining the specimen border
    # @param feature_id Unique feature id
    #
    def draw_border (self, border, feature_id):
        
        specimen_image = np.zeros (self.image.shape, dtype=np.float32)
        specimen_image = self.add_background_noise (specimen_image, bias=0.5, delta=0.1)

        specimen_mask = np.zeros (self.image.shape, dtype=np.float32)
        border.draw (specimen_mask, 1.0, fill=True)

        self.image[specimen_mask != 0] = specimen_image[specimen_mask != 0] 

        border.draw (self.border_mask, self.id_to_color (feature_id))
        
        self.objects.append (border)
                        

    #--------------------------------------------------------------------------
    # Draw rectangular feature
    #
    # @param rect       Rectangle used for the feature
    # @param feature_id Unique feature id
    #
    def draw_rectangular_feature (self, rect, feature_id):
        
        feature_image = self.add_background_noise (self.create_array (rect.size ()), bias=0.2, delta=0.2)

        x = int (rect.p0.x)
        y = int (rect.p0.y)
        
        self.image[y:y+feature_image.shape[0], x:x+feature_image.shape[1]] = feature_image
        
        rect.draw (self.border_mask, self.id_to_color (feature_id))
        self.objects.append (rect)
        
        
            
    #--------------------------------------------------------------------------
    # Draw circular feature
    #
    # @param ellipse    Ellipse of the feature
    # @param feature_id Unique feature id
    #
    def draw_circular_feature (self, ellipse, feature_id):
        
        e = ellipse.move_to (Point2d (ellipse.radius.x, ellipse.radius.y))
        
        feature_image = self.add_background_noise (self.create_array (ellipse.rect ().size () + Size2d (1, 1)), bias=0.2, delta=0.2)                

        mask = np.zeros (feature_image.shape, dtype=np.float32)
        e.draw (mask, 1.0, fill=True)

        x = int (ellipse.rect ().p0.x)
        y = int (ellipse.rect ().p0.y)
        
        self.image[y:y+feature_image.shape[0], x:x+feature_image.shape[1]][mask > 0] = feature_image[mask > 0]

        ellipse.draw (self.border_mask, self.id_to_color (feature_id))                
        self.objects.append (ellipse)
        

    #--------------------------------------------------------------------------
    # Generate feature set in the given area
    #
    # @param image      Image to draw into
    # @param area       Area the feature may occupy
    # @param feature_id Unique feature id
    # @return Next available feature id
    #
    def create_feature_set (self, area, feature_id):

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
            feature_id = self.create_feature (area, feature_id)
            
        return feature_id

    #--------------------------------------------------------------------------
    # Generate random feature
    #
    # @param image      Image to draw into
    # @param area       Area the feature may occupy
    # @param feature_id Unique feature id
    # @return Next available feature id
    #
    def create_feature (self, area, feature_id):
        inner_rect = Rect2d (area.p0 + self.sample_size / 2, area.p2 - self.sample_size / 2)
    
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
                self.draw_circular_feature (Ellipse2d (feature_rect).to_circle (), feature_id)
                
        return feature_id + 1


    #--------------------------------------------------------------------------
    # Return single sample area from image
    #
    # The image is in pillow format
    # The label indicates the feature this sample belongs to (0 = no feature)
    #
    # @param area Area to sample
    # @return Tuple with sample in (image / label) 
    #
    def get_sample (self, area):

        r = area.as_tuple ()

        sample = self.image[r[1]:r[3]+1,r[0]:r[2]+1]
        mask = self.border_mask[r[1]:r[3]+1,r[0]:r[2]+1]
        
        return sample, self.color_to_id (mask.max ())
    
            
    #----------------------------------------------------------------------------
    # Extract specimen mask containing the area covered by the specimen
    #
    # @return Mask marking the specimen area
    #
    def get_specimen_mask (self):
        
        mask = self.create_array (self.size)
        
        for obj in self.objects:
            if isinstance (obj, Polygon2d):
                obj.draw (mask, 1.0, fill=True)
                
        for obj in self.objects:
            if not isinstance (obj, Polygon2d):
                obj.draw (mask, 0.0, fill=True)
                    
        return mask
    
    
    #----------------------------------------------------------------------------
    # Extract feature mask image containing the given feature type
    #
    # @param feature_type Type of features addressed by the cluster image or
    #                     'None' if any feature type is requested 
    # @return (Generated feature mask image, Boolean indicating if there is data
    #         in the mask at all)
    #
    def get_feature_mask (self, feature_type=None):
        
        mask = self.create_array (self.size)
        
        found = False

        for obj in self.objects:
            
            if feature_type is None or type (obj) is feature_type:
                obj.draw (mask, 1.0, fill=False)
                found = True
                    
        return mask, found
    

    #--------------------------------------------------------------------------
    # Add background noise to an image
    #
    @staticmethod
    def add_background_noise (image, bias, delta):
        
        for y in range (image.shape[0]):
            for x in range (image.shape[1]):
                image[y][x] = random.uniform (max (bias - delta, 0.0), min (bias + delta, 1.0))

        return skimage.filters.gaussian (image)

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


    #----------------------------------------------------------------------------
    # Convert feature id into a color for the border mask
    #
    def id_to_color (self, feature_id):
        assert feature_id <= 100
        return feature_id / 100.0

    #----------------------------------------------------------------------------
    # Convert color of the border mask into feature id
    #
    def color_to_id (self, color):
        assert color >= 0 and color <= 1
        return int (round (color * 100))

    #----------------------------------------------------------------------------
    # Create overlay displaying the generated / found labels
    #
    # @param labels Found labels with the same shape as self.labels
    #
    def create_result_overlay (self, labels):
    
        overlay = np.zeros ((int (round (self.size.height)), int (round (self.size.width)), 4))
    
        for y in range (labels.shape[0]):
            for x in range (labels.shape[1]):
                rect = Rect2d (Point2d (x * self.sample_size.width, y * self.sample_size.height), self.sample_size)
                _, expected = self.get_sample (rect)
                
                found = labels[y][x]
    
                #
                # Case 1: Hit
                #
                if expected > 0 and found > 0:
                    rect.draw (overlay, (0.0, 1.0, 0.0, 0.2), fill=True)
                    rect.draw (overlay, (0.0, 1.0, 0.0, 1.0), fill=False)
                    
                #
                # Case 2: False positive
                #
                elif expected == 0 and found > 0:
                    rect.draw (overlay, (0.0, 0.0, 1.0, 0.2), fill=True)
                    rect.draw (overlay, (0.0, 0.0, 1.0, 1.0), fill=False)
                    
                #
                # Case 3: False negative
                #
                elif expected > 0 and found == 0:
                    rect.draw (overlay, (1.0, 0.0, 0.0, 0.2), fill=True)
                    rect.draw (overlay, (1.0, 0.0, 0.0, 1.0), fill=False)
                                        
        return overlay



#--------------------------------------------------------------------------
# MAIN
#
if __name__ == '__main__':

    random.seed ()

    #
    # Parse command line arguments
    #
    parser = argparse.ArgumentParser ()
    
    parser.add_argument ('-x', '--width',       type=int, default=640, help='Width of the generated images')
    parser.add_argument ('-y', '--height',      type=int, default=480, help='Height of the generated images')
    parser.add_argument ('-s', '--sample-size', type=int, default=16,  help='Edge size of each sample in pixels')
    parser.add_argument ('-m', '--mode',        type=str, default='borders', choices=['borders', 'segments'], help='Image generation mode')

    args = parser.parse_args ()

    image = TestImage (args)

    if args.mode == 'borders':    
        utils.show_image ([utils.to_rgb (image.image),                'Generated image'],
                          [utils.to_rgb (image.get_specimen_mask ()), 'Specimen mask'])
    elif args.mode == 'segments':
        utils.show_image ([utils.to_rgb (image.image),                           'Generated image'],
                          [utils.to_rgb (image.get_feature_mask (Polygon2d)[0]), 'Cluster mask (Border)'],
                          [utils.to_rgb (image.get_feature_mask (Rect2d)[0]),    'Cluster mask (Rect)'],
                          [utils.to_rgb (image.get_feature_mask (Ellipse2d)[0]), 'Cluster mask (Ellipse)'])
    
