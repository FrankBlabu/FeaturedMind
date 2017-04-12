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

import skimage.filters
import skimage.io

from matplotlib import pyplot as plt
from common.geometry import Point2d, Size2d, Rect2d, Ellipse2d, Polygon2d


#--------------------------------------------------------------------------
# CLASS TestImage
#
# Class keeping the data of a single test set image
#
class TestImage:
        
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
        self.image = self.add_background_noise (np.zeros ((args.height, args.width), dtype=np.float32))
        
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
        outer_border_offset = 2 * self.sample_size
        outer_border_limit = [Point2d (outer_border_offset), Point2d (self.size - outer_border_offset)]
    
        inner_border_offset = self.size * 25 / 100
                
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
        
        self.draw_border (Polygon2d (border), self.id_to_color (feature_id))
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
                
                    self.create_feature (area, feature_id)                    
                    feature_id += 1
                    
    
    #--------------------------------------------------------------------------
    # Draw specimen border into image
    #
    # @param border     Polygon defining the specimen border
    # @param feature_id Unique feature id
    #
    def draw_border (self, border, feature_id):
        
        specimen_image = np.zeros (self.image.shape, dtype=np.float32)

        #
        # Draw specimen background (with some noise)
        #
        for y in range (specimen_image.shape[0]):
            for x in range (specimen_image.shape[1]):
                specimen_image[y, x] = random.uniform (0.3, 0.5)
        
        border.draw (specimen_image, 1.0, fill=False)

        specimen_image = skimage.filters.gaussian (specimen_image)
                
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
        
        feature_image = self.add_background_noise (self.create_array (rect.size ()))

        r = rect.move_to (Point2d (0, 0))
        r.draw (feature_image, 1.0)
        
        feature_image = skimage.filters.gaussian (feature_image)

        x = int (round (rect.p0.x))
        y = int (round (rect.p0.y))
        
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
        
        feature_image = self.add_background_noise (self.create_array (ellipse.rect ().size () + Size2d (1, 1)))
        
        e.draw (feature_image, 1.0)
        
        feature_image = skimage.filters.gaussian (feature_image)

        mask = np.zeros (feature_image.shape, dtype=np.float32)
        e.draw (mask, 1.0, fill=True)

        x = int (round (ellipse.rect ().p0.x))
        y = int (round (ellipse.rect ().p0.y))
        
        self.image[y:y+feature_image.shape[0], x:x+feature_image.shape[1]][mask > 0] = feature_image[mask > 0]

        ellipse.draw (self.border_mask, self.id_to_color (feature_id))                
        self.objects.append (ellipse)
        

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
                self.draw_circular_feature (Ellipse2d (feature_rect).to_circle (), feature_id)


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

        sample = self.image[r[1]:r[3],r[0]:r[2]]
        return sample, sample.max ()
    
            
    #----------------------------------------------------------------------------
    # Extract cluster mask image containing the given feature type
    #
    # @param feature_type Type of features addressed by the cluster image
    # @return Generated cluster mask image, Boolean indicating if there is data
    #         in the mask at all
    #
    def get_cluster_mask (self, feature_type):
        
        mask = self.create_array (self.size)
        
        found = False
        
        for obj in self.objects:
            
            if type (obj) is feature_type:
                obj.draw (mask, 1.0, fill=feature_type is not Polygon2d)
                found = True
                    
        return mask, found
    

    #--------------------------------------------------------------------------
    # Add background noise to an image
    #
    @staticmethod
    def add_background_noise (image):
        
        for y in range (image.shape[0]):
            for x in range (image.shape[1]):
                image[y][x] = random.uniform (0.0, 0.4)

        return skimage.filters.gaussian (image)

    #--------------------------------------------------------------------------
    # Generate a numpy array matching the given size
    #
    def create_array (self, size):
        return np.zeros ((int (size.height), int (size.width)), dtype=np.float32)

    
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
    # Create overlay displaying the generated / found labels
    #
    # @param labels Found labels with the same shape as self.labels
    #
    def create_result_overlay (self, labels):
    
        overlay = np.zeros ((int (self.size.width), int (self.size.height), 4))
    
        for y in range (labels.shape[0]):
            for x in range (labels.shape[1]):
                rect = Rect2d (Point2d (x * self.sample_size.width, y * self.sample_size.height), self.sample_size)
                _, expected = self.get_sample (rect)
                
                found = labels[y][x]
    
                #
                # Case 1: Hit
                #
                if expected > 0 and found > 0:
                    rect.draw (overlay, (0.0, 1.0, 0.0, 0.1), fill=True)
                    rect.draw (overlay, (0.0, 1.0, 0.0, 1.0), fill=False)
                    
                #
                # Case 2: False positive
                #
                elif expected == 0 and found > 0:
                    rect.draw (overlay, (0.0, 0.0, 1.0, 0.1), fill=True)
                    rect.draw (overlay, (0.0, 0.0, 1.0, 1.0), fill=False)
                    
                #
                # Case 3: False negative
                #
                elif expected > 0 and found == 0:
                    rect.draw (overlay, (1.0, 0.0, 0.0, 0.1), fill=True)
                    rect.draw (overlay, (1.0, 0.0, 0.0, 1.0), fill=False)
                                        
        return overlay


        
#--------------------------------------------------------------------------
# Show image with title
#
def show_image (image, title):
    
    skimage.io.imshow (image)
    plt.show ()


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
    
    show_image (image.image, "Generated image")

    #enhancer = PIL.ImageEnhance.Sharpness (image.border_mask)
    #show_image (enhancer.enhance (100.0), "Border mask")
    
    #show_image (image.get_cluster_mask (Rect2d)[0],    "Cluster mask (Rect)")
    #show_image (image.get_cluster_mask (Ellipse2d)[0], "Cluster mask (Ellipse)")
    #show_image (image.get_cluster_mask (Polygon2d)[0], "Cluster mask (Border)")
    
