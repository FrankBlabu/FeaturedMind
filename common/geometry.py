#!/usr/bin/python3
#
# geometry.py - Data structures for geometry operations
#
# Frank Blankenburg, Mar. 2017
#

import math
import random
import unittest


#--------------------------------------------------------------------------
# CLASS Point2d
# 
# Two dimensional point
#
class Point2d:
    
    def __init__ (self, x, y=None):
        if y is None:
            self.x = x.width
            self.y = x.height
        else:
            self.x = float (x)
            self.y = float (y)
        
    def __add__ (self, other):
        return Point2d (self.x + other.x, self.y + other.y) if type (other) is Point2d else Point2d (self.x + other.width, self.y + other.height)
            
    def __sub__ (self, other):
        return Point2d (self.x - other.x, self.y - other.y) if type (other) is Point2d else Point2d (self.x - other.width, self.y - other.height)
    
    def __mul__ (self, other):
        return Point2d (self.x * other, self.y * other)
        
    def __rmul__ (self, other):
        return Point2d (self.x * other, self.y * other)

    def __truediv__ (self, other):
        return Point2d (self.x / other, self.y / other) 

    def __not__ (self):
        self.x = -self.x
        self.y = -self.y
            
    def __eq__ (self, other):
        return self.x == other.x and self.y == other.y

    def __lt__ (self, other):
        return self.y < other.y if self.x == other.x else self.x < other.x
            
    def __repr__ (self):
        return 'Point2d ({0}, {1})'.format (self.x, self.y)

    def as_tuple (self):
        return (int (round (self.x)), int (round (self.y)))


#--------------------------------------------------------------------------
# CLASS Size2d
# 
# Two dimensional size
#
class Size2d:
    
    def __init__ (self, width, height):
        self.width = float (width)
        self.height = float (height)
        
    def __add__ (self, other):
        return Size2d (self.width + other.width, self.height + other.height) if type (other) == Size2d else Size2d (self.width + other.x, self.height + other.y) 
            
    def __sub__ (self, other):
        return Size2d (self.width - other.width, self.height - other.height) if type (other) == Size2d else Size2d (self.width - other.x, self.height - other.y) 
    
    def __mul__ (self, other):
        return Size2d (self.width * other, self.height * other)
        
    def __rmul__ (self, other):
        return Size2d (self.width * other, self.height * other)

    def __truediv__ (self, other):
        return Size2d (self.width / other, self.height / other) 

    def __repr__ (self):
        return 'Size2d ({0}, {1})'.format (self.width, self.height)

    def __eq__ (self, other):
        return self.width == other.width and self.height == other.height

    def __lt__ (self, other):
        return self.height < other.height if self.width == other.width else self.width < other.width
    
    def as_tuple (self):
        return (int (self.width), int (self.height))


#--------------------------------------------------------------------------
# CLASS Line2d
# 
# Two dimensional line
#
class Line2d:
    
    def __init__ (self, p0, p1):
        self.p0 = p0
        self.p1 = p1
        
    #--------------------------------------------------------------------------
    # Return length of the line
    #       
    def length (self):
        diff = self.p1 - self.p0 + Point2d (1, 1)
        return math.sqrt (diff.x * diff.x + diff.y * diff.y)
        
    def __repr__ (self):
        return 'Line2d ({0}, {1})'.format (self.p0, self.p1)

    def __eq__ (self, other):
        return self.p0 == other.p0 and self.p1 == other.p1
    
    def as_tuple (self):
        return (self.p0.as_tuple (), self.p1.as_tuple ())

    
    #--------------------------------------------------------------------------
    # Return othogonal variant of this line
    #
    # The line is rotated counter clockwise around self.p0
    #
    # @return Rotated line
    #
    def orthogonal (self):
        direction = self.p1 - self.p0
        direction = Point2d (-direction.y, direction.x)
        return Line2d (self.p0, self.p0 + direction)
    
    #--------------------------------------------------------------------------
    # Angle of the line relative to the x axis (counterclockwise)
    #
    # @return Angle relative to the x axis in [0:2*PI[
    #
    def angle (self):
        angle = math.atan2 (self.p1.y - self.p0.y, self.p1.x - self.p0.x)
        if angle < 0:
            angle = 2 * math.pi + angle
        return angle


#--------------------------------------------------------------------------
# CLASS Rect2d
# 
# Two dimensional rectangle
#
class Rect2d:
    
    def __init__ (self, top_left, bottom_right):
        if type (bottom_right) is Size2d:
            bottom_right = top_left + bottom_right - Point2d (1, 1)        

        self.p0 = top_left
        self.p1 = Point2d (bottom_right.x, top_left.y)
        self.p2 = bottom_right
        self.p3 = Point2d (top_left.x, bottom_right.y)

    def __add__ (self, offset):
        return Rect2d (self.p0, self.size () + offset)

    #
    # Return size of the rectangle
    #       
    def size (self):
        return Size2d (self.p2.x - self.p0.x + 1, self.p2.y - self.p0.y + 1)
        

    #--------------------------------------------------------------------------
    # Resize rectangle while keeping top left position
    #
    # @param size New rectangle size
    # @return Resulting rectangle
    #
    def resized (self, size):
        return Rect2d (self.p0, self.p0 + size - Size2d (1, 1))
        
    #--------------------------------------------------------------------------
    # Compute the combined area of this and another rectangle
    #
    # @param rect1 Segment rect 1
    # @param rect2 Segment rect 2
    # @return Resulting rectangle covering the bounding box
    #
    def expanded (self, rect):
        return Rect2d (Point2d (min (rect.p0.x, self.p0.x), min (rect.p0.y, self.p0.y)),
                       Point2d (max (rect.p2.x, self.p2.x), max (rect.p2.y, self.p2.y)))
                
    #--------------------------------------------------------------------------
    # Move rectangle to a position without changing its size
    #
    # @param pos Position to move the top left corner of the rectangle to
    # @return Rectangle at new position
    #
    def move_to (self, pos):
        return Rect2d (pos, self.size ())
                
    #--------------------------------------------------------------------------
    # Return center coordinate of rectangle
    #
    def center (self):
        return self.p0 + (self.size () - Size2d (1, 1)) / 2
                
    def __repr__ (self):
        return 'Rect2d ({0}, {1})'.format (self.p0, self.p2)

    def __eq__ (self, other):
        return self.p0 == other.p0 and self.p2 == other.p2
    
    def as_tuple (self):        
        return (int (self.p0.x), int (self.p0.y), int (self.p2.x), int (self.p2.y))



#--------------------------------------------------------------------------
# CLASS Ellipse2d
# 
# Two dimensional ellipse
#
class Ellipse2d:
    
    def __init__ (self, center, radius=None):
        
        if radius is None:
            rect = center
            self.center = rect.center ()
            self.radius = Point2d (rect.size () - Size2d (1, 1)) / 2
        else:        
            self.center = center
            self.radius = radius

    #
    # Return bounding rectangle
    #       
    def rect (self):
        return Rect2d (self.center - Point2d (self.radius.x, self.radius.y),
                       self.center + Point2d (self.radius.x, self.radius.y))
        
    def __repr__ (self):
        return 'Ellipse2d (center={0}, radius=({1}, {2}))'.format (self.center, self.radius.x, self.radius.y)

    def __eq__ (self, other):
        return self.center == other.center and self.radius == other.radius
    
    def as_tuple (self):        
        return self.rect ().as_tuple ()



#
# Unittest for the geometry classes
#
class TestGeometry (unittest.TestCase):

    def test_Point2d (self):
        self.assertEqual (Point2d (1, 2) + Point2d (5, 6), Point2d (1 + 5, 2 + 6))
        self.assertEqual (Point2d (1, 2) - Point2d (5, 6), Point2d (1 - 5, 2 - 6))
        self.assertEqual (Point2d (1, 2) * 4, Point2d (1 * 4, 2 * 4))
        self.assertEqual (Point2d (1, 2) / 4, Point2d (1 / 4, 2 / 4))
        self.assertEqual (Point2d (5, 6).as_tuple (), (5, 6))

    def test_Size2d (self):
        self.assertEqual (Size2d (1, 2) + Size2d (5, 6), Size2d (1 + 5, 2 + 6))
        self.assertEqual (Size2d (1, 2) - Size2d (5, 6), Size2d (1 - 5, 2 - 6))
        self.assertEqual (Size2d (1, 2) * 4, Size2d (1 * 4, 2 * 4))
        self.assertEqual (Size2d (1, 2) / 4, Size2d (1 / 4, 2 / 4))
        self.assertEqual (Size2d (5, 6).as_tuple (), (5, 6))

    def test_Line2d (self):
        self.assertEqual (Line2d (Point2d (5, 5), Point2d (7, 10)).length (), math.sqrt ((3*3 + 6*6)))
        self.assertEqual (Line2d (Point2d (0, 0), Point2d (5, 5)).orthogonal (), Line2d (Point2d (0, 0), Point2d (-5, 5)))
        self.assertEqual (Line2d (Point2d (2, 3), Point2d (5, 5)).orthogonal (), Line2d (Point2d (2, 3), Point2d (2 + -2, 3 + 3)))
        
        self.assertEqual (Line2d (Point2d (1, 2), Point2d (6, 2)).angle () * 180.0 / math.pi, 0.0)
        self.assertEqual (Line2d (Point2d (3, 4), Point2d (3, 9)).angle () * 180.0 / math.pi, 90.0)
        self.assertAlmostEqual (Line2d (Point2d (0, 0), Point2d (-5, 1)).angle () * 180.0 / math.pi, 169.0, places=0)
        self.assertAlmostEqual (Line2d (Point2d (0, 0), Point2d (-5, -1)).angle () * 180.0 / math.pi, 191.0, places=0)
        self.assertAlmostEqual (Line2d (Point2d (0, 0), Point2d (5, 1)).angle () * 180.0 / math.pi, 11.0, places=0)
        self.assertAlmostEqual (Line2d (Point2d (0, 0), Point2d (5, -1)).angle () * 180.0 / math.pi, 349.0, places=0)
        
    def test_Rect2d (self):
        r1 = Rect2d (Point2d (0, 0), Point2d (10, 10))
        
        self.assertEqual (r1.p0, Point2d (0, 0))
        self.assertEqual (r1.p1, Point2d (10, 0))
        self.assertEqual (r1.p2, Point2d (10, 10))
        self.assertEqual (r1.p3, Point2d (0, 10))        
        
        self.assertEqual (r1.size (), Size2d (11, 11))
        self.assertEqual (r1.center (), Point2d (5, 5))

        r2 = Rect2d (Point2d (1, 2), Size2d (10, 10))

        self.assertEqual (r2.p2, Point2d (10, 11))
        self.assertEqual (r2.size (), Size2d (10, 10))
        self.assertEqual (r2.center (), Point2d (5.5, 6.5))
        self.assertEqual (r2.as_tuple (), (1, 2, 10, 11))

        r3 = Rect2d (Point2d (3, 5), Point2d (10, 12))
        r4 = Rect2d (Point2d (4, 2), Point2d (11, 13))
        
        self.assertEqual (r3.expanded (r4), Rect2d (Point2d (3, 2), Point2d (11, 13))) 
        self.assertEqual (r4.expanded (r3), Rect2d (Point2d (3, 2), Point2d (11, 13)))
        
        r5 = Rect2d (Point2d (0, 0), Point2d (9, 9))
        
        self.assertEqual (r5.size (), Size2d (10, 10))
        self.assertEqual (r5 + Size2d (3, 5), Rect2d (Point2d (0, 0), Size2d (13, 15)))

    def test_Ellipse2d (self):
        e1 = Ellipse2d (Point2d (2, 2), Point2d (2, 2))
        
        self.assertEqual (e1.center, Point2d (2, 2))
        self.assertEqual (e1.radius, Point2d (2, 2))
        self.assertEqual (e1.rect (), Rect2d (Point2d (0, 0), Point2d (4, 4)))
        
        e2 = Ellipse2d (Point2d (5, 10), Point2d (2, 3))
        
        self.assertEqual (e2.center, Point2d (5, 10))
        self.assertEqual (e2.radius, Point2d (2, 3))
        self.assertEqual (e2.rect (), Rect2d (Point2d (3, 7), Point2d (7, 13)))
        
        r = Rect2d (Point2d (0, 0), Point2d (4, 4))
        self.assertEqual (r.size (), Size2d (5, 5))
        self.assertEqual (r.center (), Point2d (2, 2))
        
        e3 = Ellipse2d (r)
        
        self.assertEqual (e3.center, Point2d (2, 2))
        self.assertEqual (e3.radius, Point2d (2, 2))
        self.assertEqual (e3.rect (), r)

        for _ in range (100):
            r = Rect2d (Point2d (random.randint (0, 100), random.randint (0, 100)),
                        Point2d (random.randint (10, 100), random.randint (10, 100)))
            
            e = Ellipse2d (r)
            self.assertEqual (e.rect (), r)

        
if __name__ == '__main__':
    unittest.main ()
