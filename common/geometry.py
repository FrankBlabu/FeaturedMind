#!/usr/bin/python3
#
# geometry.py - Data structures for geometry operations
#
# Frank Blankenburg, Mar. 2017
#

import unittest


#--------------------------------------------------------------------------
# CLASS Point2d
# 
# Two dimensional point
#
class Point2d:
    
    def __init__ (self, x, y=None):
        if type (x) == Size2d and y == None:
            self.x = x.width
            self.y = x.height
        else:
            assert type (x) == float or type (x) == int
            assert type (y) == float or type (y) == int
            
            self.x = x
            self.y = y
        
    def __add__ (self, other):
        assert type (self) is Point2d
        assert type (other) is Point2d or type (other) is Size2d
        
        return Point2d (self.x + other.x, self.y + other.y) if type (other) is Point2d else Point2d (self.x + (other.width - 1), self.y + (other.height - 1))
            
    def __sub__ (self, other):
        assert type (self) == Point2d
        assert type (other) is Point2d or type (other) is Size2d
        
        return Point2d (self.x - other.x, self.y - other.y) if type (other) is Point2d else Point2d (self.x - (other.width - 1), self.y - (other.height - 1))
    
    def __mul__ (self, other):
        assert type (self) == Point2d
        assert type (other) == int or type (other) == float
        
        return Point2d (self.x * other, self.y * other)
        
    def __rmul__ (self, other):
        assert type (self) == Point2d
        assert type (other) == int or type (other) == float
        
        return Point2d (self.x * other, self.y * other)

    def __truediv__ (self, other):
        assert type (self) == Point2d
        assert type (other) == int or type (other) == float
        
        return Point2d (self.x / other, self.y / other) 

    def __not__ (self):
        self.x = -self.x
        self.y = -self.y
            
    def __eq__ (self, other):
        assert type (other) is Point2d
        return self.x == other.x and self.y == other.y

    def __lt__ (self, other):
        assert type (other) is Point2d
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
        self.width = width
        self.height = height
        
    def __add__ (self, other):
        assert type (self) == Size2d
        assert type (other) == Size2d or type (other) == Point2d
        
        return Size2d (self.width + other.width, self.height + other.height) if type (other) == Size2d else Size2d (self.width + other.x, self.height + other.y) 
            
    def __sub__ (self, other):
        assert type (self) == Size2d
        assert type (other) == Size2d or type (other) == Point2d

        return Size2d (self.width - other.width, self.height - other.height) if type (other) == Size2d else Size2d (self.width - other.x, self.height - other.y) 
    
    def __mul__ (self, other):
        assert type (self) == Size2d
        assert type (other) == int or type (other) == float
        
        return Size2d (self.width * other, self.height * other)
        
    def __rmul__ (self, other):
        assert type (self) == Size2d
        assert type (other) == int or type (other) == float
        
        return Size2d (self.width * other, self.height * other)

    def __truediv__ (self, other):
        assert type (self) == Size2d
        assert type (other) == int or type (other) == float
        
        return Size2d (self.width / other, self.height / other) 

    def __repr__ (self):
        return 'Size2d ({0}, {1})'.format (self.width, self.height)

    def __eq__ (self, other):
        assert type (other) is Size2d
        return self.width == other.width and self.height == other.height

    def __lt__ (self, other):
        assert type (other) is Size2d
        return self.height < other.height if self.width == other.width else self.width < other.width
    
    def as_tuple (self):
        return (int (self.width), int (self.height))


#--------------------------------------------------------------------------
# CLASS Rect2d
# 
# Two dimensional rectangle
#
class Rect2d:
    
    def __init__ (self, top_left, bottom_right):
        assert type (top_left) is Point2d
        assert type (bottom_right) is Point2d or type (bottom_right) is Size2d

        if type (bottom_right) is Size2d:
            bottom_right = top_left + bottom_right        

        self.p0 = top_left
        self.p1 = Point2d (bottom_right.x, top_left.y)
        self.p2 = bottom_right
        self.p3 = Point2d (top_left.x, bottom_right.y)

    #
    # Return size of the rectangle
    #       
    def size (self):
        return Size2d (self.p2.x - self.p0.x + 1, self.p2.y - self.p0.y + 1)
        
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
        return Point2d ((self.p0.x + self.size ().width) / 2, (self.p0.y + self.size ().height) / 2)
                
    def __repr__ (self):
        return 'Rect2d ({0}, {1})'.format (self.p0, self.p2)

    def __eq__ (self, other):
        assert type (other) is Rect2d
        return self.p0 == other.p0 and self.p2 == other.p2
    
    def as_tuple (self):
        return (self.p0.as_tuple (), self.p2.as_tuple ())



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
        
    def test_Rect2d (self):
        r1 = Rect2d (Point2d (0, 0), Point2d (10, 10))
        
        self.assertEqual (r1.p0, Point2d (0, 0))
        self.assertEqual (r1.p1, Point2d (10, 0))
        self.assertEqual (r1.p2, Point2d (10, 10))
        self.assertEqual (r1.p3, Point2d (0, 10))        
        self.assertEqual (r1.size (), Size2d (11, 11))

        r2 = Rect2d (Point2d (0, 0), Size2d (10, 10))

        self.assertEqual (r2.p2, Point2d (9, 9))
        self.assertEqual (r2.size (), Size2d (10, 10))
        self.assertEqual (r2.as_tuple (), ((0, 0), (9, 9)))

        r3 = Rect2d (Point2d (3, 5), Point2d (10, 12))
        r4 = Rect2d (Point2d (4, 2), Point2d (11, 13))
        
        self.assertEqual (r3.expanded (r4), Rect2d (Point2d (3, 2), Point2d (11, 13))) 
        self.assertEqual (r4.expanded (r3), Rect2d (Point2d (3, 2), Point2d (11, 13))) 

        
if __name__ == '__main__':
    unittest.main ()
