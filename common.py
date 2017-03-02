#!/usr/bin/python3
#
# common.py - Common data structures
#
# Frank Blankenburg, Mar. 2017
#


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

