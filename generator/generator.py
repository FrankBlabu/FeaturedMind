#!/usr/bin/python3
#
# generator.py - Abstract base class for all generator classes
#
# Frank Blankenburg, Oct. 2017
#

from abc import ABC, abstractmethod


#----------------------------------------------------------------------------------------------------------------------
# CLASS Generator
#
# Abstract base class for all generator classes
#
class Generator (ABC):

    #
    # Generate a new image instance
    #
    # @return Tuple consisting of the image itself and a mask marking all relevant image points. The second
    #         return parameter can be 'None'  if there is no relevant mask and all pixels are valid, which is
    #         especially the case for background images.
    #
    @abstractmethod
    def generate (self):
        pass
