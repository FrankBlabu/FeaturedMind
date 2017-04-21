#!/usr/bin/python3
#
# log.py - Logging functions
#
# Frank Blankenburg, Apr. 2017
#

import os.path
import warnings
import xml.etree.ElementTree as ET
import xml.dom.minidom

import numpy as np

import skimage.color
import skimage.io


#--------------------------------------------------------------------------
# CLASS NoLogger
#
class NoLogger:
    '''
    Logger which does not log anything. Meant to be a stub for 'with'
    statements.
    '''

    def __init__ (self):
        pass
    
    def __enter__ (self):
        return self
    
    def __exit__ (self, exc_type, exc_value, exc_tb):
        pass
        
    def add_caption (self, text):
        pass
            
    def add_table (self, rows):
        pass
    
    def add_content (self, element, content):
        pass

    def add_image (self, image, element=None):
        pass
    
            
#--------------------------------------------------------------------------
# CLASS HTMLLogger
#
class HTMLLogger:
    '''
    This class is used to generate HTML based logs. It can list strings and
    images in a HTML table and add various resources to a log directory so 
    that the resultin HTML page is complete and well formatted.
    '''

    def __init__ (self, directory, title, header):
        '''
        Initialize logger
        
        @param directory Directory the log and the resources files are written
                         into. If not existing, the directory will be created.
        @param title     Log title
        @param header    List of header cells
        '''
        
        self.directory = os.path.abspath (directory)
        self.header = header
        self.resource_counter = 0
        
        if not os.path.exists (self.directory):
            os.makedirs (self.directory)
        
        self.document = ET.Element ('html')
        head = ET.SubElement (self.document, 'head')
        
        if title:
            title_tag = ET.SubElement (head, 'title')
            title_tag.text = title
        
        self.body = ET.SubElement (self.document, 'body')
        
        if title:
            caption= ET.SubElement (self.body, 'h1')
            caption.text = title

    def __enter__ (self):
        return self
    
    def __exit__ (self, exc_type, exc_value, exc_tb):
        '''
        Exit 'with' scope. The document is written in this case.
        '''
        if self.document:
            file = open (os.path.join (self.directory, 'index.html'), 'w')
        
            text = ET.tostring (self.document, 'utf-8')
            file.write (xml.dom.minidom.parseString (text).toprettyxml (indent='  '))
            
            file.flush ()
            file.close ()
        
    def add_caption (self, text):
        '''
        Add caption to document body
        
        @param text Caption text to add
        '''
        
        caption = ET.SubElement (self.body, 'h2')
        caption.text = caption
            
    def add_table (self, rows):
        '''
        Add HTML table to log
        
        @param rows    Table rows
        '''
        
        table = ET.SubElement (self.body, 'table')        
        
        for row in rows:
            tr = ET.SubElement (table, 'tr')
        
            for cell in row:
                td = ET.SubElement (tr, 'td')
                self.add_content (td, cell) 
    
    
    def add_content (self, element, content):
        '''
        Add content (text, image, ...) to document
        
        @param element Element the content is added to
        @param content Content to be added. Can be a string or a numpy array representing an image.
        '''        
        
        if isinstance (content, str):
            element.text = content
            
        elif isinstance (content, np.ndarray):
            self.add_image (content, element=element)
            
        else:
            raise Exception ("Unsupported data type '{0}'".format (type (content))) 
        

    def add_image (self, image, element=None):
        '''
        Add image to document
        
        @param image   Image to add (numpy array format)
        @param element Element the image is added to (body by default)
        '''

        if element is None:
            element = self.body

        file = 'image_{0}.png'.format (self.resource_counter)
        self.resource_counter += 1
            
        ET.SubElement (element, 'img', {'src': './' + file})
            
        if len (image.shape) == 3:
            image = image.reshape ((image.shape[0], image.shape[1]))            
            image = skimage.color.gray2rgb (image)
            
            with warnings.catch_warnings ():
                warnings.simplefilter ('ignore')
                skimage.io.imsave (os.path.join (self.directory, file), skimage.img_as_uint (image))
                
                

